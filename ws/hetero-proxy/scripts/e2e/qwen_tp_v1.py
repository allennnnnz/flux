################################################################################
# hetero-proxy E2E (2026-10-14): minimal tensor-parallel Qwen2.5-32B-Instruct inference with real weights, the
# substrate for the A / B / C end-to-end comparison (STATUS section 0). Configuration A = this engine as is:
# TP over all 8 A100s, NCCL all-reduce after o_proj and down_proj (the standard Megatron split), eager PyTorch.
#   column-parallel: fused q|k|v (5 q heads + 1 kv head per rank at TP8, with bias), fused gate|up
#   row-parallel:    o_proj, down_proj  -> torch.distributed.all_reduce (NCCL)
#   replicated:      embeddings, norms, lm_head (only the last position of each sequence is projected)
#   numerics follow HF Qwen2 (fp32 RMSNorm then bf16 weight multiply, fp32 RoPE tables cast to bf16, SDPA attention)
# Batches here are fixed-shape: B sequences of equal prompt length T (left-padding is not needed); decode keeps a
# preallocated KV cache. No CUDA graphs (eager): decode includes Python launch overhead, identically for A / B / C.
# Modes:
#   check  greedy-generate the prompts of results/e2e/hf_reference.json and compare tokens / logits with HF
#   bench  fixed batches: prefill time and decode per-token time, CUDA events on rank 0, interleaved repeats
# Usage (8 ranks; the side venv adds safetensors / transformers to the pixi torch):
#   env -u CUDA_DEVICE_MAX_CONNECTIONS CUDA_MODULE_LOADING=EAGER pixi run --manifest-path pixi.toml \
#       ~/.venvs/hetero-proxy-hf/bin/python -m torch.distributed.run --nproc_per_node 8 \
#       ws/hetero-proxy/scripts/e2e/qwen_tp_v1.py --mode check|bench --out <dir>
################################################################################
import argparse
import glob
import json
import math
import os
import statistics
import sys
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from safetensors import safe_open

MODEL = "Qwen/Qwen2.5-32B-Instruct"


def model_dir():
    hits = glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/models--{MODEL.replace('/', '--')}/snapshots/*/"))
    assert hits, "model not downloaded"
    return hits[0]


class Cfg:
    def __init__(self, d):
        c = json.load(open(os.path.join(d, "config.json")))
        self.H, self.L, self.NQ, self.NKV = c["hidden_size"], c["num_hidden_layers"], c["num_attention_heads"], \
            c["num_key_value_heads"]
        self.FFN, self.V, self.eps, self.theta = c["intermediate_size"], c["vocab_size"], c["rms_norm_eps"], \
            c["rope_theta"]
        self.HD = self.H // self.NQ


def rmsnorm(x, w, eps):
    """HF Qwen2RMSNorm: fp32 normalisation, cast back, multiply by the bf16 weight."""
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return w * xf.to(x.dtype)


def rotate_half(x):
    a, b = x.chunk(2, dim=-1)
    return torch.cat((-b, a), dim=-1)


class QwenTP:
    def __init__(self, rank, world, dev, max_batch, max_len, layers=None):
        self.rank, self.world, self.dev = rank, world, dev
        d = model_dir()
        self.cfg = c = Cfg(d)
        assert c.NQ % world == 0 and c.NKV % world == 0 and c.FFN % world == 0
        self.nq, self.nkv, self.ffn = c.NQ // world, c.NKV // world, c.FFN // world
        self.L = layers or c.L
        idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
        files = {}

        def get(name, rows=None, cols=None):
            fn = idx[name]
            if fn not in files:
                files[fn] = safe_open(os.path.join(d, fn), framework="pt", device="cpu")
            sl = files[fn].get_slice(name)
            if rows is not None:
                t = sl[rows[0]:rows[1]]
            elif cols is not None:
                t = sl[:, cols[0]:cols[1]]
            else:
                t = sl[:]
            return t.to(dev, non_blocking=False).contiguous()
        r, hd = rank, c.HD
        qr, kr = (r * self.nq * hd, (r + 1) * self.nq * hd), (r * self.nkv * hd, (r + 1) * self.nkv * hd)
        fr = (r * self.ffn, (r + 1) * self.ffn)
        self.embed = get("model.embed_tokens.weight")
        self.layers = []
        for i in range(self.L):
            p = f"model.layers.{i}."
            wqkv = torch.cat([get(p + "self_attn.q_proj.weight", qr), get(p + "self_attn.k_proj.weight", kr),
                              get(p + "self_attn.v_proj.weight", kr)])
            bqkv = torch.cat([get(p + "self_attn.q_proj.bias", qr), get(p + "self_attn.k_proj.bias", kr),
                              get(p + "self_attn.v_proj.bias", kr)])
            self.layers.append(dict(
                ln1=get(p + "input_layernorm.weight"), ln2=get(p + "post_attention_layernorm.weight"),
                wqkv=wqkv, bqkv=bqkv, wo=get(p + "self_attn.o_proj.weight", cols=qr),
                wgu=torch.cat([get(p + "mlp.gate_proj.weight", fr), get(p + "mlp.up_proj.weight", fr)]),
                wd=get(p + "mlp.down_proj.weight", cols=fr)))
        self.norm = get("model.norm.weight")
        self.lm_head = get("lm_head.weight")
        files.clear()
        inv = 1.0 / (c.theta ** (torch.arange(0, hd, 2, dtype=torch.int64).float().to(dev) / hd))
        t = torch.arange(max_len, device=dev).float()
        emb = torch.cat([torch.outer(t, inv)] * 2, dim=-1)
        self.cos, self.sin = emb.cos().to(torch.bfloat16), emb.sin().to(torch.bfloat16)
        self.max_batch, self.max_len = max_batch, max_len
        shape = (self.L, max_batch, self.nkv, max_len, hd)
        self.kc = torch.empty(shape, dtype=torch.bfloat16, device=dev)
        self.vc = torch.empty(shape, dtype=torch.bfloat16, device=dev)
        self.final_norm = self._final_norm_gpu  # configuration B replaces this hook

    def _final_norm_gpu(self, x_last):
        return rmsnorm(x_last, self.norm, self.cfg.eps)

    def layer(self, i, x, pos0):
        c, w = self.cfg, self.layers[i]
        B, T, _ = x.shape
        hd, nq, nkv = c.HD, self.nq, self.nkv
        h = rmsnorm(x, w["ln1"], c.eps)
        qkv = F.linear(h, w["wqkv"], w["bqkv"])
        q, k, v = qkv.split([nq * hd, nkv * hd, nkv * hd], dim=-1)
        q = q.view(B, T, nq, hd).transpose(1, 2)
        k = k.view(B, T, nkv, hd).transpose(1, 2)
        v = v.view(B, T, nkv, hd).transpose(1, 2)
        cos, sin = self.cos[pos0:pos0 + T], self.sin[pos0:pos0 + T]
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        self.kc[i, :B, :, pos0:pos0 + T] = k
        self.vc[i, :B, :, pos0:pos0 + T] = v
        kk, vv = self.kc[i, :B, :, :pos0 + T], self.vc[i, :B, :, :pos0 + T]
        a = F.scaled_dot_product_attention(q, kk, vv, is_causal=(T > 1 and pos0 == 0), enable_gqa=True,
                                           scale=1.0 / math.sqrt(hd))
        o = F.linear(a.transpose(1, 2).reshape(B, T, nq * hd), w["wo"])
        if self.world > 1:
            dist.all_reduce(o)
        x = x + o
        h = rmsnorm(x, w["ln2"], c.eps)
        g, u = F.linear(h, w["wgu"]).split([self.ffn, self.ffn], dim=-1)
        d = F.linear(F.silu(g) * u, w["wd"])
        if self.world > 1:
            dist.all_reduce(d)
        return x + d

    @torch.inference_mode()
    def forward(self, ids, pos0):
        """ids [B, T] on device; returns logits of the last position [B, V] (fp32)."""
        x = F.embedding(ids, self.embed)
        for i in range(self.L):
            x = self.layer(i, x, pos0)
        xl = self.final_norm(x[:, -1])
        return F.linear(xl, self.lm_head).float()

    @torch.inference_mode()
    def generate(self, ids, n_new):
        """Greedy; ids [B, T] equal-length prompts. Returns (tokens [B, n_new], first-step logits)."""
        B, T = ids.shape
        logits = self.forward(ids, 0)
        first = logits
        out = []
        nxt = logits.argmax(-1)
        for s in range(n_new):
            out.append(nxt)
            if s == n_new - 1:
                break
            logits = self.forward(nxt[:, None], T + s)
            nxt = logits.argmax(-1)
        return torch.stack(out, 1), first


def init():
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    lr = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(lr)
    return rank, world, torch.device(f"cuda:{lr}")


def mode_check(m, rank, args):
    ref = json.load(open(args.reference))
    rows = []
    for item in ref["items"]:
        ids = torch.tensor([item["prompt_ids"]], device=m.dev)
        toks, first = m.generate(ids, len(item["hf_tokens"]))
        toks = toks[0].tolist()
        hf = item["hf_tokens"]
        n_same = next((i for i, (a, b) in enumerate(zip(toks, hf)) if a != b), len(hf))
        hl = torch.tensor(item["hf_first_logits_top"], device=m.dev)  # [[id, logit], ...] HF top-20 at step 0
        diff = (first[0, hl[:, 0].long()] - hl[:, 1]).abs().max().item()
        rows.append(dict(name=item["name"], tokens=len(hf), identical_prefix=n_same, all_identical=toks == hf,
                         first_step_top20_max_abs_logit_diff=diff, tp_tokens=toks))
        if rank == 0:
            print(f"[check] {item['name']}: identical first {n_same}/{len(hf)} tokens; step-0 top-20 max |dlogit| "
                  f"{diff:.3f}", flush=True)
    if rank == 0:
        with open(os.path.join(args.out, "check_tp.json"), "w") as f:
            json.dump(dict(world=m.world, rows=rows), f, indent=1)


def mode_bench(m, rank, args):
    """Fixed batches. prefill: B x T prompt; decode: per-token time over `decode_steps` steps after a T-token prompt.
    Shapes are interleaved round by round; rank 0 records CUDA events; median over rounds."""
    g = torch.Generator().manual_seed(0)
    shapes = [tuple(int(v) for v in s.split("x")) for s in args.prefill.split(",")]
    dshapes = [int(b) for b in args.decode.split(",")]
    res = {f"prefill_{b}x{t}": [] for b, t in shapes}
    res.update({f"decode_b{b}": [] for b in dshapes})
    ids_p = {(b, t): torch.randint(0, m.cfg.V, (b, t), generator=g).to(m.dev) for b, t in shapes}
    ids_d = {b: torch.randint(0, m.cfg.V, (b, args.decode_prompt), generator=g).to(m.dev) for b in dshapes}
    for _ in range(2):  # warm-up
        for (b, t), ids in ids_p.items():
            m.forward(ids, 0)
        for b, ids in ids_d.items():
            m.generate(ids, 4)
    torch.cuda.synchronize()
    for r in range(args.rounds):
        for (b, t), ids in ids_p.items():
            dist.barrier()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            m.forward(ids, 0)
            e1.record()
            e1.synchronize()
            res[f"prefill_{b}x{t}"].append(e0.elapsed_time(e1))
        for b, ids in ids_d.items():
            dist.barrier()
            logits = m.forward(ids, 0)
            nxt = logits.argmax(-1)
            torch.cuda.synchronize()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            for s in range(args.decode_steps):
                nxt = m.forward(nxt[:, None], args.decode_prompt + s).argmax(-1)
            e1.record()
            e1.synchronize()
            res[f"decode_b{b}"].append(e0.elapsed_time(e1) / args.decode_steps)
    if rank == 0:
        summ = {k: dict(median_ms=statistics.median(v), p25=sorted(v)[len(v) // 4], p75=sorted(v)[3 * len(v) // 4],
                        rounds=len(v)) for k, v in res.items()}
        for k, s in summ.items():
            print(f"[bench] {k:<18} median {s['median_ms']:9.3f} ms  (p25 {s['p25']:.3f}, p75 {s['p75']:.3f}, "
                  f"{s['rounds']} rounds)", flush=True)
        with open(os.path.join(args.out, f"bench_{args.tag}.json"), "w") as f:
            json.dump(dict(config=args.tag, world=m.world, raw_ms=res, summary=summ, args=vars(args)), f, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["check", "bench"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--reference", default=None)
    ap.add_argument("--tag", default="A")
    ap.add_argument("--max_batch", type=int, default=64)
    ap.add_argument("--max_len", type=int, default=4352)
    ap.add_argument("--prefill", default="1x1024,8x1024,4x4096")
    ap.add_argument("--decode", default="1,8,32,64")
    ap.add_argument("--decode_prompt", type=int, default=512)
    ap.add_argument("--decode_steps", type=int, default=64)
    ap.add_argument("--rounds", type=int, default=20)
    args = ap.parse_args()
    rank, world, dev = init()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    m = QwenTP(rank, world, dev, args.max_batch, args.max_len)
    dist.barrier()
    if rank == 0:
        print(f"[load] {time.time() - t0:.1f} s, world {world}, mem {torch.cuda.memory_allocated() / 2**30:.1f} GiB",
              flush=True)
    if args.mode == "check":
        mode_check(m, rank, args)
    else:
        mode_bench(m, rank, args)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
