################################################################################
# hetero-proxy E2E v4 (2026-10-14): v3 with fused Triton kernels (fused_kernels.py: add+RMSNorm, RoPE+KV write,
# decode attention, silu*up) in decode and prefill - nsys showed ~3850 kernels per v3 decode step, half of the GPU time
# in small elementwise kernels. Configurations and measurement as v3.
# (v3 header follows) v2 + decode controls AsplitF / AsplitM + per-round progress. (v2 run1 showed
# Bmid decode FASTER than A: the graph split itself changes the step time, so each B gets a no-hop control with the
# same split.) The CPU backend's rmsnorm is now a C / OpenMP kernel (hxb/csrc/hxbdma.c).
# Qwen2.5-32B-Instruct TP8 (NCCL) with CUDA-graph decode, plus the PCIe hop to
# the CPU "chip" through hxb (STATUS section 0, E2E plan). Configurations in ONE process, interleaved:
#   A        8 GPUs, NVLink / NCCL only. Decode: one CUDA graph per step (embed -> 64 layers -> norm -> lm_head).
#   Bfinal   the final RMSNorm runs on the CPU: rank 0 sends the last position of each sequence (B x 5120 bf16) over
#            PCIe (hxb Outbound, GPU0 copy engine -> host staging -> CPU device), the CPU normalises, hxb Inbound
#            brings it back to GPU0, NCCL broadcasts it (NVLink) to the 8 ranks, lm_head follows.
#            Decode: graph1 (embed -> 64 layers) | hop | graph2 (broadcast -> lm_head -> argmax).
#   BmidN    layer 33's input RMSNorm runs on the CPU for ALL tokens: after layer 32, rank 0 streams the hidden
#            states to the CPU in N row chunks (Flux-style chunk overlap: D2H / CPU / H2D / per-chunk NCCL
#            broadcast all in flight together), then layer 33 continues from the normalised input on every rank.
#            Bmid1 = the same without chunking (level 0, ablation). Decode moves B rows: always one chunk.
#            Decode: graph1 (embed -> 32 layers) | hop | graph2 (broadcast -> layer 33 from the CPU's output -> 64).
#   AsplitF  decode control for Bfinal: the same two graphs, final norm on the GPU, no hop, no broadcast.
#   AsplitM  decode control for Bmid: graph1 (32 layers) | graph2 (layers 33-64 + head), no hop, no broadcast.
#            (prefill of A / AsplitF / AsplitM is the same code; Asplit* are run for decode only.)
# The CPU device (hxb CpuBackend, NUMA node 1) is created by rank 0 only. Device-side commands of a whole run are
# posted before it starts (Outbound / Inbound post_device), so command transport is off the critical path; the GPU
# side is posted step by step. Ranks are pinned to NUMA-0 cores (rank r -> cpus 4r..4r+3), the same for every
# configuration. Decode attention: SDPA over a fixed KV window Lb with a position mask (graph-safe).
# Modes:
#   check  greedy-generate the HF reference prompts with every configuration; tokens vs HF and vs each other
#   bench  fixed batches, configurations interleaved round by round: prefill (CUDA events, rank 0) and decode per
#          token (graph replays + hops, CUDA events on rank 0's compute stream)
# PRE-REGISTERED for v3 run2 (written before it; run1 = v2 numbers in results/e2e/bench_v2/run1) [推論]:
#   S1 decode: Bfinal - AsplitF and Bmid* - AsplitM are both > 0 (the hop costs time) and below 1 ms per token.
#   S2 decode: A vs AsplitF / AsplitM differ by more than the run-to-run spread (the split effect seen in run1).
#   S3 prefill Bmid1 - A at 8x1024 drops from 87 ms (run1) to <= 30 ms with the C rmsnorm (8.7-10 ms for 8192 rows)
#      + D2H 3.5 + H2D 3.8 + device DMA ~8 + broadcast; Bmid8 - A at 8x1024 <= 15 ms.
#   S4 tokens of every configuration identical to A on the 4 HF prompts (check mode).
# Usage:
#   env -u CUDA_DEVICE_MAX_CONNECTIONS CUDA_MODULE_LOADING=EAGER pixi run --manifest-path pixi.toml \
#       ~/.venvs/hetero-proxy-hf/bin/python -m torch.distributed.run --nproc_per_node 8 \
#       ws/hetero-proxy/scripts/e2e/qwen_tp_v2.py --mode check|bench --out <dir> [--configs A,AsplitF,Bfinal,AsplitM,Bmid8,Bmid1]
################################################################################
import argparse
import json
import math
import os
import random
import statistics
import sys
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from qwen_tp_v1 import QwenTP, init, rmsnorm, rotate_half  # noqa: E402
from fused_kernels import add_rmsnorm, decode_attn, rope_cache, silu_mul  # noqa: E402

MID = 32  # 0-based index of the layer whose input RMSNorm runs on the CPU in Bmid (the 33rd layer)


class Hop:
    """Rank 0's PCIe path to the CPU device: RMSNorm of `rows` x H bf16 rows, in n chunks."""

    def __init__(self, m, max_rows, chunk_rows_list):
        from hxb.bridge import Bridge
        from hxb.cpu_backend import CpuBackend
        self.m, self.H = m, m.cfg.H
        self.be = CpuBackend()
        self.br = Bridge(self.be)
        be = self.be
        self.q_in, self.q_op, self.q_out = be.queue(), be.queue(), be.queue()
        self.dX = be.alloc((max_rows, self.H), torch.bfloat16)
        self.dY = be.alloc((max_rows, self.H), torch.bfloat16)
        self.w = {"final": self.br.upload(self.q_in, m.norm.cpu()),
                  "mid": self.br.upload(self.q_in, m.layers[MID]["ln1"].cpu())}
        self.computed, self.k = be.signal("computed"), 0
        self.ch = {cr: (self.br.outbound(m.dev.index, cr, self.H, torch.bfloat16, slots=4 if cr < max_rows else 1),
                        self.br.inbound(m.dev.index, cr, self.H, torch.bfloat16, slots=4 if cr < max_rows else 1))
                   for cr in chunk_rows_list}

    def post_device(self, which, rows, n):
        """Device half of one hop of `rows` rows in n chunks (posted ahead)."""
        cr = -(-rows // n)
        ob, ib = self.ch[self._chan(cr)]
        for r0 in range(0, rows, cr):
            r1 = min(rows, r0 + cr)
            sig, v = ob.post_device(self.dX, (r0, r1), r1 - r0, self.q_in)
            self.k += 1
            self.be.launch(self.q_op, "rmsnorm", {"x": self.dX, "w": self.w[which], "y": self.dY, "rows": (r0, r1),
                                                  "eps": self.m.cfg.eps}, wait=[(sig, v)], done=(self.computed, self.k))
            ib.post_device(self.dY, (r0, r1), r1 - r0, self.q_out, [(self.computed, self.k)])

    def _chan(self, cr):
        fit = [c for c in self.ch if c >= cr]
        return min(fit)

    def post_gpu(self, src, dst, rows, n, ready):
        """GPU half: src[0:rows] (contiguous 2-D GPU tensor) -> CPU -> dst[0:rows]; returns one CUDA event per chunk
        (recorded on the inbound stream when that chunk has landed in dst)."""
        cr = -(-rows // n)
        ob, ib = self.ch[self._chan(cr)]
        # the copies run on the channel streams: keep the caching allocator from reusing src / dst before they do
        src.record_stream(ob.stream)
        dst.record_stream(ib.stream)
        evs = []
        for r0 in range(0, rows, cr):
            r1 = min(rows, r0 + cr)
            ob.post_gpu(src, r0, r1, ready)
            ib.post_gpu(dst, r0, r1)
            e = torch.cuda.Event()
            e.record(ib.stream)
            evs.append(e)
        return evs

    def close(self):
        self.br.close()
        self.be.close()


class Engine(QwenTP):
    def __init__(self, rank, world, dev, max_batch, max_len, hop_rows=0, hop_chunks=(8,)):
        super().__init__(rank, world, dev, max_batch, max_len)
        self.arangeB = torch.arange(max_batch, device=dev)
        self.arangeL = torch.arange(max_len, device=dev)
        self.hop = None
        self.hop_rows = hop_rows
        if rank == 0 and hop_rows:
            crs = sorted({-(-hop_rows // n) for n in hop_chunks} | {max_batch})
            self.hop = Hop(self, hop_rows, crs)
        self.mid_buf = torch.empty(max(hop_rows, max_batch), self.cfg.H, dtype=torch.bfloat16, device=dev)
        self.pool = torch.cuda.graph_pool_handle()
        self.graphs = {}

    # ---------------- prefill (eager, fused norms / silu) ----------------
    def _layer_pre(self, i, x, h, B, T, nw):
        """x residual, h normalised input, both [B*T, H]. nw: weight of the next fused norm (None: residual only)."""
        c, w = self.cfg, self.layers[i]
        hd, nq, nkv = c.HD, self.nq, self.nkv
        qkv = F.linear(h, w["wqkv"], w["bqkv"])
        q, k, v = qkv.split([nq * hd, nkv * hd, nkv * hd], dim=-1)
        q = q.view(B, T, nq, hd).transpose(1, 2)
        k = k.view(B, T, nkv, hd).transpose(1, 2)
        v = v.view(B, T, nkv, hd).transpose(1, 2)
        cos, sin = self.cos[:T], self.sin[:T]
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        self.kc[i, :B, :, :T] = k
        self.vc[i, :B, :, :T] = v
        a = F.scaled_dot_product_attention(q, self.kc[i, :B, :, :T], self.vc[i, :B, :, :T], is_causal=T > 1,
                                           enable_gqa=True, scale=1 / math.sqrt(hd))
        o = F.linear(a.transpose(1, 2).reshape(B * T, nq * hd), w["wo"])
        dist.all_reduce(o)
        x, h2 = add_rmsnorm(x, w["ln2"], c.eps, r=o)
        d = F.linear(silu_mul(F.linear(h2, w["wgu"])), w["wd"])
        dist.all_reduce(d)
        if nw is None:
            return x + d, None
        return add_rmsnorm(x, nw, c.eps, r=d)

    def _hop_broadcast(self, which, x2d, n):
        """Rank 0: x2d -> CPU RMSNorm -> mid_buf, chunk by chunk; every rank: per-chunk NCCL broadcast of mid_buf."""
        rows = x2d.shape[0]
        out = self.mid_buf[:rows]
        cr = -(-rows // n)
        if self.rank == 0:
            ready = torch.cuda.Event()
            ready.record()
            evs = self.hop.post_gpu(x2d, out, rows, n, ready)
        for ci, r0 in enumerate(range(0, rows, cr)):
            if self.rank == 0:
                torch.cuda.current_stream().wait_event(evs[ci])
            dist.broadcast(out[r0:min(rows, r0 + cr)], src=0)
        return out

    def _next_w(self, i, cfg):
        """Weight of the fused norm after layer i, or None where the residual goes to the CPU instead."""
        if cfg.startswith("Bmid") and i + 1 == MID:
            return None
        if i + 1 < self.L:
            return self.layers[i + 1]["ln1"]
        return None if cfg == "Bfinal" else self.norm

    @torch.inference_mode()
    def prefill(self, ids, cfg):
        """cfg: 'A' | 'Bfinal' | 'Bmid<n>' (Asplit* = A). Returns last-position logits [B, V] (fp32)."""
        B, T = ids.shape
        if cfg.startswith("Asplit"):
            cfg = "A"  # identical prefill path
        if self.rank == 0 and cfg != "A":  # device side of this forward's hop, posted ahead
            if cfg == "Bfinal":
                self.hop.post_device("final", B, 1)
            else:
                self.hop.post_device("mid", B * T, int(cfg[4:]))
        x = F.embedding(ids, self.embed).view(B * T, -1)
        _, h = add_rmsnorm(x, self.layers[0]["ln1"], self.cfg.eps)
        for i in range(self.L):
            if cfg.startswith("Bmid") and i == MID:
                h = self._hop_broadcast("mid", x, int(cfg[4:]))
            x, h = self._layer_pre(i, x, h, B, T, self._next_w(i, cfg))
        if cfg == "Bfinal":
            xl = self._hop_broadcast("final", x.view(B, T, -1)[:, -1].contiguous(), 1)
        else:
            xl = h.view(B, T, -1)[:, -1]
        return F.linear(xl, self.lm_head).float()

    # ---------------- decode (CUDA graphs, fused kernels) ----------------
    def _layer_dec(self, i, x, h, pos, nw):
        """x residual, h normalised input, both [B, H]; returns (x', h') with h' = None when nw is None."""
        c, w = self.cfg, self.layers[i]
        B = x.shape[0]
        hd, nq, nkv = c.HD, self.nq, self.nkv
        qkv = F.linear(h, w["wqkv"], w["bqkv"])
        kc, vc = self.kc[i, :B], self.vc[i, :B]
        q = rope_cache(qkv, self.cos, self.sin, pos, kc, vc, nq, nkv, hd)
        a = decode_attn(q, kc, vc, pos, 1 / math.sqrt(hd))
        o = F.linear(a, w["wo"])
        dist.all_reduce(o)
        x, h2 = add_rmsnorm(x, w["ln2"], c.eps, r=o)
        d = F.linear(silu_mul(F.linear(h2, w["wgu"])), w["wd"])
        dist.all_reduce(d)
        if nw is None:
            return x + d, None
        return add_rmsnorm(x, nw, c.eps, r=d)

    def _capture(self, fn):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self.pool):
            fn()
        return g

    @torch.inference_mode()
    def build_decode(self, cfg, B, Lb):
        """Capture the graphs of one decode step for (cfg, B, Lb). Static buffers: tok [B], pos [B] in; nxt [B] out."""
        key = (cfg, B, Lb)
        if key in self.graphs:
            return self.graphs[key]
        H, L, eps = self.cfg.H, self.L, self.cfg.eps
        st = dict(tok=torch.zeros(B, dtype=torch.long, device=self.dev),
                  pos=torch.zeros(B, dtype=torch.long, device=self.dev),
                  xmid=torch.zeros(B, H, dtype=torch.bfloat16, device=self.dev),
                  hbuf=torch.zeros(B, H, dtype=torch.bfloat16, device=self.dev),
                  nxt=torch.zeros(B, dtype=torch.long, device=self.dev))

        def run(x, h, l0, l1, last_w):
            for i in range(l0, l1):
                nw = self.layers[i + 1]["ln1"] if i + 1 < l1 else last_w
                x, h = self._layer_dec(i, x, h, st["pos"], nw)
            return x, h

        def embed_norm():
            x = F.embedding(st["tok"], self.embed)
            return x, add_rmsnorm(x, self.layers[0]["ln1"], eps)[1]

        def head(xl):
            st["nxt"].copy_(F.linear(xl, self.lm_head).float().argmax(-1))

        if cfg == "A":
            def f1():
                _, h = run(*embed_norm(), 0, L, self.norm)
                head(h)
            gs = [self._capture(f1)]
        elif cfg in ("AsplitF", "Bfinal"):
            def f1():
                x, _ = run(*embed_norm(), 0, L, None)
                st["xmid"].copy_(x)

            if cfg == "AsplitF":
                def f2():
                    head(add_rmsnorm(st["xmid"], self.norm, eps)[1])
            else:
                def f2():
                    dist.broadcast(st["hbuf"], src=0)
                    head(st["hbuf"])
            gs = [self._capture(f1), self._capture(f2)]
        else:  # AsplitM / Bmid*
            def f1():
                x, _ = run(*embed_norm(), 0, MID, None)
                st["xmid"].copy_(x)

            if cfg == "AsplitM":
                def f2():
                    h = add_rmsnorm(st["xmid"], self.layers[MID]["ln1"], eps)[1]
                    head(run(st["xmid"], h, MID, L, self.norm)[1])
            else:
                def f2():
                    dist.broadcast(st["hbuf"], src=0)
                    head(run(st["xmid"], st["hbuf"], MID, L, self.norm)[1])
            gs = [self._capture(f1), self._capture(f2)]
        self.graphs[key] = (gs, st)
        return self.graphs[key]

    @torch.inference_mode()
    def decode_run(self, cfg, B, Lb, tok0, pos0, steps):
        """steps greedy decode steps from tokens tok0 [B] at positions pos0 (int). Returns tokens [B, steps]."""
        gs, st = self.build_decode(cfg, B, Lb)
        which = "final" if cfg == "Bfinal" else "mid"
        hop = cfg.startswith("B")
        if self.rank == 0 and hop:
            for _ in range(steps):
                self.hop.post_device(which, B, 1)
        st["tok"].copy_(tok0)
        out = []
        for s in range(steps):
            st["pos"].fill_(pos0 + s)
            gs[0].replay()
            if len(gs) > 1 and not hop:
                gs[1].replay()
            elif hop:
                if self.rank == 0:
                    ready = torch.cuda.Event()
                    ready.record()
                    ev = self.hop.post_gpu(st["xmid"], st["hbuf"], B, 1, ready)[0]
                    torch.cuda.current_stream().wait_event(ev)
                gs[1].replay()
            out.append(st["nxt"].clone())
            st["tok"].copy_(st["nxt"])
        return torch.stack(out, 1)


def pin(rank):
    os.sched_setaffinity(0, list(range(4 * rank, 4 * rank + 4)))
    torch.set_num_threads(4)


def mode_check(m, rank, args, cfgs):
    ref = json.load(open(args.reference))
    rows = []
    for item in ref["items"]:
        ids = torch.tensor([item["prompt_ids"]], device=m.dev)
        T, n_new = ids.shape[1], len(item["hf_tokens"])
        Lb = 128 * math.ceil((T + n_new) / 128)
        res = {}
        for cfg in cfgs:
            logits = m.prefill(ids, cfg)
            t0 = logits.argmax(-1)
            toks = [int(t0)] + m.decode_run(cfg, 1, Lb, t0, T, n_new - 1)[0].tolist()
            hf = item["hf_tokens"]
            res[cfg] = dict(identical_prefix=next((i for i, (a, b) in enumerate(zip(toks, hf)) if a != b), len(hf)),
                            tokens=toks)
        same = {c: res[c]["tokens"] == res[cfgs[0]]["tokens"] for c in cfgs}
        rows.append(dict(name=item["name"], n=n_new, results=res, identical_to_first_cfg=same))
        if rank == 0:
            print(f"[check] {item['name']}: " + "  ".join(f"{c} vs HF {res[c]['identical_prefix']}/{n_new}" for c in cfgs)
                  + f"  | all configs identical to {cfgs[0]}: {all(same.values())}", flush=True)
    if rank == 0:
        with open(os.path.join(args.out, "check_v4.json"), "w") as f:
            json.dump(rows, f, indent=1)


def mode_bench(m, rank, args, cfgs):
    g = torch.Generator().manual_seed(0)
    shapes = [tuple(int(v) for v in s.split("x")) for s in args.prefill.split(",") if s]
    dbs = [int(b) for b in args.decode.split(",") if b]
    Lb = 128 * math.ceil((args.decode_prompt + args.decode_steps) / 128)
    ids_p = {(b, t): torch.randint(0, m.cfg.V, (b, t), generator=g).to(m.dev) for b, t in shapes}
    ids_d = {b: torch.randint(0, m.cfg.V, (b, args.decode_prompt), generator=g).to(m.dev) for b in dbs}
    for b in dbs:
        for cfg in cfgs:
            m.build_decode(cfg, b, Lb)
    raw = {f"{cfg}|{k}": [] for cfg in cfgs for k in
           ([f"prefill_{b}x{t}" for b, t in shapes] if not cfg.startswith("Asplit") else []) + [f"decode_b{b}" for b in dbs]}
    t_start = time.time()
    rng = random.Random(1)

    def one_prefill(cfg, b, t):
        dist.barrier()
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        m.prefill(ids_p[(b, t)], cfg)
        e1.record()
        e1.synchronize()
        return e0.elapsed_time(e1)

    def one_decode(cfg, b):
        logits = m.prefill(ids_d[b], "A")  # identical KV cache for every configuration
        t0 = logits.argmax(-1)
        dist.barrier()
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        m.decode_run(cfg, b, Lb, t0, args.decode_prompt, args.decode_steps)
        e1.record()
        e1.synchronize()
        return e0.elapsed_time(e1) / args.decode_steps

    for _ in range(2):  # warm-up every path
        for cfg in cfgs:
            for b, t in shapes:
                if not cfg.startswith("Asplit"):
                    one_prefill(cfg, b, t)
            for b in dbs:
                one_decode(cfg, b)
    for r in range(args.rounds):
        jobs = [(cfg, "p", s) for cfg in cfgs if not cfg.startswith("Asplit") for s in shapes] + \
               [(cfg, "d", b) for cfg in cfgs for b in dbs]
        rng.shuffle(jobs)  # same order on every rank (same seed): collectives stay matched
        for cfg, kind, s in jobs:
            if kind == "p":
                raw[f"{cfg}|prefill_{s[0]}x{s[1]}"].append(one_prefill(cfg, *s))
            else:
                raw[f"{cfg}|decode_b{s}"].append(one_decode(cfg, s))
        if rank == 0:
            print(f"[progress] round {r + 1}/{args.rounds} done, {time.time() - t_start:.0f} s", flush=True)
    if rank == 0:
        summ = {}
        for k, v in raw.items():
            sv = sorted(v)
            summ[k] = dict(median_ms=statistics.median(v), p25=sv[len(sv) // 4], p75=sv[3 * len(sv) // 4], rounds=len(v))
        base = {k.split("|")[1]: s["median_ms"] for k, s in summ.items() if k.startswith("A|")}
        for k, s in summ.items():
            cfg, case = k.split("|")
            rel = s["median_ms"] / base[case] - 1 if cfg != "A" and case in base else 0.0
            print(f"[bench] {cfg:<7} {case:<16} median {s['median_ms']:9.3f} ms  (p25 {s['p25']:.3f}, p75 {s['p75']:.3f})"
                  + (f"  vs A {100 * rel:+6.2f}%" if cfg != "A" else ""), flush=True)
        with open(os.path.join(args.out, "bench_v4.json"), "w") as f:
            json.dump(dict(raw_ms=raw, summary=summ, args=vars(args)), f, indent=1)


def main():
    import faulthandler
    faulthandler.dump_traceback_later(int(os.environ.get("HXB_STACK_DUMP_S", "0")) or 10 ** 7, exit=False)
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["check", "bench"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--reference", default=None)
    ap.add_argument("--configs", default="A,AsplitF,Bfinal,AsplitM,Bmid8,Bmid1")
    ap.add_argument("--max_batch", type=int, default=64)
    ap.add_argument("--max_len", type=int, default=4352)
    ap.add_argument("--prefill", default="1x1024,8x1024,4x4096")
    ap.add_argument("--decode", default="1,8,32,64")
    ap.add_argument("--decode_prompt", type=int, default=512)
    ap.add_argument("--decode_steps", type=int, default=64)
    ap.add_argument("--rounds", type=int, default=20)
    args = ap.parse_args()
    rank, world, dev = init()
    pin(rank)
    os.makedirs(args.out, exist_ok=True)
    cfgs = args.configs.split(",")
    shapes = [tuple(int(v) for v in s.split("x")) for s in args.prefill.split(",") if s]
    hop_rows = max([b * t for b, t in shapes] + [args.max_batch, 1024]) if any(c.startswith("B") for c in cfgs) else 0
    chunks = sorted({int(c[4:]) for c in cfgs if c.startswith("Bmid")} | {1})
    t0 = time.time()
    m = Engine(rank, world, dev, args.max_batch, args.max_len, hop_rows=hop_rows, hop_chunks=chunks)
    dist.barrier()
    if rank == 0:
        print(f"[load] {time.time() - t0:.1f} s, world {world}, configs {cfgs}, hop rows {hop_rows}, "
              f"mem {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)
    ok = False
    try:
        if args.mode == "check":
            mode_check(m, rank, args, cfgs)
        else:
            mode_bench(m, rank, args, cfgs)
        ok = True
    except Exception:
        import traceback
        traceback.print_exc()  # before any collective clean-up: a failed rank would leave the others waiting
        sys.stderr.flush()
        raise
    finally:
        if m.hop is not None:
            m.hop.close()
        if ok:
            dist.barrier()
            # destroy_process_group() hangs after NCCL collectives were captured in CUDA graphs (seen 2026-10-14);
            # every result is written by now, so leave without it
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)


if __name__ == "__main__":
    main()
