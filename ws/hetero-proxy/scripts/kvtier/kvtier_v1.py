# WITHDRAWN 2026-10-16: decode-graph capture warm-up overwrote KV position 0 in the check (results/kvtier/check_v1). Replaced by
# ws/hetero-proxy/scripts/kvtier/kvtier_v2.py.
################################################################################
# hetero-proxy KV tier v1 (2026-10-16): the CPU as a KV-cache tier for multi-turn conversations, on the real
# Qwen2.5-32B-Instruct TP8 engine (e2e/qwen_tp_v5.py). Question: when a conversation's KV no longer fits in GPU
# memory, how much faster is "bring it back over PCIe" than "recompute it" (what vLLM / SGLang do after eviction)?
# Design (STATUS section 0, KV tier plan; LMCache / SGLang HiCache style, pull mode):
#   * every rank keeps its own KV shard (1 KV head at TP8, 32 KiB per token per GPU) in host memory, pinned and
#     NUMA-local to its GPU; host slot layout [layer, K|V, token, 128] so one layer's K (or V) for tokens [a, e) is
#     one contiguous run on both sides
#   * save: D2H of a token range, every layer, on the tier's copy stream
#   * restore: H2D layer by layer on the copy stream, one CUDA event per layer; the new turn's prefill waits for
#     layer i's event right before layer i's attention, so the copies of later layers overlap the compute of earlier
#     ones (Flux-style "start when the data is there", at layer granularity). Copies are issued LOOK layers ahead of
#     the compute launches (an event must be recorded before anything waits on it)
#   * split (Cake, ICML 2025): the GPU recomputes tokens [0, X) while PCIe brings [X, P), then the new turn
#   * the new turn's prefill ("extend"): T tokens at positions P.. attending to the cached [0, P) + causally to each
#     other (SDPA with a lower-right causal mask); at P = 0 it is exactly the v5 prefill (is_causal)
# Modes:
#   check  real multi-turn ShareGPT prompts (kvtier/hf_reference_kv_v1.py): resident / restore-serial /
#          restore-layer / split / recompute paths, every one after overwriting the GPU KV with noise; restore paths
#          must be bit-identical to resident (logits and tokens); all paths vs HF tokens
#   bench  TTFT of one request, history P tokens + new turn T tokens, all 8 ranks; cases interleaved and shuffled per
#          round, CUDA events on every rank, rank-max per case and round, medians; host slots rotate (cold DRAM)
#          ideal      the history is still on the GPU: extend(T) only                     (lower bound)
#          recompute  prefill of P + T from scratch                                       (baseline)
#          serial     restore all layers, then extend(T)
#          layer      layer-by-layer restore overlapped with extend(T)                    (this design)
#          split<x>   recompute [0, X = x P) while restoring [X, P), then extend(T)       (Cake)
#          load       the restore alone (per-GPU H2D GB/s; anchor 10-12 GB/s with 8 GPUs, results/e2e/c_diag)
#          save       D2H of the history alone (anchor ~10 GB/s with 8 GPUs, results/e2e/c_diag)
#          Job wall windows are written out for the SM-clock log (nvidia-smi -lms, run_bench_v1.sh).
# PRE-REGISTERED for bench (written before the first run) [推論]:
#   K1 load: 9-12.5 GB/s per GPU with all 8 GPUs restoring (anchor b_h2d_all8 10.1-12.2 GB/s)
#   K2 recompute TTFT: P=8192 450-650 ms; P=32000 1.8-3.0 s (8x1024 prefill is 398 ms in bench_v5)
#   K3 ideal TTFT (T=128): 10-30 ms at every P, P=32000 at most 2x P=1024
#   K4 layer TTFT = max(load, ideal) + at most 3 ms: P=8192 27-36 ms, P=32000 100-130 ms
#   K5 layer is >= 10x faster than recompute for P >= 4096 and >= 3x at P = 1024
#   K6 serial - layer = the part of extend hidden under the copies: 5-25 ms
#   K7 split: the best x is < 10% faster than layer, and may be slower (PCIe traffic slows the NCCL all-reduce of
#      the recompute, results/e2e/c_diag f: 0.40 -> 4.5 ms)
#   K8 check: restore-serial / restore-layer bit-identical to resident on every item
# Usage (8 ranks):
#   env -u CUDA_DEVICE_MAX_CONNECTIONS CUDA_MODULE_LOADING=EAGER pixi run --manifest-path pixi.toml \
#       ~/.venvs/hetero-proxy-hf/bin/python -m torch.distributed.run --nproc_per_node 8 \
#       ws/hetero-proxy/scripts/kvtier/kvtier_v1.py --mode check|bench --out <dir> [--reference <hf_reference_kv.json>]
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
from torch.nn.attention.bias import causal_lower_right

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "e2e"))
sys.path.insert(0, os.path.join(HERE, ".."))
from qwen_tp_v1 import init, rotate_half  # noqa: E402
from qwen_tp_v5 import Engine, pin  # noqa: E402
from fused_kernels import add_rmsnorm, silu_mul  # noqa: E402
from hxb import cudamem as cm  # noqa: E402

LOOK = 8  # restore copies are issued this many layers ahead of the compute launches


def node_cpus(node):
    out = []
    for part in open(f"/sys/devices/system/node/node{node}/cpulist").read().strip().split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def numa_pages(addr):
    """Pages per NUMA node of the mapping that contains addr (/proc/self/numa_maps)."""
    best = None
    for line in open("/proc/self/numa_maps"):
        a = int(line.split()[0], 16)
        if a <= addr and (best is None or a > best[0]):
            best = (a, line)
    return {kv.split("=")[0]: int(kv.split("=")[1]) for kv in best[1].split()
            if kv[0] == "N" and "=" in kv and kv[1:].split("=")[0].isdigit()}


class KVTier:
    """This rank's host-memory tier for its KV shard. Host slot layout [L, 2 (K | V), cap, HD] bf16."""

    def __init__(self, m, n_slots, cap):
        assert m.nkv == 1
        self.m, self.L, self.HD, self.cap = m, m.L, m.cfg.HD, cap
        self.node = 0 if m.dev.index < 4 else 1
        old, nt = os.sched_getaffinity(0), torch.get_num_threads()
        os.sched_setaffinity(0, node_cpus(self.node))
        torch.set_num_threads(1)  # first touch by this thread, on the GPU's NUMA node
        self.buf = torch.zeros((n_slots, self.L, 2, cap, self.HD), dtype=torch.bfloat16)
        os.sched_setaffinity(0, old)
        torch.set_num_threads(nt)
        self.nbytes = self.buf.numel() * 2
        cm.register(self.buf.data_ptr(), self.nbytes)
        self.numa = numa_pages(self.buf.data_ptr())
        self.stream = torch.cuda.Stream(m.dev)
        self.evs = [torch.cuda.Event() for _ in range(self.L)]
        self.done = torch.cuda.Event()

    def _h(self, h, i, kv, a):
        s = self.buf.stride()
        return self.buf.data_ptr() + 2 * (h * s[0] + i * s[1] + kv * s[2] + a * s[3])

    def _g(self, i, kv, b, a):
        t = self.m.kc if kv == 0 else self.m.vc
        return t.data_ptr() + 2 * (i * t.stride(0) + b * t.stride(1) + a * t.stride(3))

    def save(self, h, b, a, e):
        """D2H tokens [a, e) of GPU cache slot b, every layer, into host slot h. Returns the completion event."""
        s = self.stream
        s.wait_stream(torch.cuda.current_stream())  # the KV was written by the compute stream
        n = (e - a) * self.HD * 2
        for i in range(self.L):
            for kv in (0, 1):
                cm.memcpy(s, self._h(h, i, kv, a), self._g(i, kv, b, a), n)
        self.done.record(s)
        return self.done

    def restore(self, h, b, a, e):
        return RestoreJob(self, h, b, a, e)


class RestoreJob:
    """H2D of tokens [a, e) of host slot h into GPU slot b, layer by layer; issue_upto(n) posts layers < n."""

    def __init__(self, tier, h, b, a, e):
        self.t, self.h, self.b, self.a, self.e = tier, h, b, a, e
        self.n = (e - a) * tier.HD * 2
        self.issued = 0
        tier.stream.wait_stream(torch.cuda.current_stream())  # the request starts now

    def issue_upto(self, n):
        t, s = self.t, self.t.stream
        while self.issued < min(n, t.L):
            i = self.issued
            if self.n:
                for kv in (0, 1):
                    cm.memcpy(s, t._g(i, kv, self.b, self.a), t._h(self.h, i, kv, self.a), self.n)
            t.evs[i].record(s)
            self.issued += 1

    def wait(self, i):
        assert self.issued > i
        torch.cuda.current_stream().wait_event(self.t.evs[i])

    def wait_all(self):
        self.issue_upto(self.t.L)
        self.wait(self.t.L - 1)


class KVEngine(Engine):
    def _layer_ext(self, i, x, h, b, pos0, T, job):
        c, w = self.cfg, self.layers[i]
        hd, nq, nkv = c.HD, self.nq, self.nkv
        qkv = F.linear(h, w["wqkv"], w["bqkv"])
        q, k, v = qkv.split([nq * hd, nkv * hd, nkv * hd], dim=-1)
        q = q.view(1, T, nq, hd).transpose(1, 2)
        k = k.view(1, T, nkv, hd).transpose(1, 2)
        v = v.view(1, T, nkv, hd).transpose(1, 2)
        cos, sin = self.cos[pos0:pos0 + T], self.sin[pos0:pos0 + T]
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        S = pos0 + T
        self.kc[i, b:b + 1, :, pos0:S] = k
        self.vc[i, b:b + 1, :, pos0:S] = v
        if job is not None:
            job.wait(i)
        kk, vv = self.kc[i, b:b + 1, :, :S], self.vc[i, b:b + 1, :, :S]
        if pos0 == 0:
            a = F.scaled_dot_product_attention(q, kk, vv, is_causal=T > 1, enable_gqa=True, scale=1 / math.sqrt(hd))
        else:
            a = F.scaled_dot_product_attention(q, kk, vv, attn_mask=causal_lower_right(T, S), enable_gqa=True,
                                               scale=1 / math.sqrt(hd))
        o = F.linear(a.transpose(1, 2).reshape(T, nq * hd), w["wo"])
        dist.all_reduce(o)
        x, h2 = add_rmsnorm(x, w["ln2"], c.eps, r=o)
        d = F.linear(silu_mul(F.linear(h2, w["wgu"])), w["wd"])
        dist.all_reduce(d)
        nw = self.layers[i + 1]["ln1"] if i + 1 < self.L else self.norm
        return add_rmsnorm(x, nw, c.eps, r=d)

    @torch.inference_mode()
    def extend(self, ids, pos0, b=0, job=None, issue=None, head=True):
        """ids [T] of one sequence at positions pos0.. of cache slot b. job: a RestoreJob whose layer i gates layer i's
        attention; issue: a RestoreJob whose copies are posted LOOK layers ahead of this loop (may be job). Returns the
        last position's logits [V] (fp32), or None without head."""
        T = ids.numel()
        x = F.embedding(ids, self.embed)
        _, h = add_rmsnorm(x, self.layers[0]["ln1"], self.cfg.eps)
        for i in range(self.L):
            if issue is not None:
                issue.issue_upto(i + LOOK)
            x, h = self._layer_ext(i, x, h, b, pos0, T, job)
        if issue is not None:
            issue.issue_upto(self.L)
        if not head:
            return None
        return F.linear(h[-1:], self.lm_head).float()[0]


# ------------------------------------------------------------------ the request paths (one sequence, slot 0)
def run_path(m, tier, kind, hist, new, h, x=None):
    """hist: ids [P] of the history, new: ids [T]. Returns last-position logits. GPU slot 0."""
    P = hist.numel()
    if kind == "ideal":
        return m.extend(new, P)
    if kind == "recompute":
        return m.extend(torch.cat([hist, new]), 0)
    if kind == "serial":
        job = tier.restore(h, 0, 0, P)
        job.wait_all()
        return m.extend(new, P)
    if kind == "layer":
        job = tier.restore(h, 0, 0, P)
        return m.extend(new, P, job=job, issue=job)
    if kind == "split":
        X = split_point(P, x)
        job = tier.restore(h, 0, X, P)
        m.extend(hist[:X], 0, issue=job, head=False)
        return m.extend(new, P, job=job, issue=job)
    raise ValueError(kind)


def split_point(P, x):
    return max(128, int(P * x) // 128 * 128)


def mode_check(m, tier, rank, args):
    ref = json.load(open(args.reference))
    rows = []
    for it in ref["items"]:
        ids = torch.tensor(it["prompt_ids"], device=m.dev)
        P = it["prefix_len"]
        hist, new = ids[:P], ids[P:]
        n_new, hf = len(it["hf_tokens"]), it["hf_tokens"]
        Lb = 128 * math.ceil((ids.numel() + n_new) / 128)
        res, logit0 = {}, {}

        def noise():
            m.kc[:, 0].normal_()
            m.vc[:, 0].normal_()

        def finish(name, logits):
            t0 = logits.argmax(-1)
            toks = [int(t0)] + m.decode_run("A", 1, Lb, t0.view(1), ids.numel(), n_new - 1)[0].tolist()
            logit0[name] = logits.clone()
            res[name] = dict(tokens=toks, vs_hf=next((i for i, (a, b) in enumerate(zip(toks, hf)) if a != b), n_new))

        noise()
        m.extend(hist, 0, head=False)  # the previous turn left the history in the cache
        tier.save(0, 0, 0, P)
        torch.cuda.current_stream().wait_event(tier.done)
        finish("resident", m.extend(new, P))
        for kind, x in (("serial", None), ("layer", None), ("split", 0.25), ("recompute", None)):
            noise()
            dist.barrier()
            finish(kind if x is None else f"split{x}", run_path(m, tier, kind, hist, new, 0, x))
        base = logit0["resident"]
        for k in res:
            res[k]["logits_bit_identical_to_resident"] = bool(torch.equal(logit0[k], base))
            res[k]["max_abs_logit_diff_vs_resident"] = float((logit0[k] - base).abs().max())
            res[k]["tokens_identical_to_resident"] = res[k]["tokens"] == res["resident"]["tokens"]
        hl = torch.tensor(it["hf_first_logits_top"], device=m.dev)
        for k in res:
            res[k]["hf_top20_max_abs_logit_diff"] = float((logit0[k][hl[:, 0].long()] - hl[:, 1]).abs().max())
        rows.append(dict(name=it["name"], prefix_len=P, new_len=new.numel(), n=n_new, results=res))
        if rank == 0:
            print(f"[check] {it['name']} (P {P}, T {new.numel()}): " + "  ".join(
                f"{k}: HF {r['vs_hf']}/{n_new} bit={int(r['logits_bit_identical_to_resident'])} "
                f"d={r['max_abs_logit_diff_vs_resident']:.3f}" for k, r in res.items()), flush=True)
    if rank == 0:
        with open(os.path.join(args.out, "check_kvtier_v1.json"), "w") as f:
            json.dump(dict(numa_pages_rank0=tier.numa, rows=rows), f, indent=1)


def mode_bench(m, tier, rank, args):
    Ps = [int(p) for p in args.P.split(",")]
    T, Pmax = args.T, max(Ps)
    xs = [float(x) for x in args.split.split(",") if x]
    g = torch.Generator().manual_seed(0)
    seq = torch.randint(0, m.cfg.V, (Pmax + T,), generator=g).to(m.dev)  # same on every rank
    hist_all, new = seq[:Pmax], seq[Pmax:]
    m.extend(hist_all, 0, head=False)
    for h in range(args.slots):
        tier.save(h, 0, 0, Pmax)
    torch.cuda.synchronize()
    kinds = [("ideal", None), ("recompute", None), ("serial", None), ("layer", None)] + [("split", x) for x in xs]
    cases = [(k, x, P) for P in Ps for k, x in kinds] + [(k, None, P) for P in Ps for k in ("load", "save")]
    name = lambda k, x, P: f"{k if x is None else f'split{x:g}'}|P{P}"  # noqa: E731
    raw = {name(*c): [] for c in cases}
    windows = {name(*c): [] for c in cases}
    rot = [0]

    def one(kind, x, P):
        h = rot[0] % args.slots
        rot[0] += 1
        hist = hist_all[:P]
        dist.barrier()
        torch.cuda.synchronize()
        w0 = time.time()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        if kind == "load":
            tier.restore(h, 0, 0, P).wait_all()
        elif kind == "save":
            torch.cuda.current_stream().wait_event(tier.save(h, 0, 0, P))
        else:
            run_path(m, tier, kind, hist, new, h, x)
        e1.record()
        e1.synchronize()
        return e0.elapsed_time(e1), (w0, time.time())

    for _ in range(2):  # warm-up every case (allocator, cuBLAS / SDPA plans, NCCL)
        for c in cases:
            one(*c)
    rng = random.Random(1)
    t_start = time.time()
    for r in range(args.rounds):
        order = cases[:]
        rng.shuffle(order)  # same order on every rank
        for c in order:
            ms, w = one(*c)
            raw[name(*c)].append(ms)
            windows[name(*c)].append(w)
        if rank == 0 and (r + 1) % 10 == 0:
            print(f"[progress] round {r + 1}/{args.rounds}, {time.time() - t_start:.0f} s", flush=True)
    allr = [None] * m.world
    dist.all_gather_object(allr, raw)
    if rank == 0:
        rmax = {k: [max(allr[q][k][j] for q in range(m.world)) for j in range(len(raw[k]))] for k in raw}
        summ = {}
        for k, v in rmax.items():
            sv = sorted(v)
            summ[k] = dict(median_ms=statistics.median(v), p25=sv[len(sv) // 4], p75=sv[3 * len(sv) // 4],
                           rounds=len(v))
        nb = lambda P: P * 32 * 1024  # noqa: E731  KV bytes per GPU (64 layers x 1 head x 128 x K,V x bf16)
        for P in Ps:
            rec = summ[f"recompute|P{P}"]["median_ms"]
            line = [f"P={P:<6}"]
            for k, x in kinds:
                s = summ[name(k, x, P)]["median_ms"]
                line.append(f"{name(k, x, P).split('|')[0]} {s:8.2f}" + (f" (x{rec / s:5.1f})" if k != "recompute" else ""))
            ld, sv = summ[f"load|P{P}"]["median_ms"], summ[f"save|P{P}"]["median_ms"]
            line.append(f"load {ld:7.2f} ms = {nb(P) / ld / 1e6:5.1f} GB/s/GPU  save {sv:7.2f} ms = "
                        f"{nb(P) / sv / 1e6:5.1f} GB/s/GPU")
            print("[bench] " + "  ".join(line), flush=True)
        with open(os.path.join(args.out, "bench_kvtier_v1.json"), "w") as f:
            json.dump(dict(rank_max_ms=rmax, per_rank_ms=allr, windows_rank0=windows, summary=summ, args=vars(args),
                           numa_pages_rank0=tier.numa), f, indent=1)


def main():
    import faulthandler
    faulthandler.dump_traceback_later(int(os.environ.get("HXB_STACK_DUMP_S", "0")) or 10 ** 7, exit=False)
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["check", "bench"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--reference", default=None)
    ap.add_argument("--P", default="1024,2048,4096,8192,16384,32000")
    ap.add_argument("--T", type=int, default=128)
    ap.add_argument("--split", default="0.0625,0.125,0.25")
    ap.add_argument("--slots", type=int, default=4, help="host slots rotated in bench (cold DRAM)")
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--max_len", type=int, default=32768)
    args = ap.parse_args()
    rank, world, dev = init()
    pin(rank)
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    m = KVEngine(rank, world, dev, 1, args.max_len)
    if args.mode == "check":
        cap = max(len(it["prompt_ids"]) for it in json.load(open(args.reference))["items"])
        tier = KVTier(m, 1, cap)
    else:
        tier = KVTier(m, args.slots, max(int(p) for p in args.P.split(",")))
    nodes = [None] * world
    dist.all_gather_object(nodes, (tier.node, tier.numa))
    dist.barrier()
    if rank == 0:
        print(f"[load] {time.time() - t0:.1f} s, world {world}, GPU mem {torch.cuda.memory_allocated() / 2**30:.1f} GiB,"
              f" host tier {tier.nbytes / 2**30:.2f} GiB per rank, NUMA pages per rank {nodes}", flush=True)
    ok = False
    try:
        if args.mode == "check":
            mode_check(m, tier, rank, args)
        else:
            mode_bench(m, tier, rank, args)
        ok = True
    except Exception:
        import traceback
        traceback.print_exc()
        sys.stderr.flush()
        raise
    finally:
        if ok:
            dist.barrier()
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)  # destroy_process_group() hangs after graph-captured NCCL (qwen_tp_v5.py)


if __name__ == "__main__":
    main()
