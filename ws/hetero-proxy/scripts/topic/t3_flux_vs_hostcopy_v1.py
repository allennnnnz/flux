"""T3 (2026-10-30, topic search): are Flux's fused AG+GEMM / GEMM+RS kernels slowed by background GPU->host copies
the way NCCL is (T1)?  Flux synchronizes every tile with system-scope fences / acquire loads
(include/flux/cuda/system_barrier.hpp:41-52), the same fence kind X1c found slowed by host copies.

Shapes: Qwen2.5-32B layers, TP8 + sequence parallel, 4096 tokens per step (M = 4096):
  AG  gate_up  n = 2*27648/8 = 6912, K = 5120   (AG of [4096, 5120] bf16 = 40 MiB)
  AG  qkv      n = (5120+2*1024)/8 = 896, K = 5120
  RS  down     N = 5120, k = 27648/8 = 3456      (RS of [4096, 5120] bf16 = 40 MiB)
  RS  o_proj   N = 5120, k = 5120/8 = 640
Items per layer: fused (Flux AGKernel.forward / GemmRS.forward), nccl (NCCL all_gather + mm / mm + reduce_scatter),
gemm (mm only, same per-rank GEMM).  Background modes: none | d2h8 (all 8 GPUs copy to host) | d2h2 (GPU0+GPU1).
Same window / pairing as T1: shuffled per round, >= 30 rounds, 2 warm-up + 10 timed calls, rank-max per-call time,
ratio to the same round's 'none'.  Launch with ./launch.sh (Flux needs its NVSHMEM env), CUDA_DEVICE_MAX_CONNECTIONS=8.
"""
import argparse
import json
import os
import random
import statistics
import sys

import torch
import torch.distributed as dist

import flux
from flux.testing import initialize_distributed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../common/measure"))
from clock_logger import ClockLogger  # noqa: E402

MiB = 1 << 20
CHUNK = 16 * MiB
DT = torch.bfloat16
M = 4096
AG_LAYERS = {"gate_up": (55296, 5120), "qkv": (7168, 5120)}   # (N total, K)
RS_LAYERS = {"down": (5120, 27648), "o_proj": (5120, 5120)}  # (N, K total)
NUMA_CPUS = {0: list(range(0, 32)) + list(range(64, 96)), 1: list(range(32, 64)) + list(range(96, 128))}


def ag_option():
    opt = flux.AllGatherOption()
    opt.mode = flux.AGRingMode.All2All
    opt.use_read = True
    opt.use_cuda_core_local = False
    opt.use_cuda_core_ag = False
    opt.fuse_sync = False
    opt.input_buffer_copied = False
    return opt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--iters", type=int, default=10)
    args = ap.parse_args()

    tp = initialize_distributed()
    # undo flux init_seed(): production cuBLAS (CLAUDE.md trap table)
    os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    torch.use_deterministic_algorithms(False)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    rank, W = tp.rank(), tp.size()
    lr = int(os.environ.get("LOCAL_RANK", 0))
    os.sched_setaffinity(0, NUMA_CPUS[0 if lr < 4 else 1])
    nnodes = flux.testing.NNODES()

    host = torch.empty(256 * MiB, dtype=torch.uint8, pin_memory=True)
    host.fill_(1)
    gbuf = torch.empty(256 * MiB, dtype=torch.uint8, device="cuda")
    side = torch.cuda.Stream()
    opt = ag_option()

    fns = {}
    checks = {}
    for name, (N, K) in AG_LAYERS.items():
        n = N // W
        op = flux.AGKernel(tp, nnodes, M, n, K, DT, output_dtype=DT)  # built once, never destroyed (trap table)
        w = (torch.randn((n, K), device="cuda") * 0.01).to(DT)
        x = torch.randn((M // W, K), device="cuda").to(DT)
        full = torch.empty((M, K), dtype=DT, device="cuda")
        oA = torch.empty((M, n), dtype=DT, device="cuda")
        oB = torch.empty((M, n), dtype=DT, device="cuda")
        dist.all_gather_into_tensor(full, x)
        fns[f"AG_{name}|fused"] = lambda op=op, x=x, w=w, oA=oA: op.forward(x, w, output=oA, transpose_weight=False,
                                                                               all_gather_option=opt)
        fns[f"AG_{name}|nccl"] = lambda x=x, w=w, full=full, oB=oB: (dist.all_gather_into_tensor(full, x),
                                                                       torch.mm(full, w.t(), out=oB))
        fns[f"AG_{name}|gemm"] = lambda w=w, full=full, oB=oB: torch.mm(full, w.t(), out=oB)
        checks[f"AG_{name}"] = (oA, oB, fns[f"AG_{name}|fused"], fns[f"AG_{name}|nccl"])
    for name, (N, K) in RS_LAYERS.items():
        k = K // W
        op = flux.GemmRS(tp, 1, M, N, DT, DT, transpose_weight=False)
        w = (torch.randn((N, k), device="cuda") * 0.01).to(DT)
        x = torch.randn((M, k), device="cuda").to(DT)
        part = torch.empty((M, N), dtype=DT, device="cuda")
        orsB = torch.empty((M // W, N), dtype=DT, device="cuda")
        res = {}

        def fused(op=op, x=x, w=w, res=res):
            res["A"] = op.forward(x, w)
        fns[f"RS_{name}|fused"] = fused
        fns[f"RS_{name}|nccl"] = lambda x=x, w=w, part=part, orsB=orsB: (torch.mm(x, w.t(), out=part),
                                                                          dist.reduce_scatter_tensor(orsB, part))
        fns[f"RS_{name}|gemm"] = lambda x=x, w=w, part=part: torch.mm(x, w.t(), out=part)
        checks[f"RS_{name}"] = (res, orsB, fused, fns[f"RS_{name}|nccl"])

    # correctness once (fused vs nccl path), no background
    ok = {}
    for key, (a, b, fa, fb) in checks.items():
        fa(); fb(); torch.cuda.synchronize()
        A = a["A"] if isinstance(a, dict) else a
        ok[key] = bool(torch.allclose(A.float(), b.float(), atol=0.05, rtol=0.05))

    def barrier():
        torch.cuda.synchronize()
        dist.barrier()
        torch.cuda.synchronize()

    def window(fn, mode, nchunks):
        barrier()
        t0 = torch.cuda.Event(enable_timing=True)
        t0.record()
        on = mode == "d2h8" or (mode == "d2h2" and rank in (0, 1))
        last = None
        if on:
            side.wait_event(t0)
            with torch.cuda.stream(side):
                nh = host.numel() // CHUNK
                for i in range(nchunks):
                    kk = i % nh
                    host[kk * CHUNK:(kk + 1) * CHUNK].copy_(gbuf[kk * CHUNK:(kk + 1) * CHUNK], non_blocking=True)
                last = torch.cuda.Event(enable_timing=True)
                last.record()
        for _ in range(2):
            fn()
        cs, ce = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        cs.record()
        for _ in range(args.iters):
            fn()
        ce.record()
        torch.cuda.synchronize()
        cov = True if last is None else t0.elapsed_time(last) >= t0.elapsed_time(ce)
        return cs.elapsed_time(ce) / args.iters, cov

    nchunks = {}
    for key, fn in fns.items():
        t = max(window(fn, "none", 0)[0] for _ in range(3))
        loc = torch.tensor([t * (args.iters + 2) / 1e3], device="cuda")
        dist.all_reduce(loc, op=dist.ReduceOp.MAX)
        nchunks[key] = int(3.0 * loc.item() * 14e9 / CHUNK) + 4

    combos = [(k, m) for k in fns for m in ("none", "d2h8", "d2h2")]
    rec = []
    with ClockLogger(lr, period_s=0.25) as clk:
        import time
        for r in range(args.rounds):
            order = combos[:]
            random.Random(2000 + r).shuffle(order)
            for key, m in order:
                w0 = time.perf_counter()
                t, cov = window(fns[key], m, nchunks[key])
                rec.append({"round": r, "key": key, "mode": m, "ms": t, "covered": cov, "w": (w0, time.perf_counter())})
        for x in rec:
            x["clock"] = clk.min_clock_between(*x["w"])
            del x["w"]

    allrec = [None] * W
    dist.all_gather_object(allrec, rec)
    allok = [None] * W
    dist.all_gather_object(allok, ok)
    if rank == 0:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "t3_raw.json"), "w") as f:
            json.dump({"meta": {"M": M, "iters": args.iters, "rounds": args.rounds, "nchunks": nchunks, "check_allclose": allok,
                                "env": {k: v for k, v in os.environ.items() if k.startswith(("NCCL", "CUDA", "NVSHMEM"))}},
                       "ranks": allrec}, f)
        summarize(allrec, list(fns), args.out, allok)
    dist.barrier()
    dist.destroy_process_group()


def summarize(allrec, keys, out, allok):
    keyed = {}
    for recs in allrec:
        for x in recs:
            keyed.setdefault((x["round"], x["key"], x["mode"]), []).append(x)
    clocks = [x["clock"] for recs in allrec for x in recs if x["clock"]]
    modal = statistics.mode(clocks)
    lines = [f"T3 summary: rank-max per-call ms; ratio = paired same round; modal clock {modal} MHz (windows < 95% dropped); "
             f"fused==nccl allclose on all ranks: {all(all(o.values()) for o in allok)}"]
    rows = []
    for key in keys:
        base, rat = {}, {"d2h8": [], "d2h2": []}
        drop = unc = 0
        for (r, k, m), xs in keyed.items():
            if k != key:
                continue
            if any(x["clock"] and x["clock"] < 0.95 * modal for x in xs):
                drop += 1
                continue
            t = max(x["ms"] for x in xs)
            if m == "none":
                base[r] = t
            elif all(x["covered"] for x in xs):
                rat[m].append((r, t))
            else:
                unc += 1
        tb = statistics.median(base.values())
        line = f"{key:22s} none {tb:7.3f} ms"
        row = {"key": key, "none_ms": tb}
        for m in ("d2h8", "d2h2"):
            rs = sorted(t / base[r] for r, t in rat[m] if r in base)
            med = statistics.median(rs)
            line += f" | {m} x{med:.3f} [{rs[len(rs) // 10]:.3f}-{rs[(9 * len(rs)) // 10]:.3f}] -> {tb * med:7.3f} ms"
            row[m] = med
        line += f"  (n={len(base)}, dropped {drop}, uncovered {unc})"
        lines.append(line)
        rows.append(row)
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(out, "t3_summary.txt"), "w") as f:
        f.write(txt + "\n")
    with open(os.path.join(out, "t3_rows.json"), "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
