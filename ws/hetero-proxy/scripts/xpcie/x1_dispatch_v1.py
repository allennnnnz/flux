################################################################################
# hetero-proxy X1 (2026-10-28): why does background PCIe DMA slow tensor-parallel decode? (STATUS section 0, X series)
# K4.1 (results/kvtier/k4/interference_v2, place_v1): background PCIe copies slow a TP8 decode step +45-48% (H2D) /
# ~+145% (D2H); an eager nsys trace (diag_v1_nsys/gaps.txt) showed the gap between consecutive NCCL kernels growing
# 2.5 -> 23.9 us while the kernels themselves did not change (H2D). Hypothesis H: PCIe bulk DMA delays the START of
# each kernel (dispatch), so the damage grows with the number of kernels, not with what they do.
# Victims (each a CUDA graph, all 8 ranks; a 1-thread Triton kernel writes the GPU's %globaltimer into a buffer -
# "stamp" - so intervals are measured on the GPU itself, no profiler):
#   tiny   1024 stamp kernels back to back: the per-kernel interval is pure dispatch
#   gemm   128 x [stamp, GEMM 16x5120 @ 5120x6912 bf16] (a decode-sized, ~50 us memory-bound GEMM), no communication
#   ar     128 x [stamp, all_reduce(16x5120 bf16)] on a graph-only NCCL communicator (decode-sized all-reduces)
#   mix    64 x [stamp, GEMM, all_reduce, GEMM, all_reduce] (a decode step's pattern)
# Background (side stream, 4 MiB cuMemcpyAsync pieces, --bg_mb per copying GPU, enqueued right before the victim):
#   none | h2d_all | d2h_all (every GPU copies) | h2d_self (only GPU0 copies) | h2d_others (GPU1-7 copy, GPU0 not)
# Per case and round: rank-max CUDA-event time of the victim; GPU0's per-item interval = (last stamp - first stamp) /
# (items - 1). Cases shuffled per round (same order on every rank), --rounds rounds, medians.
# PRE-REGISTERED [推論]:
#   P1 tiny: per-kernel interval with h2d_all >= 5x the no-traffic interval; d2h_all >= h2d_all
#   P2 the time ADDED per item under h2d_all is about the same for tiny, gemm and ar (within 2x of each other), i.e. a
#      per-dispatch cost, so a ~50 us GEMM loses a small fraction and a ~1 us kernel a large one
#   P3 h2d_others adds < 20% of what h2d_self adds to GPU0's tiny interval (the effect is local to the copying GPU)
#   P4 mix under h2d_all slows by 30-60% (K4.1 decode step: +45-48%)
# Usage (repo root, exclusive guard, tmux):
#   env -u CUDA_DEVICE_MAX_CONNECTIONS CUDA_MODULE_LOADING=EAGER pixi run --manifest-path pixi.toml \
#     ~/.venvs/hetero-proxy-hf/bin/python -m torch.distributed.run --nproc_per_node 8 \
#     ws/hetero-proxy/scripts/xpcie/x1_dispatch_v1.py --out <dir> [--rounds 100]
################################################################################
import argparse
import json
import os
import random
import statistics
import sys
import time

import torch
import torch.distributed as dist
import triton
import triton.language as tl

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "kvtier"))
from hxb import cudamem as cm  # noqa: E402
from kvtier_v2 import node_cpus  # noqa: E402

MiB = 2 ** 20


@triton.jit
def _stamp(buf_ptr, idx):
    t = tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", "=l", [], dtype=tl.int64, is_pure=False, pack=1)
    tl.store(buf_ptr + idx, t)


def stamp(buf, i):
    _stamp[(1,)](buf, i)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=100)
    ap.add_argument("--bg_mb", type=int, default=512)
    args = ap.parse_args()
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    lr = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(lr)
    dev = torch.device(f"cuda:{lr}")
    os.sched_setaffinity(0, list(range(4 * rank, 4 * rank + 4)))
    torch.set_num_threads(4)
    g_graph = dist.new_group(list(range(world)))
    one = torch.zeros(1, device=dev)
    dist.all_reduce(one, group=g_graph)  # create the graph communicator before any capture
    torch.cuda.synchronize()

    # background buffers on this GPU's NUMA node
    nb, ch = args.bg_mb * MiB, 4 * MiB
    node = 0 if lr < 4 else 1
    old = os.sched_getaffinity(0)
    os.sched_setaffinity(0, node_cpus(node))
    torch.set_num_threads(1)
    host = torch.zeros(nb, dtype=torch.uint8)
    os.sched_setaffinity(0, old)
    torch.set_num_threads(4)
    cm.register(host.data_ptr(), nb)
    devbuf = torch.empty(nb, dtype=torch.uint8, device=dev)
    bgs = torch.cuda.Stream(dev)

    buf = torch.zeros(4096, dtype=torch.int64, device=dev)
    x = torch.randn(16, 5120, device=dev, dtype=torch.bfloat16)
    w = torch.randn(5120, 6912, device=dev, dtype=torch.bfloat16)
    y = torch.empty(16, 6912, device=dev, dtype=torch.bfloat16)
    small = torch.randn(16, 5120, device=dev, dtype=torch.bfloat16)
    group = [None]  # warm-ups all-reduce on the default communicator, captures on g_graph

    def ar(t):
        dist.all_reduce(t, group=group[0])

    def v_tiny():
        for i in range(1024):
            stamp(buf, i)

    def v_gemm():
        for i in range(128):
            stamp(buf, i)
            torch.mm(x, w, out=y)

    def v_ar():
        for i in range(128):
            stamp(buf, i)
            ar(small)

    def v_mix():
        for i in range(64):
            stamp(buf, i)
            torch.mm(x, w, out=y)
            ar(small)
            torch.mm(x, w, out=y)
            ar(small)

    victims = dict(tiny=(v_tiny, 1024), gemm=(v_gemm, 128), ar=(v_ar, 128), mix=(v_mix, 64))
    graphs = {}
    for name, (fn, _) in victims.items():
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        group[0] = None
        with torch.cuda.stream(s):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        group[0] = g_graph
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        torch.cuda.synchronize()
        graphs[name] = g

    def copying(mode):
        if mode == "none":
            return None
        if mode in ("h2d_all", "d2h_all"):
            return mode[:3]
        if mode == "h2d_self":
            return "h2d" if rank == 0 else None
        if mode == "h2d_others":
            return "h2d" if rank != 0 else None
        raise ValueError(mode)

    def one(victim, mode):
        dist.barrier()
        torch.cuda.synchronize()
        cs = torch.cuda.current_stream()
        b0, b1, e0, e1 = (torch.cuda.Event(enable_timing=True) for _ in range(4))
        d = copying(mode)
        if d:
            bgs.wait_stream(cs)
            b0.record(bgs)
            for off in range(0, nb, ch):
                if d == "h2d":
                    cm.memcpy(bgs, devbuf.data_ptr() + off, host.data_ptr() + off, ch)
                else:
                    cm.memcpy(bgs, host.data_ptr() + off, devbuf.data_ptr() + off, ch)
            b1.record(bgs)
        e0.record(cs)
        graphs[victim].replay()
        e1.record(cs)
        torch.cuda.synchronize()
        n = victims[victim][1]
        st = buf[:n].tolist()
        interval_ns = (st[n - 1] - st[0]) / (n - 1)
        outlast = (b0.elapsed_time(b1) >= b0.elapsed_time(e1)) if d else None
        return dict(ms=e0.elapsed_time(e1), interval_ns=interval_ns, outlast=outlast)

    modes = ["none", "h2d_all", "d2h_all", "h2d_self", "h2d_others"]
    cases = [(v, m) for v in victims for m in modes]
    for _ in range(3):
        for c in cases:
            one(*c)
    raw = {f"{v}|{m}": [] for v, m in cases}
    rng = random.Random(1)
    t0 = time.time()
    for r in range(args.rounds):
        order = cases[:]
        rng.shuffle(order)
        for c in order:
            raw[f"{c[0]}|{c[1]}"].append(one(*c))
        if rank == 0 and (r + 1) % 25 == 0:
            print(f"[progress] round {r + 1}/{args.rounds}, {time.time() - t0:.0f} s", flush=True)
    allr = [None] * world
    dist.all_gather_object(allr, raw)
    if rank == 0:
        summ = {}
        for k in raw:
            n = len(raw[k])
            ms = [max(allr[q][k][j]["ms"] for q in range(world)) for j in range(n)]
            iv = [allr[0][k][j]["interval_ns"] for j in range(n)]
            ol = [allr[0][k][j]["outlast"] for j in range(n)]
            summ[k] = dict(ms=statistics.median(ms), gpu0_interval_us=statistics.median(iv) / 1e3,
                           gpu0_bg_outlasted=sum(1 for o in ol if o) if ol[0] is not None else None, rounds=n)
        for v in victims:
            base = summ[f"{v}|none"]
            line = f"[x1] {v:<5} none {base['ms']:8.3f} ms, item {base['gpu0_interval_us']:7.2f} us"
            for m in modes[1:]:
                s = summ[f"{v}|{m}"]
                line += (f" | {m} {s['ms']:8.3f} ms ({100 * (s['ms'] / base['ms'] - 1):+6.1f}%), item "
                         f"{s['gpu0_interval_us']:7.2f} us (+{s['gpu0_interval_us'] - base['gpu0_interval_us']:6.2f})")
            print(line, flush=True)
        with open(os.path.join(args.out, "x1_dispatch_v1.json"), "w") as f:
            json.dump(dict(args=vars(args), summary=summ, raw=allr), f)
    dist.barrier()
    sys.stdout.flush()
    os._exit(0)  # destroy_process_group() hangs after graph-captured NCCL (qwen_tp_v5.py)


if __name__ == "__main__":
    main()
