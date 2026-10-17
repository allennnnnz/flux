################################################################################
# hetero-proxy K4.1 diagnostic (2026-10-17): which part of a decode step does background PCIe traffic hold back?
# k4_interference_v2 (results/kvtier/k4/interference_v2): with h2d background a decode step is +48%; with d2h
# background every op ENDS when the background ends (decode 38 ms vs 15.5; new2048 138 ms vs 111; the d2h volume
# takes 35 / 140 ms) - a dependency, not a slowdown. CUDA_DEVICE_MAX_CONNECTIONS=32 changes nothing.
# Workloads (eager, all 8 ranks, timed with CUDA events on the main stream), each with background none / h2d / d2h
# (--bg_mb per GPU in 4 MiB copies on a side stream, all 8 GPUs, enqueued right before the workload):
#   gemm     64 x [16x5120 @ 5120x6912]  (decode-sized GEMMs, no communication)
#   ar       128 x all_reduce(16 x 5120 bf16) on a NCCL group of its own (decode-sized all-reduces)
#   mix      64 x [GEMM, all_reduce, GEMM, all_reduce]  (a decode step's pattern)
#   ar_big   8 x all_reduce(20 MiB) (prefill-sized)
# Reported: workload ms (rank-max median), background ms, and whether the workload ended after the background did.
# Usage: torchrun 8 ranks (exclusive guard): k4_diag_v1.py --out <dir> [--rounds 30]
################################################################################
import argparse
import json
import os
import statistics
import sys

import torch
import torch.distributed as dist

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from hxb import cudamem as cm  # noqa: E402
from kvtier_v2 import node_cpus  # noqa: E402

MiB = 2 ** 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--bg_mb", type=int, default=384)
    args = ap.parse_args()
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    lr = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(lr)
    dev = torch.device(f"cuda:{lr}")
    os.sched_setaffinity(0, list(range(4 * rank, 4 * rank + 4)))
    g_ar = dist.new_group(list(range(world)))
    nb, ch = args.bg_mb * MiB, 4 * MiB
    node = 0 if lr < 4 else 1
    old = os.sched_getaffinity(0)
    os.sched_setaffinity(0, node_cpus(node))
    torch.set_num_threads(1)
    host = torch.zeros(nb, dtype=torch.uint8)
    os.sched_setaffinity(0, old)
    cm.register(host.data_ptr(), nb)
    devbuf = torch.empty(nb, dtype=torch.uint8, device=dev)
    bgs = torch.cuda.Stream(dev)
    x = torch.randn(16, 5120, device=dev, dtype=torch.bfloat16)
    w = torch.randn(5120, 6912, device=dev, dtype=torch.bfloat16)
    small = torch.randn(16, 5120, device=dev, dtype=torch.bfloat16)
    big = torch.randn(10 * MiB, device=dev, dtype=torch.bfloat16)

    def gemm():
        for _ in range(64):
            torch.mm(x, w)

    def ar():
        for _ in range(128):
            dist.all_reduce(small, group=g_ar)

    def mix():
        for _ in range(64):
            torch.mm(x, w)
            dist.all_reduce(small, group=g_ar)
            torch.mm(x, w)
            dist.all_reduce(small, group=g_ar)

    def ar_big():
        for _ in range(8):
            dist.all_reduce(big, group=g_ar)

    works = dict(gemm=gemm, ar=ar, mix=mix, ar_big=ar_big)

    def one(wk, b):
        dist.barrier()
        torch.cuda.synchronize()
        cs = torch.cuda.current_stream()
        b0, b1, e0, e1 = (torch.cuda.Event(enable_timing=True) for _ in range(4))
        if b != "none":
            bgs.wait_stream(cs)
            b0.record(bgs)
            for off in range(0, nb, ch):
                if b == "h2d":
                    cm.memcpy(bgs, devbuf.data_ptr() + off, host.data_ptr() + off, ch)
                else:
                    cm.memcpy(bgs, host.data_ptr() + off, devbuf.data_ptr() + off, ch)
            b1.record(bgs)
        e0.record(cs)
        works[wk]()
        e1.record(cs)
        torch.cuda.synchronize()
        if b == "none":
            return e0.elapsed_time(e1), None, None
        return e0.elapsed_time(e1), b0.elapsed_time(b1), b0.elapsed_time(e1) >= b0.elapsed_time(b1) - 0.05

    cases = [(wk, b) for wk in works for b in ("none", "h2d", "d2h")]
    for _ in range(3):
        for c in cases:
            one(*c)
    res = {f"{wk}|{b}": [] for wk, b in cases}
    for _ in range(args.rounds):
        for c in cases:
            res[f"{c[0]}|{c[1]}"].append(one(*c))
    allr = [None] * world
    dist.all_gather_object(allr, res)
    if rank == 0:
        out = {}
        for k in res:
            n = len(res[k])
            ms = [max(allr[q][k][j][0] for q in range(world)) for j in range(n)]
            line = f"[diag] {k:<12} {statistics.median(ms):9.3f} ms"
            o = dict(ms=statistics.median(ms))
            if res[k][0][1] is not None:
                bg = statistics.median([allr[q][k][j][1] for q in range(world) for j in range(n)])
                ends = sum(1 for q in range(world) for j in range(n) if allr[q][k][j][2])
                line += f"   background {bg:7.2f} ms, workload ended with/after it in {ends}/{world * n} (rank, round)"
                o.update(bg_ms=bg, ended_after_bg=ends, samples=world * n)
            out[k] = o
            print(line, flush=True)
        with open(os.path.join(args.out, "k4_diag_v1.json"), "w") as f:
            json.dump(dict(args=vars(args), summary=out, raw=allr), f)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
