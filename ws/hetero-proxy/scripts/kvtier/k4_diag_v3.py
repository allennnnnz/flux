# WITHDRAWN 2026-10-17: eager decode-sized NCCL loops are launch-bound and order-sensitive (no-copy baseline 11.4 / 12.9 /
# 17.0 ms across runs; v3's tick-only case 24% faster than no ticks). Replaced by ws/hetero-proxy/scripts/kvtier/k4_place_v1.py (graphs, shuffled).
################################################################################
# hetero-proxy K4.1 diagnostic v3 (2026-10-17): v2 plus the controls it lacked - results/kvtier/k4/diag_v2 showed
# gemm-released copies cutting the d2h slowdown from +116% (free) to +7% but leaving h2d at +21% (= free).
#   ticks     the tick writes of the gated modes WITHOUT copies (their own cost)
#   free512   continuous copies in 512 KiB pieces (does the copy size alone matter?)
#   gemm128 / gemm512 / gemm1024   gemm-released chunks of 128 KiB / 512 KiB / 1 MiB
#   ar512     ar-released 512 KiB chunks (control)
# (v2 header) can Flux-style placement of the PCIe copies avoid the slowdown?
# k4_diag_v1 (results/kvtier/k4/diag_v1): background PCIe copies slow decode-sized NCCL all-reduces 1.25x (h2d) /
# 3.7x (d2h) and decode-sized GEMMs only 1.06x / 1.21x. If the damage needs the copy to be in flight while the
# all-reduce runs, copies released only at GEMM starts (and sized to end before the next all-reduce) should leave
# the all-reduces alone - the Flux idea (signal-driven overlap) applied to the PCIe link.
# Workload "mix" (decode pattern, eager, all 8 ranks): 64 x [GEMM 16x5120 @ 5120x6912, all_reduce(16x5120 bf16),
# GEMM, all_reduce]. Background modes (all 8 GPUs):
#   none
#   free      --bg_mb in 4 MiB copies, as fast as PCIe goes (k4_diag_v1)
#   gemm      one --chunk_kb copy per GEMM: the main stream writes tick = j (stream memory op) right before GEMM j,
#             the copy stream waits tick >= j, then copies chunk j  (released at GEMM starts)
#   ar        the same, released right before each all_reduce instead (control: copies on top of the all-reduces)
# direction h2d / d2h for each. Reported: mix ms (rank-max median), copy volume / time, slowdown vs none.
# PRE-REGISTERED for v3 [推論]: Q1 ticks alone cost 2-8%; Q2 d2h gemm-released minus ticks <= 3 points at every chunk
#   size; Q3 h2d gemm-released stays near free (>= 15%) at every chunk size (a different mechanism); Q4 free512 ~ free.
# Usage: torchrun 8 ranks (exclusive guard): k4_diag_v3.py --out <dir> [--rounds 30]
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
    tick = torch.zeros(1, dtype=torch.int64, device=dev)
    bgs = torch.cuda.Stream(dev)
    x = torch.randn(16, 5120, device=dev, dtype=torch.bfloat16)
    w = torch.randn(5120, 6912, device=dev, dtype=torch.bfloat16)
    small = torch.randn(16, 5120, device=dev, dtype=torch.bfloat16)
    base = [0]

    def copy(off, n, d):
        if d == "h2d":
            cm.memcpy(bgs, devbuf.data_ptr() + off, host.data_ptr() + off, n)
        else:
            cm.memcpy(bgs, host.data_ptr() + off, devbuf.data_ptr() + off, n)

    def one(mode, d):
        ck = {"gemm128": 128, "gemm512": 512, "gemm1024": 1024, "ar512": 512, "ticks": 512}.get(mode, 512) * 1024
        rel = "gemm" if mode.startswith("gemm") or mode == "ticks" else ("ar" if mode.startswith("ar") else None)
        dist.barrier()
        torch.cuda.synchronize()
        cs = torch.cuda.current_stream()
        b0, b1, e0, e1 = (torch.cuda.Event(enable_timing=True) for _ in range(4))
        n_chunks = 0
        if mode != "none":
            bgs.wait_stream(cs)
            b0.record(bgs)
            if mode in ("free", "free512"):
                step = ch if mode == "free" else 512 * 1024
                for off in range(0, nb, step):
                    copy(off, step, d)
            elif mode == "ticks":
                for j in range(128):
                    cm.wait_geq(bgs, tick.data_ptr(), base[0] + j + 1)
            else:  # 128 gated chunks, released by ticks base+1 .. base+128 written on the main stream
                for j in range(128):
                    cm.wait_geq(bgs, tick.data_ptr(), base[0] + j + 1)
                    copy((j * ck) % nb, ck, d)
                n_chunks = 128
            b1.record(bgs)
        e0.record(cs)
        j = 0
        for _ in range(64):
            for _ in range(2):
                if rel == "gemm":
                    j += 1
                    cm.write64(cs, tick.data_ptr(), base[0] + j)
                torch.mm(x, w)
                if rel == "ar":
                    j += 1
                    cm.write64(cs, tick.data_ptr(), base[0] + j)
                dist.all_reduce(small, group=g_ar)
        e1.record(cs)
        torch.cuda.synchronize()
        base[0] += 128
        if mode == "none":
            return e0.elapsed_time(e1), None, 0
        vol = nb if mode in ("free", "free512") else n_chunks * ck
        return e0.elapsed_time(e1), b0.elapsed_time(b1), vol

    cases = [("none", "h2d"), ("ticks", "h2d")] + [(mo, d) for mo in ("free", "free512", "gemm128", "gemm512", "gemm1024",
                                                                     "ar512") for d in ("h2d", "d2h")]
    for _ in range(3):
        for c in cases:
            one(*c)
    res = {f"{mo}|{d}": [] for mo, d in cases}
    for _ in range(args.rounds):
        for c in cases:
            res[f"{c[0]}|{c[1]}"].append(one(*c))
    allr = [None] * world
    dist.all_gather_object(allr, res)
    if rank == 0:
        out = {}
        n = args.rounds
        base_ms = statistics.median([max(allr[q]["none|h2d"][j][0] for q in range(world)) for j in range(n)])
        for k in res:
            ms = statistics.median([max(allr[q][k][j][0] for q in range(world)) for j in range(n)])
            o = dict(ms=ms, slowdown=ms / base_ms - 1)
            line = f"[diag3] {k:<9} mix {ms:8.3f} ms ({100 * (ms / base_ms - 1):+6.1f}%)"
            if allr[0][k][0][1] is not None and allr[0][k][0][2]:
                bg = statistics.median([allr[q][k][j][1] for q in range(world) for j in range(n)])
                vol = allr[0][k][0][2]
                o.update(bg_ms=bg, vol_MB=vol / 1e6, GBps=vol / bg / 1e6)
                line += f"   copies {vol / 1e6:6.1f} MB in {bg:6.2f} ms = {vol / bg / 1e6:5.2f} GB/s per GPU"
            out[k] = o
            print(line, flush=True)
        with open(os.path.join(args.out, "k4_diag_v3.json"), "w") as f:
            json.dump(dict(args=vars(args), summary=out, raw=allr), f)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
