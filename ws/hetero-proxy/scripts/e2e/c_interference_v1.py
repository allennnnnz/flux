################################################################################
# hetero-proxy E2E C (2026-10-14): is c_allreduce_diag_v1's (f) real? NCCL all-reduce went from 0.15-0.75 ms alone to
# ~4.6 ms with PCIe copies in the background. Controls, per round on every rank: background enqueued on side streams,
# then NCCL all-reduce of S on the main stream, CUDA events around the all-reduce; the host time to enqueue the
# background is recorded (a blocking cuMemcpyAsync would desynchronise the ranks' entry into the collective).
#   bg = none | d2h (6 x 4 MiB) | h2d (6 x 4 MiB) | both | d2d (6 x 4 MiB device-to-device, no PCIe)
#   also "both_presync": background enqueued, then a barrier (all ranks' host side done), then the all-reduce.
# Usage: torchrun 8 ranks (as c_allreduce_diag_v1.py) --out <dir>
################################################################################
import argparse
import json
import os
import statistics
import sys
import time

import torch
import torch.distributed as dist

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from hxb import cudamem as cm  # noqa: E402
from c_allreduce_bw_v1 import HostAllReduce  # noqa: E402

MiB = 2 ** 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=20)
    args = ap.parse_args()
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    lr = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(lr)
    dev = torch.device(f"cuda:{lr}")
    os.sched_setaffinity(0, list(range(4 * rank, 4 * rank + 4)))
    har = HostAllReduce(rank, dev, 16 * MiB)
    slot = har.base + 4096 + rank * har.slot
    result = har.base + 4096 + 8 * har.slot
    g = torch.randn(8 * MiB, device=dev).to(torch.bfloat16)
    g2 = torch.empty_like(g)
    s1, s2 = torch.cuda.Stream(dev), torch.cuda.Stream(dev)

    def bg(kind):
        for _ in range(6):
            if kind in ("d2h", "both", "both_presync"):
                cm.memcpy(s1, slot, g.data_ptr(), 4 * MiB)
            if kind in ("h2d", "both", "both_presync"):
                cm.memcpy(s2, g.data_ptr(), result, 4 * MiB)
            if kind == "d2d":
                cm.memcpy(s1, g2.data_ptr(), g.data_ptr(), 4 * MiB)
    res = {}
    for S in (10, 80):
        t = torch.randn(int(S * MiB) // 2, device=dev).to(torch.bfloat16)
        for kind in ("none", "d2h", "h2d", "both", "both_presync", "d2d"):
            ts, hs = [], []
            for r in range(args.reps + 2):
                dist.barrier()
                torch.cuda.synchronize()
                h0 = time.perf_counter()
                bg(kind)
                hs.append((time.perf_counter() - h0) * 1e3)
                if kind == "both_presync":
                    dist.barrier()
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record()
                dist.all_reduce(t)
                e1.record()
                e1.synchronize()
                if r >= 2:
                    ts.append(e0.elapsed_time(e1))
            allv = [None] * 8
            dist.all_gather_object(allv, (statistics.median(ts), statistics.median(hs[2:])))
            res[f"{S}|{kind}"] = dict(nccl_ms_rank_max=max(v[0] for v in allv), host_enqueue_ms_max=max(v[1] for v in allv))
            if rank == 0:
                print(f"[interf] {S:3d} MiB  bg={kind:<13} NCCL {res[f'{S}|{kind}']['nccl_ms_rank_max']:7.3f} ms (rank-max)   "
                      f"host enqueue of bg {res[f'{S}|{kind}']['host_enqueue_ms_max']:7.3f} ms (max over ranks)", flush=True)
    if rank == 0:
        json.dump(res, open(os.path.join(args.out, "c_interference.json"), "w"), indent=1)
    dist.barrier()
    har.close()
    torch.cuda.synchronize()
    dist.destroy_process_group()  # a normal exit lets nsys / CUPTI flush the kernel records (os._exit lost them)


if __name__ == "__main__":
    main()
