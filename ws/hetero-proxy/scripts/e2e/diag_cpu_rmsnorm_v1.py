################################################################################
# hetero-proxy E2E diagnostic (2026-10-14): cost of the CPU backend's rmsnorm op by row count (5120 bf16 per row),
# timed from the device trace; first call per size dropped. Question from bench_v2 run1: is Bmid1's 87 ms hop
# (8192 rows) the CPU norm itself? Usage: python diag_cpu_rmsnorm_v1.py [--reps 6]
################################################################################
import argparse
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from hxb.cpu_backend import CpuBackend  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=6)
    args = ap.parse_args()
    os.sched_setaffinity(0, list(range(0, 32)))
    be = CpuBackend()
    q = be.queue()
    s = be.signal("s")
    H = 5120
    x, y = be.alloc((16384, H), torch.bfloat16), be.alloc((16384, H), torch.bfloat16)
    w = be.alloc((H,), torch.bfloat16)
    v = 0
    for rows in (64, 1024, 2048, 8192, 16384):
        be.trace()
        for _ in range(args.reps + 1):
            v += 1
            be.launch(q, "rmsnorm", {"x": x, "w": w, "y": y, "rows": (0, rows), "eps": 1e-6}, done=(s, v), tag="n")
            s.wait(v, timeout=120)
        tr = sorted(be.trace(), key=lambda r: r["t_start"])[1:]
        ms = statistics.median(r["t_end"] - r["t_start"] for r in tr) * 1e3
        print(f"[cpu-rmsnorm] rows {rows:6d} ({rows * H * 2 / 2**20:6.1f} MiB bf16): {ms:8.2f} ms  "
              f"-> {rows * H * 2 / ms / 1e6:6.2f} GB/s of input", flush=True)
    be.close()


if __name__ == "__main__":
    main()
