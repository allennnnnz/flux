################################################################################
# Host-side launch cost of the GEMM call styles used by the B/C arms.
# Question (E0): dispatch_map_v1 shows ~0.10 ms CPU per torch.mm(out=) call,
# which makes host/steady modes launch-bound at small M. Is that intrinsic to
# torch.mm, to the out= variant, or to the harness (clock-logger thread)?
#
# Single process, one GPU. GPU is kept busy by a long sleep kernel so launches
# never wait on the device; CPU time per call = perf_counter over n calls / n.
# Usage: pixi run --manifest-path pixi.toml python ws/fusion-dispatch/scripts/launch_overhead_v1.py
################################################################################
import os
import statistics
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "common", "measure"))
from clock_logger import ClockLogger  # noqa: E402

torch.cuda.set_device(0)
dt = torch.bfloat16
M, n, K = 64, 6144, 12288
a = torch.randn(M, K, device="cuda", dtype=dt)
w = torch.randn(n, K, device="cuda", dtype=dt) * 0.01
o = torch.empty(M, n, device="cuda", dtype=dt)
wt_contig = w.t().contiguous()

styles = {
    "mm_out(w.t())": lambda: torch.mm(a, w.t(), out=o),
    "mm(w.t())": lambda: torch.mm(a, w.t()),
    "matmul(w.t())": lambda: torch.matmul(a, w.t()),
    "F.linear(w)": lambda: F.linear(a, w),
    "mm_out(wT contig)": lambda: torch.mm(a, wt_contig, out=o),
    "empty_like(o)": lambda: torch.empty_like(o),
}


def measure(fn, n_calls=100, reps=30):
    per = []
    for _ in range(reps):
        torch.cuda._sleep(200_000_000)  # keep the device busy (>100 ms) so launches never block
        t0 = time.perf_counter()
        for _ in range(n_calls):
            fn()
        per.append((time.perf_counter() - t0) / n_calls * 1e6)
        torch.cuda.synchronize()
    return statistics.median(per), min(per), max(per)


print(f"torch {torch.__version__}  CUDA_DEVICE_MAX_CONNECTIONS={os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS')}")
for fn in styles.values():  # warmup incl. cuBLAS handle / heuristics
    for _ in range(20):
        fn()
torch.cuda.synchronize()
for label, logger in [("no clock logger", False), ("clock logger 20 ms", True)]:
    print(f"\n== {label}")
    ctx = ClockLogger(0, 0.02) if logger else None
    if ctx:
        ctx.__enter__()
    for name, fn in styles.items():
        med, lo, hi = measure(fn)
        print(f"  {name:<20} median {med:7.1f} us/call   min {lo:7.1f}  max {hi:7.1f}")
    if ctx:
        ctx.__exit__(None, None, None)
