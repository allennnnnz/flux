################################################################################
# Where does the ~0.10 ms host time per torch.mm inside dispatch_map_v1 come from?
# launch_overhead_v1.py (single process) measures 13 us per call. This repeats the
# measurement under torchrun, step by step: bare, after NCCL, after Flux op
# creation, and inside the harness item sequence (flush, sleep, align, mm).
# Usage: pixi run --manifest-path pixi.toml ./launch.sh ws/fusion-dispatch/scripts/launch_overhead_dist_v1.py
################################################################################
import os
import statistics
import sys
import time
from functools import partial

import torch
import torch.distributed as dist

import flux
from flux.testing import initialize_distributed

print = partial(print, file=sys.stderr, flush=True)
dt = torch.bfloat16


def mm_cpu(fn, reps=30, n=50):
    per = []
    for _ in range(reps):
        torch.cuda._sleep(100_000_000)
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        per.append((time.perf_counter() - t0) / n * 1e6)
        torch.cuda.synchronize()
    return statistics.median(per)


def report(tag, vals):
    t = torch.tensor(vals, dtype=torch.float64, device="cuda")
    g = [torch.zeros_like(t) for _ in range(W)]
    dist.all_gather(g, t)
    if RANK == 0:
        arr = torch.stack(g).cpu()
        print(f"{tag:<44} " + "  ".join(f"{n}: med {arr[:, j].median():6.1f} max {arr[:, j].max():6.1f}"
                                        for j, n in enumerate(NAMES)))


TP = initialize_distributed()
RANK, W = TP.rank(), TP.size()
M, n, K = 64, 6144, 12288
x = torch.randn(M // W, K, device="cuda", dtype=dt)
full = torch.randn(M, K, device="cuda", dtype=dt)
w = torch.randn(n, K, device="cuda", dtype=dt) * 0.01
o = torch.empty(M, n, device="cuda", dtype=dt)
FLUSH = torch.empty(128 * 1024 * 1024 // 4, device="cuda")
ALIGN = torch.zeros(1, device="cuda")
NAMES = ["mm_out", "F.linear"]
fns = [lambda: torch.mm(full, w.t(), out=o), lambda: torch.nn.functional.linear(full, w)]
for f in fns:
    for _ in range(20):
        f()
torch.cuda.synchronize()
report("1 torchrun, NCCL initialized", [mm_cpu(f) for f in fns])

op = flux.AGKernel(TP, 1, 16384, n, K, dt, output_dtype=dt)
ag = flux.AllGatherOp(TP, 1, 16384, K, dt)
torch.cuda.synchronize()
report("2 after AGKernel + AllGatherOp created", [mm_cpu(f) for f in fns])

opt = flux.AllGatherOption()
opt.mode = flux.AGRingMode.All2All
opt.use_read = True
opt.use_cuda_core_local = False
opt.use_cuda_core_ag = False
opt.fuse_sync = False
opt.input_buffer_copied = False
out_a = torch.empty(M, n, device="cuda", dtype=dt)
for _ in range(10):
    op.forward(x, w, output=out_a, transpose_weight=False, all_gather_option=opt)
torch.cuda.synchronize()
report("3 after AGKernel.forward has run", [mm_cpu(f) for f in fns])


def harness_seq(f, reps=100):
    per = []
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(reps):
        FLUSH.zero_()
        torch.cuda._sleep(400_000)
        dist.all_reduce(ALIGN, group=TP)
        s.record()
        c0 = time.perf_counter()
        f()
        c1 = time.perf_counter()
        e.record()
        per.append((c1 - c0) * 1e6)
    torch.cuda.synchronize()
    return statistics.median(per)


report("4 harness sequence (flush, sleep, allreduce)", [harness_seq(f) for f in fns])


def harness_seq_no_ar(f, reps=100):
    per = []
    for _ in range(reps):
        FLUSH.zero_()
        torch.cuda._sleep(400_000)
        c0 = time.perf_counter()
        f()
        per.append((time.perf_counter() - c0) * 1e6)
    torch.cuda.synchronize()
    return statistics.median(per)


report("5 same without allreduce", [harness_seq_no_ar(f) for f in fns])


def after_ar_only(f, reps=100):
    per = []
    for _ in range(reps):
        torch.cuda._sleep(400_000)
        dist.all_reduce(ALIGN, group=TP)
        c0 = time.perf_counter()
        f()
        per.append((time.perf_counter() - c0) * 1e6)
    torch.cuda.synchronize()
    return statistics.median(per)


report("6 right after allreduce (no flush)", [after_ar_only(f) for f in fns])
dist.destroy_process_group()
