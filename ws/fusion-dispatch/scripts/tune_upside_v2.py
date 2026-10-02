################################################################################
# fusion-dispatch V10 (tune_upside v2): v1 + GEMM+RS layers (flux.GemmRS.profiling).
# v1 header follows.
# fusion-dispatch E2a: how much could per-M tuning (the A* arm) buy, WITHOUT
# rebuilding Flux? (D-006: no src/ change before evidence.)
#
# AGKernel.profiling() (src/ag_gemm/ths_op/all_gather_gemm_op.cc:413-455) runs the
# fused forward for every registered hparams: 5 warmup + 10 back-to-back iters,
# rank-max of the mean, 1 s sleep between configs. It returns top-k strings.
# Its protocol differs from dispatch_map_v2 (warm, back-to-back, 10 iters), so
# this script also times the DEFAULT forward (registry or fallback hparams) with
# the same back-to-back protocol, and reports best/default. The A* estimate is
#   A*_est(M) = A_E1(M) * best/default
# which is an estimate [推論], not a measurement of A*.
# Usage (repo root): ./launch.sh ws/fusion-dispatch/scripts/tune_upside_v1.py --layer G-FC1 --Ms 64,1024
################################################################################
import argparse
import json
import os
import re
import sys
import time

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(__file__))
import dispatch_map_v2 as dm  # noqa: E402
import dispatch_map_rs_v1 as rs  # noqa: E402
import flux  # noqa: E402
from flux.testing import initialize_distributed  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--layer", required=True)
ap.add_argument("--Ms", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

TP = initialize_distributed()
os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
torch.use_deterministic_algorithms(False)
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
dm.TP_GROUP, dm.RANK, dm.W = TP, TP.rank(), TP.size()
dm.LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
dm.NNODES = 1
dm.ARGS = argparse.Namespace(atol=0.05, rtol=0.05)
rs.TP, rs.RANK, rs.W, rs.ARGS = TP, dm.RANK, dm.W, argparse.Namespace(atol=0.1, rtol=0.05)
Ms = [int(x) for x in args.Ms.split(",")]
IS_RS = args.layer in rs.LAYERS
layer = (rs.LayerRS if IS_RS else dm.Layer)(args.layer, max(16384, max(Ms)), 1)
w = layer.weights[0]


def backtoback(fn, warmup=5, iters=10, reps=5):
    """Same shape of protocol as profiling(): mean of back-to-back iters, rank-max."""
    vals = []
    for _ in range(reps):
        dist.barrier(TP)
        for _ in range(warmup):
            fn()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        t = torch.tensor([s.elapsed_time(e) / iters], device="cuda")
        dist.all_reduce(t, op=dist.ReduceOp.MAX, group=TP)
        vals.append(t.item())
    vals.sort()
    return vals[len(vals) // 2]


out = {}
for M in Ms:
    case = (rs.CaseRS if IS_RS else dm.Case)(layer, M)
    fn = case.fns["A_fused"]
    t_default = backtoback(lambda: fn(w))
    ctx = flux.ProfilingContext(f"{args.layer}_M{M}")
    t0 = time.time()
    if IS_RS:
        case.layer.op.profiling(case.x, w, prof_ctx=ctx)
    else:
        case.layer.op.profiling(case.x, w, output=case.out["A"], transpose_weight=False,
                                all_gather_option=case.layer.opt, prof_ctx=ctx)
    torch.cuda.synchronize()
    wall = time.time() - t0
    txt = "\n".join(ctx.get_all_prof_results())
    times = [float(x) for x in re.findall(r"TopK=\d+ \(([\d.]+) ms\)", txt)]
    t_after = backtoback(lambda: fn(w))  # default again, drift check
    rec = {"M": M, "default_backtoback_ms": t_default, "default_backtoback_after_ms": t_after,
           "profiling_topk_ms": times, "best_ms": min(times) if times else None,
           "best_over_default": (min(times) / t_default) if times else None,
           "profiling_wall_s": wall, "raw": txt}
    out[M] = rec
    if dm.RANK == 0:
        print(f"[{args.layer} M={M}] default {t_default:.4f} (after {t_after:.4f})  "
              f"profiling top-k {times}  best/default {rec['best_over_default']:.3f}  ({wall:.0f} s)",
              file=sys.stderr, flush=True)
    del case
if dm.RANK == 0:
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"layer": args.layer, "date": time.strftime("%Y-%m-%d %H:%M:%S"), "results": out},
              open(args.out, "w"), indent=1)
    print(f"wrote {args.out}", file=sys.stderr)
dist.destroy_process_group()
