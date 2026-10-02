################################################################################
# fusion-dispatch E2 / V8: timeline probe v2. v2 vs v1: items are INTERLEAVED per repetition in a
# random order (v1 ran each item as a block, so cross-item kernel durations mixed in clock drift).
# Original header:
# fusion-dispatch E2: timeline probe for design 3.2 / 5.5.
# Runs A_fused and D_fluxag_fluxgemm (and optionally others) at one (layer, M)
# with the gpu-mode protocol, `reps` times each, separated by long spin kernels
# so a parser can split iterations on the device timeline. Meant to run under
#   nsys profile -t cuda,nvtx --cuda-memory-usage=false -o <out> ...
# and be parsed by nsys_parse_v1.py. Timing numbers come from dispatch_map_v2.py;
# this only answers "what runs when".
################################################################################
import argparse
import os
import sys

import torch
import torch.cuda.nvtx as nvtx
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(__file__))
import dispatch_map_v2 as dm  # noqa: E402
from flux.testing import initialize_distributed  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--layer", required=True)
ap.add_argument("--M", type=int, required=True)
ap.add_argument("--items", default="A_fused,D_fluxag_fluxgemm,c_flux_ag,c_fluxgemm")
ap.add_argument("--reps", type=int, default=20)
args = ap.parse_args()

TP = initialize_distributed()
os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
torch.use_deterministic_algorithms(False)
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
dm.TP_GROUP, dm.RANK, dm.W = TP, TP.rank(), TP.size()
dm.LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
dm.NNODES = 1
dm.ARGS = argparse.Namespace(atol=0.05, rtol=0.05)
FLUSH = torch.empty(128 * 1024 * 1024 // 4, device="cuda")
ALIGN = torch.zeros(1, device="cuda")

layer = dm.Layer(args.layer, max(args.M, 16384), 1)
case = dm.Case(layer, args.M)
w = layer.weights[0]
items = args.items.split(",")
for _ in range(5):
    for it in items:
        case.fns[it](w)
torch.cuda.synchronize()
dist.barrier(TP)
import random  # noqa: E402
rng = random.Random(20261003)  # same order on every rank
for r in range(args.reps):
    order = items[:]
    rng.shuffle(order)
    for it in order:
        FLUSH.zero_()
        torch.cuda._sleep(2_000_000)  # ~1.5 ms gap: iteration separator on the timeline
        dist.all_reduce(ALIGN, group=TP)
        nvtx.range_push(f"{it}#{r}")
        case.fns[it](w)
        nvtx.range_pop()
    torch.cuda.synchronize()
    dist.barrier(TP)
dist.destroy_process_group()
