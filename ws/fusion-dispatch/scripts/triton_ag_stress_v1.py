################################################################################
# fusion-dispatch V9: is the Triton P2P AllGather prototype (ag_latency_v1.py:
# parallel flag barrier + P2P gather) correct when ranks are skewed?
#
# Each iteration, every rank:
#   1. sleeps a random 0..--max_skew_us on the GPU (per-rank, per-iteration skew)
#   2. writes a fresh shard  value = (it % 31) * 8 + rank  (exact in bf16) into its IPC buffer
#   3. barrier (epoch = it + 1), 4. gathers all peers' shards, 5. compares with the expected
#      pattern on the GPU (mismatch count accumulated on device; no host sync per iteration)
# Variants:
#   single   one IPC buffer: a fast rank may overwrite its shard for iteration it+1 while a slow
#            peer is still reading iteration it -> EXPECTED to fail under skew
#   double   two IPC buffers alternated by iteration parity: a shard is rewritten only after the
#            barrier of the next iteration, which every peer passes only after finishing its
#            previous gather (same-stream order) -> expected correct
# Usage: ./launch.sh triton_ag_stress_v1.py --iters 5000 --M 64 --K 8192 --out <json>
################################################################################
import argparse
import json
import os
import random
import sys
import time

import torch
import torch.distributed as dist
import triton

sys.path.insert(0, os.path.dirname(__file__))
from ag_latency_v1 import _p2p_barrier_par, _p2p_gather  # noqa: E402
import flux  # noqa: E402
from flux.testing import initialize_distributed  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--iters", type=int, default=5000)
ap.add_argument("--M", type=int, default=64)
ap.add_argument("--K", type=int, default=8192)
ap.add_argument("--max_skew_us", type=float, default=200.0)
ap.add_argument("--out", required=True)
a = ap.parse_args()
TP = initialize_distributed()
RANK, W = TP.rank(), TP.size()
DT = torch.bfloat16
m, K = a.M // W, a.K
chunk = m * K
clk_hz = 1.41e9  # A100 max SM clock; torch 2.6 device properties expose no clock_rate (2026-09-30 fix)
rng = random.Random(1000 + RANK)  # different skew per rank
res = {}
for variant in ("single", "double"):
    nbuf = 1 if variant == "single" else 2
    bufs = [flux.create_tensor_list([a.M * K], DT, TP) for _ in range(nbuf)]
    ptrs = [torch.tensor([t.data_ptr() for t in b], dtype=torch.int64, device="cuda") for b in bufs]
    flags = flux.create_tensor_list([W], torch.int32, TP)
    flags[RANK].zero_()
    fptr = torch.tensor([t.data_ptr() for t in flags], dtype=torch.int64, device="cuda")
    exp_row = torch.arange(W, device="cuda").repeat_interleave(chunk).to(torch.float32)  # shard index per element
    bad = torch.zeros(1, dtype=torch.int64, device="cuda")
    bad_iters = torch.zeros(1, dtype=torch.int64, device="cuda")
    torch.cuda.synchronize()
    dist.barrier(TP)
    grid = (triton.cdiv(chunk, 2048), W - 1)
    t0 = time.time()
    for it in range(a.iters):
        b = it % nbuf
        mine = bufs[b][RANK]
        torch.cuda._sleep(int(rng.random() * a.max_skew_us * 1e-6 * clk_hz))
        base = float((it % 31) * 8)
        mine[RANK * chunk:(RANK + 1) * chunk].fill_(base + RANK)
        _p2p_barrier_par[(W,)](fptr, RANK, it + 1, WORLD=W)
        _p2p_gather[grid](ptrs[b], mine, chunk, RANK, WORLD=W, BLOCK=2048)
        nb = (mine.float() != exp_row + base).sum()
        bad += nb
        bad_iters += (nb > 0).to(torch.int64)
    torch.cuda.synchronize()
    wall = time.time() - t0
    tot = torch.cat([bad, bad_iters])
    dist.all_reduce(tot, group=TP)
    res[variant] = {"iters": a.iters, "bad_elements_all_ranks": int(tot[0]), "bad_rank_iterations": int(tot[1]),
                    "wall_s": wall}
    if RANK == 0:
        print(f"[{variant}] iters={a.iters} M={a.M} K={K} max_skew={a.max_skew_us}us  "
              f"bad elements={int(tot[0])}  bad (rank,iter) pairs={int(tot[1])}  ({wall:.1f}s)", file=sys.stderr, flush=True)
if RANK == 0:
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"args": vars(a), "results": res, "date": time.strftime("%Y-%m-%d %H:%M:%S")}, open(a.out, "w"), indent=1)
dist.barrier(TP)
os._exit(0)
