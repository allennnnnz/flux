################################################################################
# fusion-dispatch V6: why does NCCL all_gather read 0.565 ms (M=4096, K=12288) with the Phase 0
# script (reproduced 3x on 2026-09-30, results/v6_phase0_repro/) but 0.478-0.509 ms with
# dispatch_map_v2? Flux AllGather reads the same (~0.47) under both. Same round, interleaved
# (shared random order), ops x protocols:
#   P0        dist.barrier(); start; op; end                (Phase 0 flux_comm_baseline._bench)
#   P0_flush  128 MB L2 flush; dist.barrier(); start; op; end
#   host      flush; cuda.synchronize(); dist.barrier(); start; op; end   (dispatch_map host mode)
#   gpu       flush; GPU sleep pad; 1-elem NCCL all_reduce; start; op; end (dispatch_map gpu mode)
# ops: nccl = dist.all_gather_into_tensor, flux = flux.AllGatherOp.run (All2All pull), fp16.
# Both statistics are reported from the same samples:
#   phase0_stat = max over ranks of the per-rank median   (flux_comm_baseline.py)
#   ours_stat   = median over rounds of the per-round rank-max (dispatch_map_v2)
################################################################################
import argparse
import itertools
import json
import os
import random
import statistics
import sys

import torch
import torch.distributed as dist

import flux
from flux.testing import initialize_distributed

ap = argparse.ArgumentParser()
ap.add_argument("--Ms", default="1024,4096,16384")
ap.add_argument("--K", type=int, default=12288)
ap.add_argument("--rounds", type=int, default=200)
ap.add_argument("--out", required=True)
a = ap.parse_args()
TP = initialize_distributed()
RANK, W = TP.rank(), TP.size()
DT = torch.float16
FLUSH = torch.empty(128 * 1024 * 1024 // 4, device="cuda")
ALIGN = torch.zeros(1, device="cuda")
rng = random.Random(20261005)
res = {}
for M in [int(x) for x in a.Ms.split(",")]:
    x = torch.randn(M // W, a.K, device="cuda").to(DT)
    out = torch.empty(M, a.K, device="cuda", dtype=DT)
    ag = flux.AllGatherOp(TP, 1, M, a.K, DT)
    o = flux.AllGatherOption()
    o.mode, o.use_read, o.use_cuda_core_local, o.use_cuda_core_ag = flux.AGRingMode.All2All, True, False, False
    o.fuse_sync, o.input_buffer_copied = False, False
    ops = {"nccl": lambda: dist.all_gather_into_tensor(out, x, group=TP),
           "flux": lambda: ag.run(x, None, o, torch.cuda.current_stream().cuda_stream)}
    prots = ["P0", "P0_flush", "host", "gpu"]
    combos = list(itertools.product(ops, prots))
    for _ in range(5):
        for op_ in ops.values():
            op_()
    torch.cuda.synchronize()
    ev = {c: [] for c in combos}
    for r in range(a.rounds):
        order = combos[:]
        rng.shuffle(order)
        for opn, pr in order:
            if pr in ("P0_flush", "host", "gpu"):
                FLUSH.zero_()
            if pr == "host":
                torch.cuda.synchronize()
            if pr in ("P0", "P0_flush", "host"):
                dist.barrier(TP)
            else:
                torch.cuda._sleep(400_000)
                dist.all_reduce(ALIGN, group=TP)
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            ops[opn]()
            e.record()
            ev[(opn, pr)].append((s, e))
        torch.cuda.synchronize()
    t = torch.tensor([[s.elapsed_time(e) for s, e in ev[c]] for c in combos], device="cuda", dtype=torch.float64)
    g = [torch.zeros_like(t) for _ in range(W)]
    dist.all_gather(g, t, group=TP)
    if RANK == 0:
        T = torch.stack(g).cpu()  # (W, combos, rounds)
        res[M] = {}
        for j, (opn, pr) in enumerate(combos):
            per_rank_med = [statistics.median(T[rk, j].tolist()) for rk in range(W)]
            per_round_max = T[:, j].max(0).values.tolist()
            res[M][f"{opn}/{pr}"] = {"phase0_stat_ms": max(per_rank_med), "ours_stat_ms": statistics.median(per_round_max)}
        line = "  ".join(f"{k}: p0stat {v['phase0_stat_ms']:.3f} ours {v['ours_stat_ms']:.3f}" for k, v in res[M].items())
        print(f"[M={M}] {line}", file=sys.stderr, flush=True)
    del ag
if RANK == 0:
    json.dump(res, open(a.out, "w"), indent=1)
dist.barrier(TP)
os._exit(0)
