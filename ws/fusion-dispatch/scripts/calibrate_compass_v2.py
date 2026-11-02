################################################################################
# fusion-dispatch C1 v2 (2026-11-02): calibrate_compass_v1.py with decomp_v2 (PyNccl + side stream; the comm items use
# the same PyNccl communicator), run in the vLLM env through vLLM's distributed init (as validate_block_v6.py).
# v1 header:
# fusion-dispatch C1 (2026-11-02): calibration data for the Compass selector (arXiv 2609.10549, Sec. IV-C: "a one-time,
# offline linear regression" of the decomposition overhead eta(d) = alpha + beta * d, Eq. 2-3). Per op of one
# transformer layer (TP sharding as validate_block_v5.py: qkv / gate_up = AG+GEMM, o / down = GEMM+RS) and per M:
#   comp   the op's GEMM alone at full M (per-rank shape)          -> T_comp
#   comm   the op's collective alone (AG of (M/W, K) / RS of (M, N)) -> T_comm
#   dec<d> the op decomposed into d pipelined chunks (decomp_v1.DecompDispatcher), d in --ds where d divides M/W
# Protocol as validate_block_v4/v5: items interleaved per round in a shared random order, L2 flush, GPU sleep pad,
# 1-element all_reduce alignment, CUDA events, per-round rank-max, median over rounds. Calibration M values must be
# disjoint from the test points (run_c1_v1.sh). Run through run_xnode.sh (cross-node) or launch_tp.sh (one node).
# Output: <out_dir>/compass_cal.csv (model, W, op, kind, n, k, M, item, median_ms, p10_ms, p90_ms), meta_compass_cal.json
################################################################################
import argparse
import csv
import json
import os
import random
import statistics
import sys
import time

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decomp_v2 import DecompDispatcher  # noqa: E402

MODELS = {"qwen2.5-32b": (5120, 40, 8, 128, 27648), "llama3-8b": (4096, 32, 8, 128, 14336)}  # H, NQ, NKV, HD, FFN
DT = torch.bfloat16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--Ms", required=True)
    ap.add_argument("--ds", default="2,4,8,16")
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--pad_cycles", type=int, default=400_000)
    ap.add_argument("--seed", type=int, default=20261102)
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    from vllm.distributed.parallel_state import get_tp_group
    lr = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(lr)
    init_distributed_environment(world_size=int(os.environ["WORLD_SIZE"]), rank=int(os.environ["RANK"]),
                                 distributed_init_method="env://", local_rank=lr, backend="nccl")
    initialize_model_parallel(tensor_model_parallel_size=int(os.environ["WORLD_SIZE"]))
    pyn = get_tp_group().device_communicator.pynccl_comm
    assert not pyn.disabled
    rank, W = dist.get_rank(), dist.get_world_size()
    os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)  # production cuBLAS
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    H, NQ, NKV, HD, FFN = MODELS[a.model]
    ops = {  # name: (kind, out width, in width) per rank
        "qkv": ("ag", (NQ // W + 2 * NKV // W) * HD, H),
        "gate_up": ("ag", 2 * FFN // W, H),
        "o": ("rs", H, NQ // W * HD),
        "down": ("rs", H, FFN // W),
    }
    ds = [int(v) for v in a.ds.split(",")]
    g = torch.Generator(device="cuda").manual_seed(7 + rank)
    Wt = {op: (torch.randn(n, k, device="cuda", generator=g) * 0.02).to(DT) for op, (_, n, k) in ops.items()}
    decs = {d: DecompDispatcher(dist.group.WORLD, pyn, d=d) for d in [1] + ds}
    FLUSH = torch.empty(128 * 1024 * 1024 // 4, device="cuda")
    ALIGN = torch.zeros(1, device="cuda")
    rng = random.Random(a.seed)
    rows, meta = [], {"args": vars(a), "world": W, "ops": ops, "date": time.strftime("%Y-%m-%d %H:%M:%S")}
    for M in [int(v) for v in a.Ms.split(",")]:
        m = M // W
        items = {}
        for op, (kind, n, k) in ops.items():
            w = Wt[op]
            if kind == "ag":
                x = torch.randn(m, k, device="cuda", generator=g).to(DT)
                full = torch.randn(M, k, device="cuda", generator=g).to(DT)
                gbuf = torch.empty(M, k, device="cuda", dtype=DT)
                items[(op, "comp")] = lambda full=full, w=w: torch.mm(full, w.t())
                items[(op, "comm")] = lambda x=x, gbuf=gbuf: pyn.all_gather(gbuf, x)
                for d in ds:
                    if m % d == 0:
                        items[(op, f"dec{d}")] = lambda x=x, w=w, d=d: decs[d].ag_gemm(x, w)
            else:
                x = torch.randn(M, k, device="cuda", generator=g).to(DT)
                part = torch.randn(M, n, device="cuda", generator=g).to(DT)
                obuf = torch.empty(m, n, device="cuda", dtype=DT)
                items[(op, "comp")] = lambda x=x, w=w: torch.mm(x, w.t())
                items[(op, "comm")] = lambda part=part, obuf=obuf: pyn.reduce_scatter(obuf, part)
                for d in ds:
                    if m % d == 0:
                        items[(op, f"dec{d}")] = lambda x=x, w=w, d=d: decs[d].gemm_rs(x, w)
        keys = list(items)
        for _ in range(a.warmup):
            for kk in keys:
                items[kk]()
        torch.cuda.synchronize()
        t = torch.zeros(a.rounds, len(keys), device="cuda", dtype=torch.float64)
        for r in range(a.rounds):
            order = list(range(len(keys)))
            rng.shuffle(order)
            evs = {}
            for j in order:
                FLUSH.zero_()
                torch.cuda._sleep(a.pad_cycles)
                dist.all_reduce(ALIGN)
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record()
                items[keys[j]]()
                e1.record()
                evs[j] = (e0, e1)
            torch.cuda.synchronize()
            for j, (e0, e1) in evs.items():
                t[r, j] = e0.elapsed_time(e1)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)  # per-round rank-max
        if rank == 0:
            for j, (op, item) in enumerate(keys):
                v = sorted(t[:, j].tolist())
                kind, n, k = ops[op]
                rows.append([a.model, W, op, kind, n, k, M, item, f"{statistics.median(v):.5f}",
                             f"{v[len(v) // 10]:.5f}", f"{v[9 * len(v) // 10]:.5f}"])
            print(f"[compass_cal {a.model} W={W} M={M}] " + "  ".join(
                f"{op}/{it}={statistics.median(t[:, j].tolist()):.3f}" for j, (op, it) in enumerate(keys)),
                file=sys.stderr, flush=True)
    if rank == 0:
        os.makedirs(a.out_dir, exist_ok=True)
        with open(os.path.join(a.out_dir, f"compass_cal_{a.model}.csv"), "w", newline="") as f:
            w_ = csv.writer(f)
            w_.writerow(["model", "W", "op", "kind", "n", "k", "M", "item", "median_ms", "p10_ms", "p90_ms"])
            w_.writerows(rows)
        json.dump(meta, open(os.path.join(a.out_dir, f"meta_compass_cal_{a.model}.json"), "w"), indent=1)
    dist.barrier()
    os._exit(0)


if __name__ == "__main__":
    main()
