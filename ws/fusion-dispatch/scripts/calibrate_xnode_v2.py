################################################################################
# fusion-dispatch E4a2 (2026-10-05): cross-node calibration v2. v2 vs v1 (calibrate_xnode_v1.py): measures at
# the EXACT (H, M) pairs the block will use (--cases "H:M,M,...;H:M,..."), because in E4a the block's message
# sizes fell between v1's grid points (H=6144, M = powers of 2) and narrow NCCL cliffs were missed
# (reports/20261005_e4a_slow_link_layout.md section 3). These are single-collective microbenchmarks
# (component times), not block measurements. Everything else as in v1:
# ----- v1 header -----
# fusion-dispatch E4a (2026-10-05): cross-node calibration for the LAYOUT decision (css-host-158 + 159).
# Copy of calibrate_block_v1.py (same protocol: dispatch_map_v2.run_mode, interleaved, L2 flush,
# gpu / steady, rank-max, clock filter) with two more items, because the sequence-parallel side of the
# layout model needs NCCL all_gather / reduce_scatter curves measured on THIS interconnect:
#   vllm_ar   vLLM tensor_model_parallel_all_reduce (M, 6144) (cross-node: CustomAllreduce is off, PyNccl)
#   nccl_ag   torch all_gather_into_tensor (M/W, 6144) -> (M, 6144)
#   nccl_rs   torch reduce_scatter_tensor  (M, 6144) -> (M/W, 6144)
#   rmsnorm, add   as in v1
# Hidden size 6144 belongs to no model in the test set. Launch: run_xnode.sh <out> <ppn> vllm <this> --out_dir <out>
# (NVSHMEM_HCA_LIST=mlx5_0: initialize_distributed() starts NVSHMEM, which needs all-pairs reachability).
# Output: raw_xnode_cal.csv, summary_xnode_cal.csv, meta_xnode_cal.json
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

sys.path.insert(0, os.path.dirname(__file__))
import dispatch_map_v2 as dm  # noqa: E402
from validate_block_v3 import rmsnorm  # noqa: E402  (same fused kernel the block runs)

H = 6144
MS = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
DT = torch.bfloat16


class _W:
    def __init__(self, n):
        self.weights = [None] * n


class Case:
    def __init__(self, M, L, H):
        self.M, self.layer = M, _W(L)
        self.x = torch.randn((M, H), device="cuda").to(DT)
        self.y = torch.randn((M, H), device="cuda").to(DT)
        self.ar = torch.randn((M, H), device="cuda").to(DT)
        self.lnw = torch.ones(H, device="cuda", dtype=DT)
        self.shard = torch.randn((M // W, H), device="cuda").to(DT)
        self.full = torch.empty((M, H), device="cuda", dtype=DT)
        self.rs_out = torch.empty((M // W, H), device="cuda", dtype=DT)
        self.fns = {"vllm_ar": lambda w: VLLM_AR(self.ar),
                    "nccl_ag": lambda w: dist.all_gather_into_tensor(self.full, self.shard, group=TP),
                    "nccl_rs": lambda w: dist.reduce_scatter_tensor(self.rs_out, self.y, group=TP),
                    "rmsnorm": lambda w: rmsnorm(self.x, self.lnw),
                    "add": lambda w: torch.add(self.x, self.y)}


def main():
    modes = ARGS.modes.split(",")
    L = ARGS.steady_L if "steady" in modes else 1
    rng = random.Random(ARGS.seed)
    if RANK == 0:
        os.makedirs(ARGS.out_dir, exist_ok=True)
    rows, summ, first = [], [], True
    t0 = time.time()
    cases = [(int(h), int(m)) for part in ARGS.cases.split(";") for h, ms in [part.split(":")] for m in ms.split(",")]
    for H, M in cases:
        case = Case(M, L, H)
        items = list(case.fns)
        for mode in modes:
            rounds = ARGS.rounds if mode == "gpu" else ARGS.rounds_steady
            T, C, Kc, orders, kept, modal = dm.run_mode(case, items, mode, rounds, ARGS.warmup_first if first else ARGS.warmup, rng)
            first = False
            if RANK != 0:
                continue
            rmax = T.max(0).values
            ks = [r for r in range(T.shape[1]) if kept[r]] or list(range(T.shape[1]))
            for r in range(T.shape[1]):
                for j, it in enumerate(items):
                    rows.append([M, H, mode, r, it, f"{rmax[r, j]:.5f}", int(kept[r])])
            for j, it in enumerate(items):
                summ.append([M, H, mode, it, f"{statistics.median(rmax[r, j].item() for r in ks):.5f}", len(ks)])
            print(f"[xnode-cal2 tp{W} H={H} M={M} {mode}] " + "  ".join(f"{s[3]}={s[4]}" for s in summ[-len(items):]),
                  file=sys.stderr, flush=True)
        del case
    if RANK == 0:
        with open(os.path.join(ARGS.out_dir, "raw_xnode_cal.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["M", "H", "mode", "round", "item", "rank_max_ms", "kept"])
            w.writerows(rows)
        with open(os.path.join(ARGS.out_dir, "summary_xnode_cal.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["M", "H", "mode", "item", "median_ms", "n_kept"])
            w.writerows(summ)
        json.dump({"script": "ws/fusion-dispatch/scripts/calibrate_xnode_v2.py", "hosts": os.uname().nodename, "args": vars(ARGS), "world": W,
                   "date": time.strftime("%Y-%m-%d %H:%M:%S"), "wall_s": time.time() - t0,
                   "custom_ar": {"present": CA is not None, "disabled": getattr(CA, "disabled", None),
                                 "max_size": getattr(CA, "max_size", None)}},
                  open(os.path.join(ARGS.out_dir, "meta_xnode_cal.json"), "w"), indent=1)
    dist.barrier()
    torch.cuda.synchronize()
    os._exit(0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default="gpu,steady")
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--rounds_steady", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--warmup_first", type=int, default=100)
    ap.add_argument("--steady_L", type=int, default=16)
    ap.add_argument("--pad_cycles", type=int, default=400_000)
    ap.add_argument("--flush_mb", type=int, default=128)
    ap.add_argument("--clock_period", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=20261003)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--cases", required=True, help='"H:M,M,...;H:M,..." exact (hidden, tokens) pairs')
    ARGS = ap.parse_args()
    from flux.testing import initialize_distributed
    TP = initialize_distributed()
    RANK, W = TP.rank(), TP.size()
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    from vllm.distributed import init_distributed_environment, initialize_model_parallel, tensor_model_parallel_all_reduce
    from vllm.distributed.parallel_state import get_tp_group
    init_distributed_environment(world_size=W, rank=RANK, distributed_init_method="env://", local_rank=LOCAL_RANK,
                                 backend="nccl")
    initialize_model_parallel(tensor_model_parallel_size=W)
    CA = get_tp_group().device_communicator.ca_comm
    VLLM_AR = tensor_model_parallel_all_reduce
    dm.TP_GROUP, dm.RANK, dm.W, dm.LOCAL_RANK, dm.ARGS = TP, RANK, W, LOCAL_RANK, ARGS
    dm.FLUSH = torch.empty(ARGS.flush_mb * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
    dm.ALIGN = torch.zeros(1, dtype=torch.float32, device="cuda")
    main()
