################################################################################
# fusion-dispatch G4 (D-009): calibration for the LAYOUT decision (vLLM default TP + all-reduce vs
# sequence parallel). Must run in the vLLM venv:
#   TP=<n> bash ws/fusion-dispatch/scripts/launch_vllm_env.sh calibrate_block_v1.py --out_dir <dir>
# Measures, with the dispatch_map_v2.run_mode protocol (interleaved, L2 flush, gpu / steady, rank-max):
#   vllm_ar   vLLM's tensor_model_parallel_all_reduce on an (M, 6144) bf16 tensor (CustomAllreduce below
#             8 MiB on a fully NVLinked node, PyNccl above; exactly what vLLM serving calls)
#   rmsnorm   the fused Triton RMSNorm of validate_block_v3.py on (M, 6144)
#   add       residual add (M, 6144)
# Hidden size 6144 belongs to no model in the G4 test set. M = 8 ... 16384.
# Output: raw_block_cal.csv, summary_block_cal.csv (medians), meta_block_cal.json
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
    def __init__(self, M, L):
        self.M, self.layer = M, _W(L)
        self.x = torch.randn((M, H), device="cuda").to(DT)
        self.y = torch.randn((M, H), device="cuda").to(DT)
        self.ar = torch.randn((M, H), device="cuda").to(DT)
        self.lnw = torch.ones(H, device="cuda", dtype=DT)
        self.fns = {"vllm_ar": lambda w: VLLM_AR(self.ar),
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
    for M in MS:
        case = Case(M, L)
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
            print(f"[block-cal tp{W} M={M} {mode}] " + "  ".join(f"{s[3]}={s[4]}" for s in summ[-len(items):]),
                  file=sys.stderr, flush=True)
        del case
    if RANK == 0:
        with open(os.path.join(ARGS.out_dir, "raw_block_cal.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["M", "H", "mode", "round", "item", "rank_max_ms", "kept"])
            w.writerows(rows)
        with open(os.path.join(ARGS.out_dir, "summary_block_cal.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["M", "H", "mode", "item", "median_ms", "n_kept"])
            w.writerows(summ)
        json.dump({"script": "ws/fusion-dispatch/scripts/calibrate_block_v1.py", "args": vars(ARGS), "world": W,
                   "date": time.strftime("%Y-%m-%d %H:%M:%S"), "wall_s": time.time() - t0,
                   "custom_ar": {"present": CA is not None, "disabled": getattr(CA, "disabled", None),
                                 "max_size": getattr(CA, "max_size", None)}},
                  open(os.path.join(ARGS.out_dir, "meta_block_cal.json"), "w"), indent=1)
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
