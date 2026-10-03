################################################################################
# fusion-dispatch G4 (D-009): measure the STANDALONE GEMMs of one model's tensor-parallel layers at
# the deployment M buckets. These replace the GEMM model in the decision (cuBLAS / Flux config cliffs,
# G3); communication and overlap stay modelled. Every timed op is local to a GPU (no communication
# inside the timed region); the TP group is only needed because AGKernel is a collective object.
#   AG layers (qkv, gate_up; weight (n, K)):  c_cublas  torch.mm (M x n x K)
#                                             c_fluxgemm AGKernel.gemm_only (the GEMM the fused op runs)
#   RS layers (o, down; weight (N, k)):       c_cublas  torch.mm (M x N x k)
#                                             c_fluxgemm_only flux.GemmOnly (GemmRS is modelled from it)
# Protocol: dispatch_map_v2.run_mode (interleaved, L2 flush, gpu / steady, rank-max, clock filter).
# All Flux ops are built once and never deleted (CLAUDE.md trap: re-creating Flux ops can deadlock).
# Usage: pixi run ... launch_tp.sh <TP> probe_gemm_v1.py --model qwen2.5-32b --Ms 24,96 --out_dir <dir>
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
from build_table_v2 import layers  # noqa: E402
import flux  # noqa: E402
from flux.testing import initialize_distributed  # noqa: E402

DT = torch.bfloat16


class _W:
    def __init__(self, weights):
        self.weights = weights


class Case:
    def __init__(self, side, op, weights, M, d0, d1):
        self.M, self.layer = M, _W(weights)
        if side == "ag":
            n, K = d0, d1
            self.x = torch.randn((M, K), dtype=torch.float32, device="cuda").to(DT)
            self.out = [torch.empty((M, n), dtype=DT, device="cuda") for _ in range(2)]
            self.fns = {"c_cublas": lambda w: torch.mm(self.x, w.t(), out=self.out[0]),
                        "c_fluxgemm": lambda w: op.gemm_only(self.x, w, output=self.out[1], transpose_weight=False)}
        else:
            N, k = d0, d1
            self.x = torch.randn((M, k), dtype=torch.float32, device="cuda").to(DT)
            self.out = [torch.empty((M, N), dtype=DT, device="cuda") for _ in range(2)]
            self.fns = {"c_cublas": lambda w: torch.mm(self.x, w.t(), out=self.out[0]),
                        "c_fluxgemm_only": lambda w: op.forward(self.x, w, output_buf=self.out[1])}

    def check(self, w):
        ref = torch.mm(self.x.float(), w.float().t())
        res = {}
        for j, it in enumerate(self.fns):
            self.out[j].zero_()
            self.fns[it](w)
            torch.cuda.synchronize()
            ok = torch.allclose(self.out[j].float(), ref, atol=0.05, rtol=0.05)
            f = torch.tensor([1 if ok else 0], dtype=torch.int32, device="cuda")
            dist.all_reduce(f, op=dist.ReduceOp.MIN, group=TP)
            res[it] = bool(f.item())
        return res


def main():
    Ms = [int(x) for x in ARGS.Ms.split(",")]
    modes = ARGS.modes.split(",")
    L = ARGS.steady_L if "steady" in modes else 1
    rng = random.Random(ARGS.seed)
    if RANK == 0:
        os.makedirs(ARGS.out_dir, exist_ok=True)
    dist.barrier(TP)
    tag = f"{ARGS.model}_tp{W}"
    lay = layers(ARGS.model, W)
    t0 = time.time()
    ops, wts = {}, {}
    for side, shapes in lay.items():  # build everything first, delete nothing
        for name, (d0, d1) in shapes.items():
            if side == "ag":
                ops[name] = flux.AGKernel(TP, NNODES, max(Ms), d0, d1, DT, output_dtype=DT)
            else:
                ops[name] = flux.GemmOnly(DT, DT, DT, transpose_weight=False)
            wts[name] = [(torch.randn((d0, d1), dtype=torch.float32, device="cuda") * 0.01).to(DT) for _ in range(L)]
    raw = open(os.path.join(ARGS.out_dir, f"raw_{tag}.csv"), "w", newline="") if RANK == 0 else None
    rw = csv.writer(raw) if raw else None
    if rw:
        rw.writerow(["side", "layer", "dim0", "dim1", "M", "mode", "round", "item", "rank_max_ms", "min_sm_clock_mhz", "kept"])
    summ, checks, first = [], {}, True
    for side, shapes in lay.items():
        for name, (d0, d1) in shapes.items():
            for M in Ms:
                case = Case(side, ops[name], wts[name], M, d0, d1)
                chk = case.check(wts[name][0])
                checks[f"{name}@{M}"] = chk
                items = [it for it in case.fns if chk[it]]
                for mode in modes:
                    rounds = ARGS.rounds if mode == "gpu" else ARGS.rounds_steady
                    T, C, Kc, orders, kept, modal = dm.run_mode(case, items, mode, rounds,
                                                                ARGS.warmup_first if first else ARGS.warmup, rng)
                    first = False
                    if RANK != 0:
                        continue
                    rmax, kmin = T.max(0).values, Kc.clamp(min=0).min(0).values
                    ks = [r for r in range(T.shape[1]) if kept[r]]
                    for r in range(T.shape[1]):
                        for j, it in enumerate(items):
                            rw.writerow([side, name, d0, d1, M, mode, r, it, f"{rmax[r, j]:.5f}", int(kmin[r]), int(kept[r])])
                    for j, it in enumerate(items):
                        v = [rmax[r, j].item() for r in ks] or [rmax[r, j].item() for r in range(T.shape[1])]
                        summ.append([side, name, d0, d1, M, mode, it, f"{statistics.median(v):.5f}", len(ks)])
                    print(f"[{tag} {name} M={M} {mode}] kept {len(ks)}/{T.shape[1]}  " +
                          "  ".join(f"{s_[6]}={s_[7]}" for s_ in summ[-len(items):]), file=sys.stderr, flush=True)
                del case
    if RANK == 0:
        raw.close()
        with open(os.path.join(ARGS.out_dir, f"gemm_{tag}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["side", "layer", "dim0", "dim1", "M", "mode", "item", "median_ms", "n_kept"])
            w.writerows(summ)
        json.dump({"script": "ws/fusion-dispatch/scripts/probe_gemm_v1.py", "args": vars(ARGS), "world": W,
                   "date": time.strftime("%Y-%m-%d %H:%M:%S"), "layers": lay, "checks": checks,
                   "wall_s_after_init": time.time() - t0, "cublas_settings": CUBLAS_SETTINGS},
                  open(os.path.join(ARGS.out_dir, f"meta_{tag}.json"), "w"), indent=1)
        print(f"wrote gemm_{tag}.csv ({time.time() - t0:.0f} s after init)", file=sys.stderr, flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--Ms", required=True)
    p.add_argument("--modes", default="gpu,steady")
    p.add_argument("--rounds", type=int, default=50)
    p.add_argument("--rounds_steady", type=int, default=20)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--warmup_first", type=int, default=100)
    p.add_argument("--steady_L", type=int, default=16)
    p.add_argument("--pad_cycles", type=int, default=400_000)
    p.add_argument("--flush_mb", type=int, default=128)
    p.add_argument("--clock_period", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=20261003)
    p.add_argument("--out_dir", required=True)
    return p.parse_args()


if __name__ == "__main__":
    TP = initialize_distributed()
    os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)  # undo init_seed() (CLAUDE.md trap)
    torch.use_deterministic_algorithms(False)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    CUBLAS_SETTINGS = {"CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                       "deterministic": torch.are_deterministic_algorithms_enabled()}
    RANK, W = TP.rank(), TP.size()
    NNODES = flux.testing.NNODES()
    ARGS = parse_args()
    dm.TP_GROUP, dm.RANK, dm.W = TP, RANK, W
    dm.LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    dm.ARGS = ARGS
    dm.FLUSH = torch.empty(ARGS.flush_mb * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
    dm.ALIGN = torch.zeros(1, dtype=torch.float32, device="cuda")
    main()
    dist.destroy_process_group()
