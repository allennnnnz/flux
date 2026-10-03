################################################################################
# fusion-dispatch F0.3 (E5): GEMM + ReduceScatter side of the dispatch map.
# Same protocol as dispatch_map_v2.py (whose timing loop run_mode() is reused):
# interleaved items in a shared random order, L2 flush, gpu/steady modes,
# per-round rank-max, SM-clock filter, production cuBLAS settings.
#
# Row-parallel layer: each rank holds x (M, K/8) and w (N, K/8); output is the
# reduce-scattered (M/8, N) (sequence-parallel layout, same as the AG side).
# Items (names chosen so analyze_v1.py can read the CSV):
#   A_fused        flux.GemmRS.forward                         -> (M/8, N)
#   B_nccl_cublas  torch.mm + dist.reduce_scatter_tensor       -> (M/8, N)
#   E_allreduce    torch.mm + dist.all_reduce  (vLLM default TP layout, output
#                  (M, N) replicated: a REFERENCE, not a drop-in replacement)
#   c_cublas, c_nccl_rs, c_nccl_ar, c_fluxgemm (flux.GemmOnly) components
# GEMM_RS comm is not separable inside Flux on sm80 (E_CORRECTION E.4), so only
# end-to-end A vs B is a decision; c_fluxgemm is a kernel-quality diagnostic.
# Usage: ./launch.sh dispatch_map_rs_v1.py --layer L-O --Ms 64,4096 --modes gpu,steady --out_dir <dir>
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
import dispatch_map_v2 as dm  # noqa: E402  (reuse run_mode + helpers)
import flux  # noqa: E402
from flux.testing import initialize_distributed  # noqa: E402

DTYPE = torch.bfloat16
# (N, K_full) of the row-parallel layer; each rank holds (N, K_full // world)
LAYERS = {
    "G-FC2": (12288, 49152),   # GPT-3 175B FC2
    "G-O": (12288, 12288),     # GPT-3 175B attention out
    "L-down": (8192, 28672),   # Llama-3-70B down_proj
    "L-O": (8192, 8192),       # Llama-3-70B o_proj
    # G3 (2026-10-02): models the predictor has never seen (plan appendix A.4)
    "Q-down": (8192, 29568),   # Qwen2.5-72B down_proj
    "L8-O": (4096, 4096),      # Llama-3-8B o_proj
    "L8-down": (4096, 14336),  # Llama-3-8B down_proj
    # G4 (2026-10-03): fresh test model, never used before (D-009)
    "Q32-O": (5120, 5120),     # Qwen2.5-32B o_proj
    "Q32-down": (5120, 27648), # Qwen2.5-32B down_proj
}
ALL_ITEMS = ["A_fused", "B_nccl_cublas", "E_allreduce", "c_cublas", "c_nccl_rs", "c_nccl_ar", "c_fluxgemm"]
GEMM_ITEMS = {"A_fused", "B_nccl_cublas", "E_allreduce", "c_cublas", "c_fluxgemm"}


class LayerRS:
    def __init__(self, name, max_m, n_weights):
        N, K = LAYERS[name]
        self.name, self.N, self.K, self.k = name, N, K, K // W
        self.op = flux.GemmRS(TP, 1, max_m, N, DTYPE, DTYPE, transpose_weight=False)
        self.gemm_only = flux.GemmOnly(DTYPE, DTYPE, DTYPE, transpose_weight=False)
        self.weights = [(torch.randn((N, self.k), dtype=torch.float32, device="cuda") * 0.01).to(DTYPE)
                        for _ in range(n_weights)]


class CaseRS:
    def __init__(self, layer, M):
        assert M % W == 0
        self.layer, self.M = layer, M
        N, k = layer.N, layer.k
        self.x = torch.randn((M, k), dtype=torch.float32, device="cuda").to(DTYPE)
        self.part = torch.empty((M, N), dtype=DTYPE, device="cuda")
        self.part2 = torch.empty((M, N), dtype=DTYPE, device="cuda")
        self.out_rs = torch.empty((M // W, N), dtype=DTYPE, device="cuda")
        self.out_x = torch.empty((M, N), dtype=DTYPE, device="cuda")
        self.out_rs2 = torch.empty((M // W, N), dtype=DTYPE, device="cuda")
        self.res = {}
        x, op, go = self.x, layer.op, layer.gemm_only

        def a(w):
            self.res["A_fused"] = op.forward(x, w)

        def b(w):
            torch.mm(x, w.t(), out=self.part)
            dist.reduce_scatter_tensor(self.out_rs, self.part, group=TP)

        def e(w):
            torch.mm(x, w.t(), out=self.part2)
            dist.all_reduce(self.part2, group=TP)

        self.fns = {
            "A_fused": a,
            "B_nccl_cublas": b,
            "E_allreduce": e,
            "c_cublas": lambda w: torch.mm(x, w.t(), out=self.out_x),
            "c_nccl_rs": lambda w: dist.reduce_scatter_tensor(self.out_rs2, self.part, group=TP),
            "c_nccl_ar": lambda w: dist.all_reduce(self.part, group=TP),
            "c_fluxgemm": lambda w: go.forward(x, w, output_buf=self.out_x),
        }

    def check(self, items):
        w = self.layer.weights[0]
        full = torch.mm(self.x.float(), w.float().t())
        dist.all_reduce(full, group=TP)  # fp32 reference sum over ranks
        m = self.M // W
        ref_rs = full[RANK * m:(RANK + 1) * m]
        ref_part = torch.mm(self.x.float(), w.float().t())
        res = {}
        for it in items:
            raised = ""
            try:
                self.fns[it](w)
                torch.cuda.synchronize()
            except RuntimeError as exc:
                raised = str(exc).strip().splitlines()[-1][:200]
            ok, err = False, float("nan")
            if not raised:
                got, ref = {
                    "A_fused": (lambda: self.res["A_fused"], ref_rs),
                    "B_nccl_cublas": (lambda: self.out_rs, ref_rs),
                    "E_allreduce": (lambda: self.part2, full),
                    "c_cublas": (lambda: self.out_x, ref_part),
                    "c_fluxgemm": (lambda: self.out_x, ref_part),
                }.get(it, (None, None))
                if got is None:  # pure comm items: shape/finite check only (inputs change per call)
                    ok, err = True, 0.0
                else:
                    g = got().float()
                    ok = g.shape == ref.shape and torch.allclose(g, ref, atol=ARGS.atol, rtol=ARGS.rtol)
                    err = (g - ref).abs().max().item() if g.shape == ref.shape else float("nan")
            flag = torch.tensor([1 if ok else 0], dtype=torch.int32, device="cuda")
            dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=TP)
            e_ = torch.tensor([err], dtype=torch.float64, device="cuda")
            dist.all_reduce(e_, op=dist.ReduceOp.MAX, group=TP)
            res[it] = {"ok_all_ranks": bool(flag.item()), "max_abs_err": float(e_.item()), "error": raised}
        return res


def main():
    Ms = [int(x) for x in ARGS.Ms.split(",")]
    modes = ARGS.modes.split(",")
    items = ARGS.items.split(",") if ARGS.items else ALL_ITEMS[:]
    max_m = ARGS.max_m or max(Ms)
    layer = LayerRS(ARGS.layer, max_m, ARGS.steady_L if "steady" in modes else 1)
    rng = random.Random(ARGS.seed)
    if RANK == 0:
        os.makedirs(ARGS.out_dir, exist_ok=True)
    raw_path = os.path.join(ARGS.out_dir, f"raw_{ARGS.layer}.csv")
    meta_path = os.path.join(ARGS.out_dir, f"meta_{ARGS.layer}.json")
    meta = {"script": "ws/fusion-dispatch/scripts/dispatch_map_rs_v1.py", "args": vars(ARGS),
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "layer": {"name": layer.name, "N": layer.N, "K_full": layer.K, "k_per_rank": layer.k},
            "world": W, "max_m": max_m, "cublas_settings": CUBLAS_SETTINGS,
            "nccl": ".".join(map(str, torch.cuda.nccl.version())), "cases": {}}
    new = not os.path.exists(raw_path)
    fh = open(raw_path, "a", newline="") if RANK == 0 else None
    wr = csv.writer(fh) if fh else None
    if wr and new:
        wr.writerow(["layer", "N", "K", "M", "mode", "round", "item", "order_pos", "rank_max_ms",
                     "rank_min_ms", "rank0_ms", "cpu_launch_rank_max_ms", "min_sm_clock_mhz", "kept"])
    for M in Ms:
        case = CaseRS(layer, M)
        chk = case.check(items)
        good = [it for it in items if chk[it]["ok_all_ranks"]]
        meta["cases"][str(M)] = {"check": chk, "timed_items": good, "modes": {}}
        if RANK == 0:
            bad = [it for it in items if it not in good]
            print(f"[{layer.name} M={M}] correctness: {'all ok' if not bad else 'FAIL ' + str(bad)} "
                  f"{ {it: chk[it]['error'] for it in bad} if bad else ''}", file=sys.stderr, flush=True)
        for mode in modes:
            if not good:
                continue
            T, C, Kc, orders, kept, modal = dm.run_mode(case, good, mode, ARGS.rounds, ARGS.warmup, rng)
            if RANK != 0:
                continue
            rounds = T.shape[1]
            meta["cases"][str(M)]["modes"][mode] = {"rounds": rounds, "kept": sum(kept),
                                                     "modal_sm_clock_per_rank": modal}
            rmax, rmin, cmax = T.max(0).values, T.min(0).values, C.max(0).values
            kmin = Kc.clamp(min=0).min(0).values
            for r in range(rounds):
                for j, it in enumerate(good):
                    wr.writerow([layer.name, layer.N, layer.K, M, mode, r, it, orders[r].index(it),
                                 f"{rmax[r, j]:.5f}", f"{rmin[r, j]:.5f}", f"{T[0, r, j]:.5f}",
                                 f"{cmax[r, j]:.5f}", int(kmin[r]), int(kept[r])])
            fh.flush()
            ks = [r for r in range(rounds) if kept[r]]
            idx = {it: j for j, it in enumerate(good)}
            line = "  ".join(f"{it}={statistics.median(rmax[r, idx[it]].item() for r in ks):.4f}" for it in good)
            print(f"[{layer.name} M={M} {mode}] kept {len(ks)}/{rounds}  {line}", file=sys.stderr, flush=True)
        del case
        torch.cuda.empty_cache()
    if RANK == 0:
        fh.close()
        old = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
        old.setdefault("runs", []).append(meta)
        json.dump(old, open(meta_path, "w"), indent=1)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--layer", required=True, choices=list(LAYERS))
    p.add_argument("--Ms", required=True)
    p.add_argument("--max_m", type=int, default=0)
    p.add_argument("--modes", default="gpu")
    p.add_argument("--items", default="")
    p.add_argument("--rounds", type=int, default=200)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--steady_L", type=int, default=16)
    p.add_argument("--pad_cycles", type=int, default=400_000)
    p.add_argument("--flush_mb", type=int, default=128)
    p.add_argument("--clock_period", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=20260930)
    p.add_argument("--atol", type=float, default=0.1)
    p.add_argument("--rtol", type=float, default=0.05)
    p.add_argument("--out_dir", required=True)
    return p.parse_args()


if __name__ == "__main__":
    TP = initialize_distributed()
    os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)  # undo init_seed(): production cuBLAS
    torch.use_deterministic_algorithms(False)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    CUBLAS_SETTINGS = {"CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                       "deterministic": torch.are_deterministic_algorithms_enabled(),
                       "allow_bf16_reduced_precision_reduction":
                           torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction}
    RANK, W = TP.rank(), TP.size()
    ARGS = parse_args()
    # globals used by dm.run_mode
    dm.TP_GROUP, dm.RANK, dm.W = TP, RANK, W
    dm.LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    dm.ARGS = ARGS
    dm.FLUSH = torch.empty(ARGS.flush_mb * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
    dm.ALIGN = torch.zeros(1, dtype=torch.float32, device="cuda")
    main()
    dist.destroy_process_group()
