################################################################################
# fusion-dispatch G2: hardware calibration microbenchmarks for the predictor
# (common/cost_model/predictor). Plan: reports/20261002_plan_general_dispatcher.md section 2.
#
# Key rule: the predictor is fitted ONLY from these runs. Every shape here is deliberately NOT one
# of the 8 evaluation layers (AG: n x K in {6144x12288, 4608x12288, 1280x8192, 7168x8192};
# RS: N x k in {12288x6144, 12288x1536, 8192x3584, 8192x1024}), and none hits a Flux registry entry.
#
# Groups (each case = one shape and one M; its items are timed interleaved in one round):
#   comm   K=6144 (hidden size used by no evaluation layer), M = 8 ... 32768 (98 KB - 403 MB):
#          c_nccl_ag   all_gather_into_tensor (M/8, K) -> (M, K)
#          c_nccl_rs   reduce_scatter_tensor  (M, K) -> (M/8, K)
#          c_nccl_ar   all_reduce             (M, K)
#          c_flux_ag   flux.AllGatherOp.run, All2All + use_read (same option as AGKernel)
#          ce_copy1    one peer pull of one shard (M/8 x K) from rank+1, copy engine
#          ce_copy7    W-1 peer pulls on one stream (proto_ce_1s of ag_latency_v1.py)
#   ag     AG+GEMM shapes (n, K) = (2560, 5120), (5120, 10240), (1536, 16384); M = 16 ... 8192:
#          c_cublas, c_fluxgemm (AGKernel.gemm_only), c_flux_ag, A_fused (AGKernel.forward)
#   rs     GEMM+RS shapes (N, k) = (6144, 2560), (10240, 1280); M = 16 ... 8192:
#          c_cublas (M x N x k), A_gemmrs (flux.GemmRS.forward)
# Protocol: dispatch_map_v2.run_mode (interleaved, L2 flush, gpu / steady, rank-max, clock filter).
# ONE SHAPE GROUP PER PROCESS (--groups comm | ag --ag_shapes nxK | rs --rs_shapes Nxk): creating a
# second set of Flux ops after deleting the first in the same process deadlocked intermittently in
# the 2026-10-02 smoke test (all ranks in torch.cuda.synchronize, GPUs spinning; results/g2_smoke/
# debug_hang2/log.txt); the same shape alone ran fine. run_calibration_v1.sh launches one process each.
# Output: raw_<tag>.csv (one row per round x item, analyze-style columns) + meta_<tag>.json
# Usage: bash ws/fusion-dispatch/scripts/run_calibration_v1.sh <out_dir>  (guard + one process per group)
################################################################################
import argparse
import csv
import faulthandler
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
import flux  # noqa: E402
from flux.testing import initialize_distributed  # noqa: E402

DT = torch.bfloat16
COMM_K = 6144
COMM_MS = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
AG_SHAPES = [(2560, 5120), (5120, 10240), (1536, 16384)]   # (n per rank, K)
RS_SHAPES = [(6144, 2560), (10240, 1280)]                  # (N, k per rank)
GEMM_MS = [16, 128, 512, 2048, 8192]
FIRST = [True]


class _W:  # what run_mode needs from a "layer": a list of weights (distinct per call in steady)
    def __init__(self, weights):
        self.weights = weights


class CommCase:
    def __init__(self, M, ag_op, ipc, L):
        K = COMM_K
        self.M, self.layer = M, _W([None] * L)
        m = M // W
        self.x = torch.randn((m, K), dtype=torch.float32, device="cuda").to(DT)
        self.full = torch.empty((M, K), dtype=DT, device="cuda")
        self.big = torch.randn((M, K), dtype=torch.float32, device="cuda").to(DT)
        self.rs_out = torch.empty((m, K), dtype=DT, device="cuda")
        self.ar_buf = self.big.clone()
        mine, chunk = ipc[RANK], m * K
        mine[RANK * chunk:(RANK + 1) * chunk].copy_(self.x.view(-1))
        dist.barrier(TP)
        src = (RANK + 1) % W
        opt = dm.ag_option()

        def ce7():
            for i in range(1, W):
                p = (RANK + i) % W
                mine[p * chunk:(p + 1) * chunk].copy_(ipc[p][p * chunk:(p + 1) * chunk], non_blocking=True)
        self.fns = {
            "c_nccl_ag": lambda w: dist.all_gather_into_tensor(self.full, self.x, group=TP),
            "c_nccl_rs": lambda w: dist.reduce_scatter_tensor(self.rs_out, self.big, group=TP),
            "c_nccl_ar": lambda w: dist.all_reduce(self.ar_buf, group=TP),
            "c_flux_ag": lambda w: ag_op.run(self.x, None, opt, torch.cuda.current_stream().cuda_stream),
            "ce_copy1": lambda w: mine[src * chunk:(src + 1) * chunk].copy_(
                ipc[src][src * chunk:(src + 1) * chunk], non_blocking=True),
            "ce_copy7": lambda w: ce7(),
        }
        self.ag_buf = ag_op.local_input_buffer()[:M]

    def check(self):
        self.ag_buf.zero_()
        self.fns["c_flux_ag"](None)
        dist.all_gather_into_tensor(self.full, self.x, group=TP)
        torch.cuda.synchronize()
        ok = torch.equal(self.ag_buf, self.full)
        return {"c_flux_ag_equals_nccl": _all_ok(ok)}


class AGCase:
    def __init__(self, op, ag_op, weights, M, n, K):
        self.M, self.layer = M, _W(weights)
        self.x = torch.randn((M // W, K), dtype=torch.float32, device="cuda").to(DT)
        self.full = torch.empty((M, K), dtype=DT, device="cuda")
        dist.all_gather_into_tensor(self.full, self.x, group=TP)
        self.out = {k: torch.empty((M, n), dtype=DT, device="cuda") for k in "AXY"}
        opt = dm.ag_option()
        self.ag_buf = ag_op.local_input_buffer()[:M]
        self.fns = {
            "c_cublas": lambda w: torch.mm(self.full, w.t(), out=self.out["X"]),
            "c_fluxgemm": lambda w: op.gemm_only(self.full, w, output=self.out["Y"], transpose_weight=False),
            "c_flux_ag": lambda w: ag_op.run(self.x, None, opt, torch.cuda.current_stream().cuda_stream),
            "A_fused": lambda w: op.forward(self.x, w, output=self.out["A"], transpose_weight=False,
                                            all_gather_option=opt),
        }

    def check(self, w):
        ref = torch.mm(self.full.float(), w.float().t())
        res = {}
        for it, key in (("A_fused", "A"), ("c_fluxgemm", "Y"), ("c_cublas", "X")):
            self.out[key].zero_()
            self.fns[it](w)
            torch.cuda.synchronize()
            res[it] = _all_ok(torch.allclose(self.out[key].float(), ref, atol=ARGS.atol, rtol=ARGS.rtol))
        self.ag_buf.zero_()
        self.fns["c_flux_ag"](w)
        torch.cuda.synchronize()
        res["c_flux_ag"] = _all_ok(torch.equal(self.ag_buf, self.full))
        return res


class RSCase:
    def __init__(self, op, weights, M, N, k):
        self.M, self.layer = M, _W(weights)
        self.x = torch.randn((M, k), dtype=torch.float32, device="cuda").to(DT)
        self.part = torch.empty((M, N), dtype=DT, device="cuda")
        self.res = {}

        def a(w):
            self.res["A"] = op.forward(self.x, w)
        self.fns = {"c_cublas": lambda w: torch.mm(self.x, w.t(), out=self.part), "A_gemmrs": a}

    def check(self, w):
        full = torch.mm(self.x.float(), w.float().t())
        dist.all_reduce(full, group=TP)
        m = self.M // W
        ref = full[RANK * m:(RANK + 1) * m]
        self.fns["A_gemmrs"](w)
        torch.cuda.synchronize()
        g = self.res["A"].float()
        return {"A_gemmrs": _all_ok(g.shape == ref.shape and torch.allclose(g, ref, atol=ARGS.atol * 4, rtol=ARGS.rtol))}


def _all_ok(ok):
    f = torch.tensor([1 if ok else 0], dtype=torch.int32, device="cuda")
    dist.all_reduce(f, op=dist.ReduceOp.MIN, group=TP)
    return bool(f.item())


def time_case(case, group, shape, M, items, modes, rng, wr, meta_case):
    for mode in modes:
        rounds = ARGS.rounds if mode == "gpu" else (ARGS.rounds_steady_comm if group == "comm" else ARGS.rounds_steady_gemm)
        warm = ARGS.warmup_first if FIRST else ARGS.warmup  # first case of the run: GPU clock ramp-up (A.3)
        FIRST.clear()
        T, C, Kc, orders, kept, modal = dm.run_mode(case, items, mode, rounds, warm, rng)
        if RANK != 0:
            continue
        rmax, rmin, cmax = T.max(0).values, T.min(0).values, C.max(0).values
        kmin = Kc.clamp(min=0).min(0).values
        for r in range(T.shape[1]):
            for j, it in enumerate(items):
                wr.writerow([group, shape[0], shape[1], M, mode, r, it, orders[r].index(it), f"{rmax[r, j]:.5f}",
                             f"{rmin[r, j]:.5f}", f"{T[0, r, j]:.5f}", f"{cmax[r, j]:.5f}", int(kmin[r]), int(kept[r])])
        ks = [r for r in range(T.shape[1]) if kept[r]]
        meta_case.setdefault("modes", {})[mode] = {"rounds": T.shape[1], "kept": len(ks), "modal_sm_clock_per_rank": modal}
        idx = {it: j for j, it in enumerate(items)}
        line = "  ".join(f"{it}={statistics.median(rmax[r, idx[it]].item() for r in ks):.4f}" for it in items) if ks else "no kept rounds"
        print(f"[{group} {shape} M={M} {mode}] kept {len(ks)}/{T.shape[1]}  {line}", file=sys.stderr, flush=True)


def main():
    modes = ARGS.modes.split(",")
    L = ARGS.steady_L if "steady" in modes else 1
    rng = random.Random(ARGS.seed)
    os.makedirs(ARGS.out_dir, exist_ok=True) if RANK == 0 else None
    dist.barrier(TP)
    raw_path = os.path.join(ARGS.out_dir, f"raw_{ARGS.tag}.csv")
    fh = open(raw_path, "w", newline="") if RANK == 0 else None
    wr = csv.writer(fh) if fh else None
    if wr:
        wr.writerow(["group", "dim0", "dim1", "M", "mode", "round", "item", "order_pos", "rank_max_ms", "rank_min_ms",
                     "rank0_ms", "cpu_launch_rank_max_ms", "min_sm_clock_mhz", "kept"])
    meta = {"script": "ws/fusion-dispatch/scripts/calibrate_hw_v1.py", "args": vars(ARGS),
            "date": time.strftime("%Y-%m-%d %H:%M:%S"), "world": W, "torch": torch.__version__,
            "nccl": ".".join(map(str, torch.cuda.nccl.version())), "cublas_settings": CUBLAS_SETTINGS,
            "comm_K": COMM_K, "ag_shapes": AG_SHAPES, "rs_shapes": RS_SHAPES, "cases": []}
    t_start = time.time()
    groups = ARGS.groups.split(",")

    if "comm" in groups:
        max_m = max(COMM_MS)
        ag_op = flux.AllGatherOp(TP, NNODES, max_m, COMM_K, DT)
        ipc = flux.create_tensor_list([max_m * COMM_K], DT, TP)
        items = ["c_nccl_ag", "c_nccl_rs", "c_nccl_ar", "c_flux_ag", "ce_copy1", "ce_copy7"]
        for M in COMM_MS:
            case = CommCase(M, ag_op, ipc, L)
            mc = {"group": "comm", "shape": [M, COMM_K], "M": M, "check": case.check()}
            time_case(case, "comm", (M // W, COMM_K), M, items, modes, rng, wr, mc)
            meta["cases"].append(mc)
            del case
            torch.cuda.empty_cache()
        if fh:
            fh.flush()

    gemm_ms = [int(x) for x in ARGS.gemm_ms.split(",")] if ARGS.gemm_ms else GEMM_MS
    ag_shapes = [tuple(int(v) for v in s_.split("x")) for s_ in ARGS.ag_shapes.split(",")] if ARGS.ag_shapes else AG_SHAPES
    rs_shapes = [tuple(int(v) for v in s_.split("x")) for s_ in ARGS.rs_shapes.split(",")] if ARGS.rs_shapes else RS_SHAPES
    assert len(groups) == 1 and (groups[0] == "comm" or len(ag_shapes if groups[0] == "ag" else rs_shapes) == 1), \
        "one shape group per process (see header)"
    if "ag" in groups:
        for n, K in ag_shapes:
            max_m = max(GEMM_MS)
            op = flux.AGKernel(TP, NNODES, max_m, n, K, DT, output_dtype=DT)
            ag_op = flux.AllGatherOp(TP, NNODES, max_m, K, DT)
            weights = [(torch.randn((n, K), dtype=torch.float32, device="cuda") * 0.01).to(DT) for _ in range(L)]
            for M in gemm_ms:
                if RANK == 0:
                    print(f"[ag {n}x{K} M={M}] setup + check", file=sys.stderr, flush=True)
                case = AGCase(op, ag_op, weights, M, n, K)
                chk = case.check(weights[0])
                good = [it for it in ["c_cublas", "c_fluxgemm", "c_flux_ag", "A_fused"] if chk[it]]
                mc = {"group": "ag", "shape": [n, K], "M": M, "check": chk, "timed": good}
                time_case(case, "ag", (n, K), M, good, modes, rng, wr, mc)
                meta["cases"].append(mc)
                del case
                torch.cuda.empty_cache()
            if fh:
                fh.flush()

    if "rs" in groups:
        for N, k in rs_shapes:
            max_m = max(GEMM_MS)
            op = flux.GemmRS(TP, 1, max_m, N, DT, DT, transpose_weight=False)
            weights = [(torch.randn((N, k), dtype=torch.float32, device="cuda") * 0.01).to(DT) for _ in range(L)]
            for M in gemm_ms:
                case = RSCase(op, weights, M, N, k)
                chk = case.check(weights[0])
                good = ["c_cublas"] + (["A_gemmrs"] if chk["A_gemmrs"] else [])
                mc = {"group": "rs", "shape": [N, k], "M": M, "check": chk, "timed": good}
                time_case(case, "rs", (N, k), M, good, modes, rng, wr, mc)
                meta["cases"].append(mc)
                del case
                torch.cuda.empty_cache()

    if RANK == 0:
        fh.close()
        meta["wall_s"] = time.time() - t_start
        json.dump(meta, open(os.path.join(ARGS.out_dir, f"meta_{ARGS.tag}.json"), "w"), indent=1)
        print(f"wrote {raw_path}  ({meta['wall_s']:.0f} s of timing)", file=sys.stderr, flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--modes", default="gpu,steady")
    p.add_argument("--groups", required=True, help="comm | ag | rs (one per process)")
    p.add_argument("--tag", required=True, help="output file tag, e.g. comm, ag_2560x5120")
    p.add_argument("--rounds", type=int, default=200, help="gpu mode rounds")
    p.add_argument("--rounds_steady_comm", type=int, default=100)
    p.add_argument("--rounds_steady_gemm", type=int, default=50)
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--warmup_first", type=int, default=100)
    p.add_argument("--steady_L", type=int, default=16)
    p.add_argument("--pad_cycles", type=int, default=400_000)
    p.add_argument("--flush_mb", type=int, default=128)
    p.add_argument("--clock_period", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=20261002)
    p.add_argument("--atol", type=float, default=0.05)
    p.add_argument("--rtol", type=float, default=0.05)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--ag_shapes", default="", help="one n x K, e.g. 5120x10240")
    p.add_argument("--rs_shapes", default="", help="one N x k, e.g. 6144x2560")
    p.add_argument("--gemm_ms", default="", help="debug: comma list (default: GEMM_MS)")
    p.add_argument("--hang_dump_s", type=int, default=0, help="debug: dump Python stacks and exit after N s")
    return p.parse_args()


if __name__ == "__main__":
    TP = initialize_distributed()
    # undo init_seed(): production cuBLAS state, before any cuBLAS handle exists (A.3)
    os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    torch.use_deterministic_algorithms(False)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    CUBLAS_SETTINGS = {"CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                       "deterministic": torch.are_deterministic_algorithms_enabled(),
                       "allow_bf16_reduced_precision_reduction":
                           torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction}
    RANK, W = TP.rank(), TP.size()
    NNODES = flux.testing.NNODES()
    ARGS = parse_args()
    if ARGS.hang_dump_s:
        faulthandler.dump_traceback_later(ARGS.hang_dump_s, exit=True)
    dm.TP_GROUP, dm.RANK, dm.W = TP, RANK, W        # globals used by dm.run_mode / dm.ag_option
    dm.LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    dm.ARGS = ARGS
    dm.FLUSH = torch.empty(ARGS.flush_mb * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
    dm.ALIGN = torch.zeros(1, dtype=torch.float32, device="cuda")
    main()
    dist.destroy_process_group()
