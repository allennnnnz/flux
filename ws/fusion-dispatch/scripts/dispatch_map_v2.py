################################################################################
# fusion-dispatch harness v2: Flux fused vs non-fused AG+GEMM as a function of M.
# Design: ws/fusion-dispatch/reports/20260929_experiment_design.md (sections 1.2, 5)
#
# v2 vs v1: flux.testing.initialize_distributed() calls init_seed()
# (python/flux/testing/utils.py:49-62), which puts cuBLAS in a non-production
# state: CUBLAS_WORKSPACE_CONFIG=:16:8 (host launch 13 -> 71 us per torch.mm),
# deterministic algorithms (warn_only), and bf16/fp16 reduced-precision
# reduction disabled (up to +26% GPU time on some shapes). Measured in
# results/e0_anchor/cublas_settings_v1.txt. v2 restores PyTorch defaults right
# after initialize_distributed() and before the first cuBLAS call, and records
# the effective settings in meta. Flux and NCCL paths are unaffected by these.
# 2026-09-29 (after E0): check() catches RuntimeError per item and records it;
# an item that fails is excluded from timing for that M. Timing unchanged.
#
# Items, all timed inside the same round in a random order shared by all ranks:
#   A_fused            AGKernel.forward (All2All)
#   B_nccl_cublas      NCCL all_gather_into_tensor + torch.mm
#   C_fluxag_cublas    flux.AllGatherOp.run (All2All) + torch.mm
#   D_fluxag_fluxgemm  flux.AllGatherOp.run + AGKernel.gemm_only, serial
#   c_nccl_ag / c_flux_ag / c_cublas / c_fluxgemm      components alone
#
# Modes (design 5.1 / 5.2):
#   gpu    L2 flush -> GPU sleep pad -> 1-elem NCCL all_reduce (GPU-side align)
#          -> start ev -> item -> end ev. The CPU runs ahead, so this is GPU
#          execution time (what a CUDA-graph replay would see).
#   host   L2 flush -> sync -> dist.barrier -> start ev -> item -> end ev.
#          Includes launch latency. Phase 0 style; used for anchors.
#   steady L2 flush -> align -> start ev -> item x L with L distinct weights
#          -> end ev. per-call = elapsed / L. Eager layer-stack view; also
#          records host launch time per call.
#
# Every per-round time is reduced to the rank-max before anything else.
#
# Usage (repo root):
#   pixi run --manifest-path pixi.toml ./launch.sh \
#     ws/fusion-dispatch/scripts/dispatch_map_v2.py --layer G-FC1 \
#     --Ms 64,4096 --modes gpu,steady --rounds 200 --out_dir ws/fusion-dispatch/results/<tag>
################################################################################

import argparse
import csv
import json
import os
import random
import statistics
import sys
import time
from functools import partial

import torch
import torch.distributed as dist

import flux
from flux.testing import initialize_distributed

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "common", "measure"))
from clock_logger import ClockLogger  # noqa: E402

print = partial(print, file=sys.stderr, flush=True)

DTYPE = torch.bfloat16

# (N, K) of the full layer; each rank holds a (N // world, K) column shard.
LAYERS = {
    "G-FC1": (49152, 12288),    # GPT-3 175B FC1 (paper shape)
    "G-QKV": (36864, 12288),    # GPT-3 175B QKV
    "L-QKV": (10240, 8192),     # Llama-3-70B QKV, GQA 8 kv heads
    "L-GU": (57344, 8192),      # Llama-3-70B gate+up
    "P0-4096": (4096, 12288),   # Phase 0 anchor shape
    "P0-8192": (8192, 12288),   # Phase 0 anchor shape
    # G3 (2026-10-02): models the predictor has never seen (plan appendix A.4)
    "Q-GU": (59136, 8192),      # Qwen2.5-72B gate+up (ffn 29568); its QKV / O equal Llama-3-70B's
    "L8-QKV": (6144, 4096),     # Llama-3-8B QKV, 32 heads + 2 x 8 kv heads, head_dim 128
    "L8-GU": (28672, 4096),     # Llama-3-8B gate+up (ffn 14336)
    # G4 (2026-10-03): fresh test model, never used before (D-009)
    "Q32-QKV": (7168, 5120),    # Qwen2.5-32B QKV, 40 heads + 2 x 8 kv heads, head_dim 128
    "Q32-GU": (55296, 5120),    # Qwen2.5-32B gate+up (ffn 27648)
}

ALL_ITEMS = ["A_fused", "B_nccl_cublas", "C_fluxag_cublas", "D_fluxag_fluxgemm",
             "c_nccl_ag", "c_flux_ag", "c_cublas", "c_fluxgemm"]
GEMM_ITEMS = {"A_fused", "B_nccl_cublas", "C_fluxag_cublas", "D_fluxag_fluxgemm",
              "c_cublas", "c_fluxgemm"}


def ag_option():
    opt = flux.AllGatherOption()
    opt.mode = flux.AGRingMode.All2All
    opt.use_read = True
    opt.use_cuda_core_local = False
    opt.use_cuda_core_ag = False
    opt.fuse_sync = False
    opt.input_buffer_copied = False  # local shard copy is part of the cost
    return opt


class Layer:
    def __init__(self, name, max_m, n_weights):
        N, K = LAYERS[name]
        assert N % W == 0 and max_m % W == 0
        self.name, self.N, self.K, self.n = name, N, K, N // W
        self.op = flux.AGKernel(TP_GROUP, NNODES, max_m, self.n, K, DTYPE, output_dtype=DTYPE)
        self.ag = flux.AllGatherOp(TP_GROUP, NNODES, max_m, K, DTYPE)
        self.ag_buf_full = self.ag.local_input_buffer()
        self.weights = [
            (torch.randn((self.n, K), dtype=torch.float32, device="cuda") * 0.01).to(DTYPE)
            for _ in range(n_weights)
        ]
        self.opt = ag_option()


class Case:
    """Buffers and item closures for one (layer, M)."""

    def __init__(self, layer: Layer, M: int):
        assert M % W == 0, f"M={M} must be a multiple of world={W}"
        self.layer, self.M = layer, M
        K, n = layer.K, layer.n
        self.x = torch.randn((M // W, K), dtype=torch.float32, device="cuda").to(DTYPE)
        self.full_ref = torch.empty((M, K), dtype=DTYPE, device="cuda")
        dist.all_gather_into_tensor(self.full_ref, self.x, group=TP_GROUP)
        self.full_nccl = torch.empty_like(self.full_ref)
        self.out = {k: torch.empty((M, n), dtype=DTYPE, device="cuda") for k in "ABCDX"}
        self.ag_buf = layer.ag_buf_full[:M]
        op, ag, opt, x = layer.op, layer.ag, layer.opt, self.x
        out, full_ref, full_nccl, ag_buf = self.out, self.full_ref, self.full_nccl, self.ag_buf

        def flux_ag():
            ag.run(x, None, opt, torch.cuda.current_stream().cuda_stream)

        self.fns = {
            "A_fused": lambda w: op.forward(x, w, output=out["A"], transpose_weight=False,
                                            all_gather_option=opt),
            "B_nccl_cublas": lambda w: (
                dist.all_gather_into_tensor(full_nccl, x, group=TP_GROUP),
                torch.mm(full_nccl, w.t(), out=out["B"])),
            "C_fluxag_cublas": lambda w: (flux_ag(), torch.mm(ag_buf, w.t(), out=out["C"])),
            "D_fluxag_fluxgemm": lambda w: (
                flux_ag(), op.gemm_only(ag_buf, w, output=out["D"], transpose_weight=False)),
            "c_nccl_ag": lambda w: dist.all_gather_into_tensor(full_nccl, x, group=TP_GROUP),
            "c_flux_ag": lambda w: flux_ag(),
            "c_cublas": lambda w: torch.mm(full_ref, w.t(), out=out["X"]),
            "c_fluxgemm": lambda w: op.gemm_only(full_ref, w, output=out["X"],
                                                 transpose_weight=False),
        }

    def check(self, items):
        """Correctness of every item against an fp32 reference, all ranks."""
        w = self.layer.weights[0]
        ref = torch.mm(self.full_ref.float(), w.float().t())
        res = {}
        for it in items:
            for k in self.out:
                self.out[k].zero_()
            self.ag_buf.zero_()
            self.full_nccl.zero_()
            try:
                self.fns[it](w)
                torch.cuda.synchronize()
                raised = ""
            except RuntimeError as exc:  # e.g. shape unsupported by a Flux kernel
                raised = str(exc).strip().splitlines()[-1][:200]
            if raised:
                ok, err = False, float("nan")
            elif it in GEMM_ITEMS:
                got = {"A_fused": "A", "B_nccl_cublas": "B", "C_fluxag_cublas": "C",
                       "D_fluxag_fluxgemm": "D", "c_cublas": "X", "c_fluxgemm": "X"}[it]
                g = self.out[got].float()
                ok = torch.allclose(g, ref, atol=ARGS.atol, rtol=ARGS.rtol)
                err = (g - ref).abs().max().item()
            elif it == "c_flux_ag":
                ok = torch.equal(self.ag_buf, self.full_ref)
                err = 0.0 if ok else float("nan")
            else:  # c_nccl_ag
                ok = torch.equal(self.full_nccl, self.full_ref)
                err = 0.0 if ok else float("nan")
            if it in ("C_fluxag_cublas", "D_fluxag_fluxgemm") and not raised:
                ok = ok and torch.equal(self.ag_buf, self.full_ref)
            flag = torch.tensor([1 if ok else 0], dtype=torch.int32, device="cuda")
            dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=TP_GROUP)
            e = torch.tensor([err], dtype=torch.float64, device="cuda")
            dist.all_reduce(e, op=dist.ReduceOp.MAX, group=TP_GROUP)
            res[it] = {"ok_all_ranks": bool(flag.item()), "max_abs_err": float(e.item()),
                       "error": raised}
        del ref
        torch.cuda.empty_cache()
        return res


def run_mode(case: Case, items, mode, rounds, warmup, rng):
    layer = case.layer
    L = ARGS.steady_L if mode == "steady" else 1
    ws = layer.weights[:L] if mode == "steady" else layer.weights[:1]

    def one_round(ev_row, cpu_row, order):
        for it in order:
            fn = case.fns[it]
            FLUSH.zero_()
            if mode == "host":
                torch.cuda.synchronize()
                dist.barrier(TP_GROUP)
            else:
                if mode == "gpu":
                    torch.cuda._sleep(ARGS.pad_cycles)
                dist.all_reduce(ALIGN, group=TP_GROUP)
            s, e = ev_row[it]
            s.record()
            c0 = time.perf_counter()
            for w in ws:
                fn(w)
            c1 = time.perf_counter()
            e.record()
            cpu_row[it] = (c1 - c0) * 1e3 / L

    # warmup (same protocol, discarded)
    for _ in range(warmup):
        evw = {it: (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
               for it in items}
        order = items[:]
        rng.shuffle(order)
        one_round(evw, {}, order)
    torch.cuda.synchronize()
    dist.barrier(TP_GROUP)

    ev = [{it: (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
           for it in items} for _ in range(rounds)]
    cpu = [{} for _ in range(rounds)]
    orders, windows = [], []
    with ClockLogger(LOCAL_RANK, period_s=ARGS.clock_period) as clk:
        for r in range(rounds):
            order = items[:]
            rng.shuffle(order)
            orders.append(order)
            t0 = time.perf_counter()
            one_round(ev[r], cpu[r], order)
            torch.cuda.synchronize()
            windows.append((t0, time.perf_counter()))
        clocks = [clk.min_clock_between(*w_) for w_ in windows]

    n_it = len(items)
    t = torch.zeros((rounds, n_it), dtype=torch.float64, device="cuda")
    c = torch.zeros((rounds, n_it), dtype=torch.float64, device="cuda")
    for r in range(rounds):
        for j, it in enumerate(items):
            s, e = ev[r][it]
            t[r, j] = s.elapsed_time(e) / L
            c[r, j] = cpu[r][it]
    clk_t = torch.tensor([x if x else -1 for x in clocks], dtype=torch.float64, device="cuda")
    gt = [torch.zeros_like(t) for _ in range(W)]
    gc = [torch.zeros_like(c) for _ in range(W)]
    gk = [torch.zeros_like(clk_t) for _ in range(W)]
    dist.all_gather(gt, t, group=TP_GROUP)
    dist.all_gather(gc, c, group=TP_GROUP)
    dist.all_gather(gk, clk_t, group=TP_GROUP)
    T = torch.stack(gt).cpu()   # (W, rounds, items)
    C = torch.stack(gc).cpu()
    Kc = torch.stack(gk).cpu()  # (W, rounds)

    # clock filter: drop a round if any rank sampled below 95% of its modal clock
    kept = [True] * rounds
    modal = []
    for rk in range(W):
        vals = [int(v) for v in Kc[rk].tolist() if v > 0]
        m_ = statistics.mode(vals) if vals else None
        modal.append(m_)
        if m_ is None:
            continue
        for r in range(rounds):
            v = Kc[rk, r].item()
            if v > 0 and v < 0.95 * m_:
                kept[r] = False
    return T, C, Kc, orders, kept, modal


def q(v, p):
    s = sorted(v)
    return s[max(0, min(len(s) - 1, int(p * len(s))))]


def main():
    Ms = [int(x) for x in ARGS.Ms.split(",")]
    modes = ARGS.modes.split(",")
    items = ARGS.items.split(",") if ARGS.items else ALL_ITEMS[:]
    max_m = ARGS.max_m or max(Ms)
    n_weights = ARGS.steady_L if "steady" in modes else 1
    layer = Layer(ARGS.layer, max_m, n_weights)
    rng = random.Random(ARGS.seed)  # identical on all ranks: same item order everywhere

    os.makedirs(ARGS.out_dir, exist_ok=True) if RANK == 0 else None
    raw_path = os.path.join(ARGS.out_dir, f"raw_{ARGS.layer}.csv")
    meta_path = os.path.join(ARGS.out_dir, f"meta_{ARGS.layer}.json")
    meta = {
        "script": "ws/fusion-dispatch/scripts/dispatch_map_v2.py",
        "cublas_settings": CUBLAS_SETTINGS,
        "args": vars(ARGS), "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "layer": {"name": layer.name, "N": layer.N, "K": layer.K, "n_per_rank": layer.n},
        "world": W, "dtype": str(DTYPE), "max_m": max_m,
        "torch": torch.__version__, "nccl": ".".join(map(str, torch.cuda.nccl.version())),
        "env": {k: os.environ.get(k) for k in
                ["CUDA_DEVICE_MAX_CONNECTIONS", "NCCL_ALGO", "NCCL_PROTO", "NVSHMEM_DISABLE_CUDA_VMM",
                 "FLUX_TUNE_CONFIG_FILE"]},
        "cases": {},
    }
    new_file = not os.path.exists(raw_path)
    fh = open(raw_path, "a", newline="") if RANK == 0 else None
    wr = csv.writer(fh) if fh else None
    if wr and new_file:
        wr.writerow(["layer", "N", "K", "M", "mode", "round", "item", "order_pos",
                     "rank_max_ms", "rank_min_ms", "rank0_ms", "cpu_launch_rank_max_ms",
                     "min_sm_clock_mhz", "kept"])

    for M in Ms:
        case = Case(layer, M)
        chk = case.check(items)
        good = [it for it in items if chk[it]["ok_all_ranks"]]
        meta["cases"][str(M)] = {"check": chk, "timed_items": good, "modes": {}}
        if RANK == 0:
            bad = [it for it in items if not chk[it]["ok_all_ranks"]]
            print(f"[{layer.name} M={M}] correctness: {'all ok' if not bad else 'FAIL ' + str(bad)}")
        for mode in modes:
            if not good:
                continue
            T, C, Kc, orders, kept, modal = run_mode(case, good, mode, ARGS.rounds, ARGS.warmup, rng)
            if RANK != 0:
                continue
            rounds = T.shape[1]
            n_kept = sum(kept)
            meta["cases"][str(M)]["modes"][mode] = {"rounds": rounds, "kept": n_kept,
                                                     "modal_sm_clock_per_rank": modal}
            rmax = T.max(dim=0).values  # (rounds, items)
            rmin = T.min(dim=0).values
            cmax = C.max(dim=0).values
            kmin = Kc.clamp(min=0).min(dim=0).values
            for r in range(rounds):
                for j, it in enumerate(good):
                    wr.writerow([layer.name, layer.N, layer.K, M, mode, r, it,
                                 orders[r].index(it), f"{rmax[r, j]:.5f}", f"{rmin[r, j]:.5f}",
                                 f"{T[0, r, j]:.5f}", f"{cmax[r, j]:.5f}", int(kmin[r]),
                                 int(kept[r])])
            fh.flush()
            idx = {it: j for j, it in enumerate(good)}
            ks = [r for r in range(rounds) if kept[r]]
            print(f"\n[{layer.name} N={layer.N} K={layer.K} M={M} mode={mode}] "
                  f"kept {n_kept}/{rounds}  modal clock {modal}")
            print(f"  {'item':<20} {'median':>9} {'p10':>9} {'p90':>9} {'cpu/call':>9}  ms (rank-max)")
            for it in good:
                v = [rmax[r, idx[it]].item() for r in ks]
                cv = [cmax[r, idx[it]].item() for r in ks]
                print(f"  {it:<20} {statistics.median(v):9.4f} {q(v, .1):9.4f} {q(v, .9):9.4f} "
                      f"{statistics.median(cv):9.4f}")

            def diff(a, b_list):
                if a not in idx or any(b not in idx for b in b_list):
                    return None
                return [rmax[r, idx[a]].item() - sum(rmax[r, idx[b]].item() for b in b_list)
                        for r in ks]
            print(f"  {'per-round diff':<28} {'median':>9} {'p10':>9} {'p90':>9}")
            for name, a, bl in [("A - B", "A_fused", ["B_nccl_cublas"]),
                                ("A - C", "A_fused", ["C_fluxag_cublas"]),
                                ("A - D", "A_fused", ["D_fluxag_fluxgemm"]),
                                ("D - C", "D_fluxag_fluxgemm", ["C_fluxag_cublas"]),
                                ("C - B", "C_fluxag_cublas", ["B_nccl_cublas"]),
                                ("D - (fluxag+fluxgemm)", "D_fluxag_fluxgemm", ["c_flux_ag", "c_fluxgemm"]),
                                ("B - (ncclag+cublas)", "B_nccl_cublas", ["c_nccl_ag", "c_cublas"]),
                                ("C - (fluxag+cublas)", "C_fluxag_cublas", ["c_flux_ag", "c_cublas"])]:
                d = diff(a, bl)
                if d:
                    print(f"  {name:<28} {statistics.median(d):9.4f} {q(d, .1):9.4f} {q(d, .9):9.4f}")
        del case
        torch.cuda.empty_cache()

    if RANK == 0:
        fh.close()
        old = {}
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                old = json.load(f)
        old.setdefault("runs", []).append(meta)
        with open(meta_path, "w") as f:
            json.dump(old, f, indent=1)
        print(f"\nwrote {raw_path}\nwrote {meta_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--layer", required=True, choices=list(LAYERS))
    p.add_argument("--Ms", required=True, help="comma list of global M")
    p.add_argument("--max_m", type=int, default=0, help="AGKernel full_m; default max(Ms)")
    p.add_argument("--modes", default="gpu", help="comma list of gpu,host,steady")
    p.add_argument("--items", default="", help="comma list; default all")
    p.add_argument("--rounds", type=int, default=200)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--steady_L", type=int, default=16)
    p.add_argument("--pad_cycles", type=int, default=400_000, help="GPU sleep before align (gpu mode)")
    p.add_argument("--flush_mb", type=int, default=128)
    p.add_argument("--clock_period", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=20260929)
    p.add_argument("--atol", type=float, default=0.05)
    p.add_argument("--rtol", type=float, default=0.05)
    p.add_argument("--out_dir", required=True)
    return p.parse_args()


if __name__ == "__main__":
    TP_GROUP = initialize_distributed()
    # undo init_seed(): production cuBLAS state, before any cuBLAS handle exists
    os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    torch.use_deterministic_algorithms(False)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    CUBLAS_SETTINGS = {
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "allow_bf16_reduced_precision_reduction":
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
    }
    RANK, W = TP_GROUP.rank(), TP_GROUP.size()
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    NNODES = flux.testing.NNODES()
    ARGS = parse_args()
    FLUSH = torch.empty(ARGS.flush_mb * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
    ALIGN = torch.zeros(1, dtype=torch.float32, device="cuda")
    main()
    dist.destroy_process_group()
