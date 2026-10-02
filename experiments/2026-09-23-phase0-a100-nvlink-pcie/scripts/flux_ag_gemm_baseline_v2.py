################################################################################
# Flux AG+GEMM baseline with a defensible exposed-comm-time measurement.
#
# Replaces the ECT method used in CDE_REPORT.md section E, which subtracted two
# separately-averaged loops and was not stable enough to quote: gemm_only swung
# 2.528-2.877 ms across repeats and one run produced a NEGATIVE ECT. The script
# it came from also reports two mutually inconsistent "GEMM alone" numbers -- the
# printed `comm` field uses a separate flux.GemmOnly op, not AGKernel.gemm_only.
#
# Method here:
#   * The GEMM-only reference is AGKernel.gemm_only(), which runs the SAME
#     AG-fused GEMM kernel with the full input already in place and a barrier
#     tensor of all ones, i.e. every signal pre-set to true
#     (src/ag_gemm/ths_op/all_gather_gemm_op.cc:167). It is not a different op.
#   * Overlap and GEMM-only are INTERLEAVED inside each round and timed with
#     their own CUDA events, so ECT is a per-round difference of two adjacent
#     measurements rather than a difference of two separate averages.
#   * >=200 rounds; the reported ECT is the median of the per-round differences.
#   * SM clock is sampled in the background; rounds overlapping a downclock are
#     discarded.
#
# Usage:
#   ./launch.sh experiments/.../scripts/flux_ag_gemm_baseline_v2.py \
#       4096 49152 12288 --dtype=bfloat16 --rounds=200
################################################################################

import argparse
import csv
import os
import statistics
import subprocess
import sys
import threading
import time
from functools import partial

import torch
import torch.distributed as dist

import flux
from flux.testing import initialize_distributed

print = partial(print, file=sys.stderr)

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16}
RING = {"all2all": flux.AGRingMode.All2All, "ring1d": flux.AGRingMode.Ring1D,
        "ring2d": flux.AGRingMode.Ring2D}


class ClockLogger:
    """Samples SM clock for one device in the background."""

    def __init__(self, device: int, period_s: float = 0.02):
        self.device, self.period = device, period_s
        self.samples: list[tuple[float, int]] = []
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader,nounits",
                     "-i", str(self.device)],
                    capture_output=True, text=True, timeout=2,
                )
                self.samples.append((time.perf_counter(), int(out.stdout.strip().split()[0])))
            except Exception:
                pass
            self._stop.wait(self.period)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._t.join(timeout=3)

    def min_clock_between(self, t0: float, t1: float) -> int | None:
        """Min sample inside the window; rounds are shorter than the sampling
        period, so fall back to the most recent sample at or before t0."""
        v = [c for t, c in self.samples if t0 <= t <= t1]
        if v:
            return min(v)
        prev = [c for t, c in self.samples if t <= t0]
        return prev[-1] if prev else None


def main():
    M, N, K = args.M, args.N, args.K
    dtype = DTYPES[args.dtype]
    M_per_rank = M // WORLD_SIZE
    # repo convention (test_ag_kernel.py:99): the weight is column-sharded, so each
    # rank holds (N // world_size, K) and produces an (M, N_per_rank) output.
    N_per_rank = N // WORLD_SIZE
    assert M % WORLD_SIZE == 0 and N % WORLD_SIZE == 0

    local_input = torch.randn((M_per_rank, K), dtype=torch.float32, device="cuda").to(dtype)
    weight = (torch.randn((N_per_rank, K), dtype=torch.float32, device="cuda") * 0.01).to(dtype)

    full_input = torch.zeros((M, K), dtype=dtype, device="cuda")
    dist.all_gather_into_tensor(full_input, local_input, group=TP_GROUP)
    torch.cuda.synchronize()

    op = flux.AGKernel(TP_GROUP, NNODES, M, N_per_rank, K, dtype, output_dtype=dtype)
    out = torch.empty((M, N_per_rank), dtype=dtype, device="cuda")

    ag_opt = flux.AllGatherOption()
    ag_opt.mode = RING[args.ring_mode]
    ag_opt.use_read = args.use_read
    ag_opt.use_cuda_core_local = False
    ag_opt.use_cuda_core_ag = False
    ag_opt.fuse_sync = False
    ag_opt.input_buffer_copied = False

    def overlap():
        return op.forward(local_input, weight, output=out, transpose_weight=False,
                          all_gather_option=ag_opt)

    def gemm_only():
        # same AG-fused GEMM kernel, full input in place, all signals pre-set true
        return op.gemm_only(full_input, weight, transpose_weight=False)

    # correctness once, against torch
    ref = torch.matmul(full_input, weight.t())
    got = overlap()
    torch.cuda.synchronize()
    close = torch.allclose(got.float(), ref.float(), atol=args.atol, rtol=args.rtol)
    ok = torch.tensor([1 if close else 0], dtype=torch.int32, device="cuda")
    dist.all_reduce(ok, group=TP_GROUP)
    all_ok = int(ok.item()) == WORLD_SIZE
    if RANK == 0:
        print(f"allclose vs torch on all ranks: {all_ok}")

    for _ in range(args.warmup):
        overlap()
        gemm_only()
    torch.cuda.synchronize()
    dist.barrier(TP_GROUP)

    rounds = args.rounds
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True),
           torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
          for _ in range(rounds)]
    wall = []

    with ClockLogger(LOCAL_RANK) as clk:
        for i in range(rounds):
            dist.barrier(TP_GROUP)
            t0 = time.perf_counter()
            a0, a1, g0, g1 = ev[i]
            a0.record()
            overlap()
            a1.record()
            g0.record()
            gemm_only()
            g1.record()
            torch.cuda.synchronize()
            wall.append((t0, time.perf_counter()))
        torch.cuda.synchronize()

        rows = []
        for i in range(rounds):
            a0, a1, g0, g1 = ev[i]
            ov_ms, go_ms = a0.elapsed_time(a1), g0.elapsed_time(g1)
            c = clk.min_clock_between(*wall[i])
            rows.append((ov_ms, go_ms, ov_ms - go_ms, c))

    clocks = [c for *_, c in rows if c]
    modal = statistics.mode(clocks) if clocks else None
    kept = [r for r in rows if r[3] is None or modal is None or r[3] >= 0.95 * modal]
    dropped = len(rows) - len(kept)

    ov = [r[0] for r in kept]
    go = [r[1] for r in kept]
    ect = [r[2] for r in kept]

    # gather rank-max of each round's overlap time is not meaningful across ranks here;
    # report rank 0 and the spread of per-rank medians
    local = torch.tensor(
        [statistics.median(ov), statistics.median(go), statistics.median(ect)],
        dtype=torch.float64, device="cuda")
    g = [torch.zeros_like(local) for _ in range(WORLD_SIZE)]
    dist.all_gather(g, local, group=TP_GROUP)
    per_rank = [[float(x[j].item()) for x in g] for j in range(3)]

    if RANK == 0:
        def q(v, p):
            s = sorted(v)
            return s[max(0, min(len(s) - 1, int(p * len(s))))]

        print()
        print(f"M={M} N={N} K={K} dtype={args.dtype} ring={args.ring_mode} "
              f"use_read={args.use_read} world={WORLD_SIZE}")
        print(f"rounds={rounds} kept={len(kept)} dropped_for_clock={dropped} "
              f"modal_sm_clock={modal} MHz")
        print()
        print(f"{'quantity':<28} {'median':>10} {'p10':>10} {'p90':>10} {'stdev':>10}")
        print("-" * 72)
        for name, v in [("AG+GEMM overlapped ms", ov), ("GEMM only ms", go),
                        ("ECT (per-round diff) ms", ect)]:
            print(f"{name:<28} {statistics.median(v):>10.4f} {q(v,0.1):>10.4f} "
                  f"{q(v,0.9):>10.4f} {statistics.stdev(v):>10.4f}")
        neg = sum(1 for x in ect if x < 0)
        print()
        print(f"negative ECT rounds: {neg}/{len(ect)} ({neg/len(ect)*100:.1f}%)")
        print(f"per-rank median overlapped ms: "
              f"min {min(per_rank[0]):.4f} max {max(per_rank[0]):.4f}")
        print(f"per-rank median ECT ms:        "
              f"min {min(per_rank[2]):.4f} max {max(per_rank[2]):.4f}")

        if args.out:
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            with open(args.out, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["M", "N", "K", "dtype", "ring_mode", "use_read", "world",
                            "rounds", "kept", "dropped_for_clock", "modal_sm_clock_MHz",
                            "overlap_median_ms", "overlap_p10_ms", "overlap_p90_ms",
                            "gemm_only_median_ms", "ect_median_ms", "ect_p10_ms",
                            "ect_p90_ms", "ect_stdev_ms", "negative_ect_rounds"])
                w.writerow([M, N, K, args.dtype, args.ring_mode, args.use_read, WORLD_SIZE,
                            rounds, len(kept), dropped, modal,
                            f"{statistics.median(ov):.4f}", f"{q(ov,0.1):.4f}",
                            f"{q(ov,0.9):.4f}", f"{statistics.median(go):.4f}",
                            f"{statistics.median(ect):.4f}", f"{q(ect,0.1):.4f}",
                            f"{q(ect,0.9):.4f}", f"{statistics.stdev(ect):.4f}", neg])
            print(f"\nwrote {args.out}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("M", type=int)
    p.add_argument("N", type=int)
    p.add_argument("K", type=int)
    p.add_argument("--dtype", default="bfloat16", choices=list(DTYPES))
    p.add_argument("--ring_mode", default="all2all", choices=list(RING))
    p.add_argument("--use_read", default=True, action=argparse.BooleanOptionalAction)
    p.add_argument("--rounds", default=200, type=int)
    p.add_argument("--warmup", default=20, type=int)
    p.add_argument("--atol", default=0.05, type=float)
    p.add_argument("--rtol", default=0.05, type=float)
    p.add_argument("--out", default="")
    return p.parse_args()


if __name__ == "__main__":
    TP_GROUP = initialize_distributed()
    RANK, WORLD_SIZE = TP_GROUP.rank(), TP_GROUP.size()
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    NNODES = flux.testing.NNODES()
    args = parse_args()
    main()
    dist.destroy_process_group()
