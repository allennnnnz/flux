################################################################################
# Flux-native AllGather communication baseline.
#
# Measures flux.AllGatherOp -- the same comm path AGKernel.forward() uses
# internally -- in isolation, instead of substituting NCCL all_gather.
#
# Usage:
#   ./launch.sh experiments/2026-09-23-phase0-a100-nvlink-pcie/scripts/flux_comm_baseline.py \
#       4096 12288 --dtype=float16 --warmup=5 --iters=20
################################################################################

import argparse
import csv
import os
import statistics
import sys
from functools import partial

import torch
import torch.distributed as dist

import flux
from flux.testing import initialize_distributed

print = partial(print, file=sys.stderr)

DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _bench(fn, warmup, iters, tag):
    """CUDA-event timing with a process-group barrier before every timed iter."""
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    stop_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier(TP_GROUP)

    for i in range(iters):
        dist.barrier(TP_GROUP)
        start_events[i].record()
        fn()
        stop_events[i].record()
    torch.cuda.synchronize()

    times_ms = [s.elapsed_time(e) for s, e in zip(start_events, stop_events)]
    median = statistics.median(times_ms)
    stdev = statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0

    # The collective is only finished when the slowest rank is finished, so the
    # rank-max median is the number that describes the collective.
    local = torch.tensor([median, stdev], dtype=torch.float64, device="cuda")
    gathered = [torch.zeros_like(local) for _ in range(WORLD_SIZE)]
    dist.all_gather(gathered, local, group=TP_GROUP)
    medians = [float(g[0].item()) for g in gathered]
    stdevs = [float(g[1].item()) for g in gathered]

    return {
        "tag": tag,
        "rank_max_median_ms": max(medians),
        "rank_min_median_ms": min(medians),
        "rank0_median_ms": medians[0],
        "rank0_stdev_ms": stdevs[0],
        "per_rank_median_ms": medians,
        "raw_rank0_ms": times_ms,
    }


def build_ag_option(mode, use_read, use_cuda_core_ag):
    opt = flux.AllGatherOption()
    opt.mode = mode
    opt.use_read = use_read
    opt.use_cuda_core_ag = use_cuda_core_ag
    opt.use_cuda_core_local = False
    opt.fuse_sync = False
    # False => the op also performs the local shard copy, which is part of the
    # real all-gather cost inside AGKernel.forward().
    opt.input_buffer_copied = False
    return opt


def main():
    M, K = args.M, args.K
    dtype = DTYPE_MAP[args.dtype]
    assert M % WORLD_SIZE == 0, f"M={M} not divisible by world_size={WORLD_SIZE}"
    M_per_rank = M // WORLD_SIZE

    local_input = torch.randn((M_per_rank, K), dtype=torch.float32, device="cuda").to(dtype)

    # ---- reference: NCCL all_gather ------------------------------------------
    nccl_out = torch.zeros((M, K), dtype=dtype, device="cuda")
    dist.all_gather_into_tensor(nccl_out, local_input, group=TP_GROUP)
    torch.cuda.synchronize()
    reference = nccl_out.clone()

    results = []
    results.append(
        _bench(
            lambda: dist.all_gather_into_tensor(nccl_out, local_input, group=TP_GROUP),
            args.warmup,
            args.iters,
            "nccl_all_gather",
        )
    )

    # ---- flux.AllGatherOp ----------------------------------------------------
    ag_op = flux.AllGatherOp(TP_GROUP, NNODES, M, K, dtype)
    ag_buffer = ag_op.local_input_buffer()[0:M, :]
    stream = torch.cuda.current_stream()

    configs = [
        ("flux_all2all_push", flux.AGRingMode.All2All, False, False),
        ("flux_all2all_pull", flux.AGRingMode.All2All, True, False),
        ("flux_ring1d_push", flux.AGRingMode.Ring1D, False, False),
        ("flux_ring1d_pull", flux.AGRingMode.Ring1D, True, False),
        ("flux_ring2d_push", flux.AGRingMode.Ring2D, False, False),
        ("flux_all2all_push_cudacore", flux.AGRingMode.All2All, False, True),
        ("flux_ring2d_push_cudacore", flux.AGRingMode.Ring2D, False, True),
    ]

    for tag, mode, use_read, use_cuda_core_ag in configs:
        opt = build_ag_option(mode, use_read, use_cuda_core_ag)

        def run():
            ag_op.run(local_input, None, opt, stream.cuda_stream)

        # correctness against the NCCL reference before timing
        ag_buffer.zero_()
        try:
            run()
            torch.cuda.synchronize()
        except RuntimeError as exc:
            # e.g. the CUDA-core AG kernel is only instantiated for INT8+FP32 scale
            if RANK == 0:
                print(f"[{tag}] UNSUPPORTED: {str(exc).strip().splitlines()[-1]}")
            dist.barrier(TP_GROUP)
            continue
        dist.barrier(TP_GROUP)
        match = torch.equal(ag_buffer, reference)
        ok = torch.tensor([1 if match else 0], dtype=torch.int32, device="cuda")
        dist.all_reduce(ok, group=TP_GROUP)
        all_match = int(ok.item()) == WORLD_SIZE
        if RANK == 0:
            print(f"[{tag}] bitwise vs nccl all_gather on all ranks: {all_match}")
        if not all_match:
            if RANK == 0:
                print(f"[{tag}] SKIPPED timing: incorrect result")
            continue

        res = _bench(run, args.warmup, args.iters, tag)
        res["correct"] = all_match
        results.append(res)

    if RANK == 0:
        nbytes_in = M_per_rank * K * local_input.element_size()
        nbytes_out = M * K * local_input.element_size()
        print()
        print(f"M={M} K={K} dtype={args.dtype} world={WORLD_SIZE} nnodes={NNODES}")
        print(f"input bytes/rank={nbytes_in} output bytes={nbytes_out}")
        print()
        header = f"{'tag':<30} {'rank-max med ms':>16} {'rank0 med ms':>14} {'rank0 stdev':>12} {'algbw GB/s':>12}"
        print(header)
        print("-" * len(header))
        for r in results:
            algbw = nbytes_out / (r["rank_max_median_ms"] / 1e3) / 1e9
            print(
                f"{r['tag']:<30} {r['rank_max_median_ms']:>16.6f} "
                f"{r['rank0_median_ms']:>14.6f} {r['rank0_stdev_ms']:>12.6f} {algbw:>12.3f}"
            )

        if args.out:
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            with open(args.out, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(
                    [
                        "tag",
                        "rank_max_median_ms",
                        "rank_min_median_ms",
                        "rank0_median_ms",
                        "rank0_stdev_ms",
                        "output_bytes",
                        "algbw_GBps",
                        "per_rank_median_ms",
                    ]
                )
                for r in results:
                    algbw = nbytes_out / (r["rank_max_median_ms"] / 1e3) / 1e9
                    w.writerow(
                        [
                            r["tag"],
                            f"{r['rank_max_median_ms']:.6f}",
                            f"{r['rank_min_median_ms']:.6f}",
                            f"{r['rank0_median_ms']:.6f}",
                            f"{r['rank0_stdev_ms']:.6f}",
                            nbytes_out,
                            f"{algbw:.3f}",
                            ";".join(f"{v:.6f}" for v in r["per_rank_median_ms"]),
                        ]
                    )
            print(f"\nwrote {args.out}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("M", type=int, help="full M (gathered), input shard is M // world_size")
    p.add_argument("K", type=int)
    p.add_argument("--dtype", default="float16", choices=list(DTYPE_MAP))
    p.add_argument("--warmup", default=5, type=int)
    p.add_argument("--iters", default=20, type=int)
    p.add_argument("--out", default="", type=str, help="CSV output path")
    return p.parse_args()


if __name__ == "__main__":
    TP_GROUP = initialize_distributed()
    RANK, WORLD_SIZE = TP_GROUP.rank(), TP_GROUP.size()
    NNODES = flux.testing.NNODES()
    args = parse_args()
    main()
    dist.destroy_process_group()
