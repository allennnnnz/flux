#!/usr/bin/env python3
"""Pure NCCL AG/RS timing for Phase 0 E."""

from __future__ import annotations

import argparse
import os
import statistics

import torch
import torch.distributed as dist


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("op", choices=("ag", "rs"))
    parser.add_argument("M", type=int)
    parser.add_argument("N", type=int)
    parser.add_argument("K", type=int)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(name)


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    dtype = dtype_from_name(args.dtype)

    if args.op == "ag":
        if args.M % world:
            raise ValueError("M must be divisible by world size")
        local_m = args.M // world
        src = torch.empty((local_m, args.K), dtype=dtype, device="cuda")
        dst = torch.empty((args.M, args.K), dtype=dtype, device="cuda")
        bytes_per_rank = src.numel() * src.element_size()

        def run() -> None:
            dist.all_gather_into_tensor(dst, src)

    else:
        if args.M % world:
            raise ValueError("M must be divisible by world size")
        src = torch.empty((args.M, args.N), dtype=dtype, device="cuda")
        dst = torch.empty((args.M // world, args.N), dtype=dtype, device="cuda")
        bytes_per_rank = src.numel() * src.element_size()

        def run() -> None:
            dist.reduce_scatter_tensor(dst, src)

    dist.barrier()
    for _ in range(args.warmup):
        run()
    torch.cuda.synchronize()
    dist.barrier()

    times = []
    for _ in range(args.iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        run()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    dist.barrier()

    med = statistics.median(times)
    stdev = statistics.stdev(times) if len(times) > 1 else 0.0
    if rank == 0:
        print(
            "summary,"
            f"op={args.op},world={world},M={args.M},N={args.N},K={args.K},dtype={args.dtype},"
            f"warmup={args.warmup},iters={args.iters},median_ms={med:.6f},"
            f"stdev_ms={stdev:.6f},bytes_per_rank={bytes_per_rank}"
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
