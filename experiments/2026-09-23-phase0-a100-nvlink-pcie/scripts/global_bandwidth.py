#!/usr/bin/env python3
"""Measure aggregate multi-process H2D/D2H bandwidth with one global wall clock."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--direction", choices=("htod", "dtoh"), required=True)
    parser.add_argument("--size-mib", type=int, nargs="+", default=[1, 4, 16, 32, 64, 128])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--numa-node", type=int, default=None)
    return parser.parse_args()


def worker(
    device: int,
    size_bytes: int,
    direction: str,
    repeats: int,
    barrier: object,
    result_queue: object,
) -> None:
    import torch

    torch.cuda.set_device(device)
    elements = size_bytes // 4
    gpu = torch.empty(elements, device=f"cuda:{device}", dtype=torch.float32)
    host = torch.empty(elements, dtype=torch.float32, pin_memory=True)
    stream = torch.cuda.Stream(device=device)

    for _ in range(2):
        with torch.cuda.stream(stream):
            if direction == "htod":
                gpu.copy_(host, non_blocking=True)
            else:
                host.copy_(gpu, non_blocking=True)
        stream.synchronize()

    timings = []
    for _ in range(repeats):
        barrier.wait()
        start = time.perf_counter()
        with torch.cuda.stream(stream):
            if direction == "htod":
                gpu.copy_(host, non_blocking=True)
            else:
                host.copy_(gpu, non_blocking=True)
        stream.synchronize()
        end = time.perf_counter()
        timings.append(end - start)
        barrier.wait()

    result_queue.put((device, timings))


def main() -> None:
    args = parse_args()
    devices = [int(item) for item in args.devices.split(",") if item.strip()]
    if not devices:
        raise ValueError("At least one device is required")

    ctx = mp.get_context("spawn")
    print(
        f"devices={devices} direction={args.direction} repeats={args.repeats} "
        f"numa_node={args.numa_node}"
    )
    print("size_MiB,repeat,wall_seconds,aggregate_GBps,per_gpu_GBps")

    for size_mib in args.size_mib:
        size_bytes = size_mib * 1024 * 1024
        barrier = ctx.Barrier(len(devices) + 1)
        result_queue = ctx.Queue()
        processes = [
            ctx.Process(
                target=worker,
                args=(
                    device,
                    size_bytes,
                    args.direction,
                    args.repeats,
                    barrier,
                    result_queue,
                ),
            )
            for device in devices
        ]
        for process in processes:
            process.start()

        all_timings = []
        for repeat in range(args.repeats):
            barrier.wait()
            start = time.perf_counter()
            barrier.wait()
            end = time.perf_counter()
            all_timings.append(end - start)

        for process in processes:
            process.join()
            if process.exitcode != 0:
                raise RuntimeError(f"worker exited with code {process.exitcode}")
        while not result_queue.empty():
            result_queue.get()

        for repeat, seconds in enumerate(all_timings):
            aggregate = len(devices) * size_bytes / seconds / 1e9
            per_gpu = aggregate / len(devices)
            print(f"{size_mib},{repeat},{seconds:.6f},{aggregate:.3f},{per_gpu:.3f}")


if __name__ == "__main__":
    main()
