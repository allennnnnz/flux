#!/usr/bin/env python3
"""Measure aggregate H2D/D2H bandwidth with wall time and CUDA events."""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import math
import multiprocessing as mp
import os
import statistics
import time
from dataclasses import dataclass


GPU_NUMA_NODE = {
    0: 0,
    1: 0,
    2: 0,
    3: 0,
    4: 1,
    5: 1,
    6: 1,
    7: 1,
}


@dataclass(frozen=True)
class Trial:
    repeat: int
    wall_seconds: float
    aggregate_gbps: float
    per_gpu_gbps: float
    start_skew_us: float
    median_cuda_ms: float
    stdev_cuda_ms: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--direction", choices=("htod", "dtoh"), required=True)
    parser.add_argument("--size-mib", type=int, default=1024)
    parser.add_argument("--copies", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--numa-mode", choices=("none", "local", "remote"), default="none")
    parser.add_argument("--csv", action="store_true")
    return parser.parse_args()


def parse_devices(devices: str) -> list[int]:
    parsed = [int(item) for item in devices.split(",") if item.strip()]
    if not parsed:
        raise ValueError("At least one CUDA device is required")
    return parsed


def cpus_for_node(node: int) -> set[int]:
    path = f"/sys/devices/system/node/node{node}/cpulist"
    text = open(path, encoding="ascii").read().strip()
    cpus: set[int] = set()
    for part in text.split(","):
        if "-" in part:
            start, end = part.split("-", 1)
            cpus.update(range(int(start), int(end) + 1))
        else:
            cpus.add(int(part))
    return cpus


def bind_to_numa_node(node: int) -> None:
    os.sched_setaffinity(0, cpus_for_node(node))
    libname = ctypes.util.find_library("numa")
    if libname is None:
        raise RuntimeError("libnuma is unavailable; cannot set memory NUMA binding")
    numa = ctypes.CDLL(libname, use_errno=True)
    if numa.numa_available() < 0:
        raise RuntimeError("NUMA is unavailable")
    numa.numa_run_on_node.argtypes = [ctypes.c_int]
    numa.numa_run_on_node.restype = ctypes.c_int
    if numa.numa_run_on_node(node) != 0:
        errno = ctypes.get_errno()
        raise OSError(errno, f"numa_run_on_node({node}) failed")
    numa.numa_parse_nodestring.argtypes = [ctypes.c_char_p]
    numa.numa_parse_nodestring.restype = ctypes.c_void_p
    numa.numa_set_membind.argtypes = [ctypes.c_void_p]
    numa.numa_bitmask_free.argtypes = [ctypes.c_void_p]
    mask = numa.numa_parse_nodestring(str(node).encode("ascii"))
    if not mask:
        raise RuntimeError(f"failed to build NUMA mask for node {node}")
    try:
        numa.numa_set_membind(mask)
    finally:
        numa.numa_bitmask_free(mask)


def numa_node_for(device: int, mode: str) -> int | None:
    if mode == "none":
        return None
    local = GPU_NUMA_NODE[device]
    if mode == "local":
        return local
    return 1 - local


def worker(
    device: int,
    size_bytes: int,
    copies: int,
    direction: str,
    warmup: int,
    repeats: int,
    numa_mode: str,
    barrier: object,
    result_queue: object,
) -> None:
    node = numa_node_for(device, numa_mode)
    if node is not None:
        bind_to_numa_node(node)

    import torch

    torch.cuda.set_device(device)
    elements = size_bytes // 4
    gpu = torch.empty(elements, device=f"cuda:{device}", dtype=torch.float32)
    host = torch.empty(elements, dtype=torch.float32, pin_memory=True)
    stream = torch.cuda.Stream(device=device)

    def do_copies(count: int) -> None:
        with torch.cuda.stream(stream):
            for _ in range(count):
                if direction == "htod":
                    gpu.copy_(host, non_blocking=True)
                else:
                    host.copy_(gpu, non_blocking=True)
        stream.synchronize()

    for _ in range(warmup):
        do_copies(copies)

    rows = []
    for repeat in range(repeats):
        barrier.wait()
        start_time = time.perf_counter()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(stream):
            start_event.record(stream)
            for _ in range(copies):
                if direction == "htod":
                    gpu.copy_(host, non_blocking=True)
                else:
                    host.copy_(gpu, non_blocking=True)
            end_event.record(stream)
        end_event.synchronize()
        cuda_ms = start_event.elapsed_time(end_event)
        barrier.wait()
        rows.append((repeat, device, node, start_time, cuda_ms))

    result_queue.put(rows)


def format_stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)


def main() -> None:
    args = parse_args()
    devices = parse_devices(args.devices)
    if args.size_mib < 1024 and args.copies < 20:
        raise ValueError("Use either --size-mib >= 1024 or --copies >= 20")

    ctx = mp.get_context("spawn")
    size_bytes = args.size_mib * 1024 * 1024
    moved_bytes = len(devices) * size_bytes * args.copies
    barrier = ctx.Barrier(len(devices) + 1)
    result_queue = ctx.Queue()
    processes = [
        ctx.Process(
            target=worker,
            args=(
                device,
                size_bytes,
                args.copies,
                args.direction,
                args.warmup,
                args.repeats,
                args.numa_mode,
                barrier,
                result_queue,
            ),
        )
        for device in devices
    ]

    print(
        "meta,"
        f"devices={devices},direction={args.direction},size_mib={args.size_mib},"
        f"copies={args.copies},warmup={args.warmup},repeats={args.repeats},"
        f"numa_mode={args.numa_mode}"
    )
    print(
        "repeat,wall_seconds,aggregate_GBps,per_gpu_GBps,"
        "start_skew_us,median_cuda_ms,stdev_cuda_ms"
    )

    for process in processes:
        process.start()

    wall_seconds = []
    for _ in range(args.repeats):
        barrier.wait()
        start = time.perf_counter()
        barrier.wait()
        end = time.perf_counter()
        wall_seconds.append(end - start)

    for process in processes:
        process.join()
        if process.exitcode != 0:
            raise RuntimeError(f"worker exited with code {process.exitcode}")

    rows = []
    while not result_queue.empty():
        rows.extend(result_queue.get())
    by_repeat: dict[int, list[tuple[int, int | None, float, float]]] = {}
    for repeat, device, node, start_time, cuda_ms in rows:
        by_repeat.setdefault(repeat, []).append((device, node, start_time, cuda_ms))

    trials = []
    for repeat, seconds in enumerate(wall_seconds):
        worker_rows = by_repeat[repeat]
        start_times = [item[2] for item in worker_rows]
        cuda_times = [item[3] for item in worker_rows]
        aggregate = moved_bytes / seconds / 1e9
        trials.append(
            Trial(
                repeat=repeat,
                wall_seconds=seconds,
                aggregate_gbps=aggregate,
                per_gpu_gbps=aggregate / len(devices),
                start_skew_us=(max(start_times) - min(start_times)) * 1e6,
                median_cuda_ms=statistics.median(cuda_times),
                stdev_cuda_ms=format_stdev(cuda_times),
            )
        )

    for trial in trials:
        print(
            f"{trial.repeat},{trial.wall_seconds:.9f},{trial.aggregate_gbps:.3f},"
            f"{trial.per_gpu_gbps:.3f},{trial.start_skew_us:.3f},"
            f"{trial.median_cuda_ms:.3f},{trial.stdev_cuda_ms:.3f}"
        )

    aggregates = [trial.aggregate_gbps for trial in trials]
    per_gpu = [trial.per_gpu_gbps for trial in trials]
    skews = [trial.start_skew_us for trial in trials]
    cuda_medians = [trial.median_cuda_ms for trial in trials]
    print(
        "summary,"
        f"aggregate_median_GBps={statistics.median(aggregates):.3f},"
        f"aggregate_stdev_GBps={format_stdev(aggregates):.3f},"
        f"per_gpu_median_GBps={statistics.median(per_gpu):.3f},"
        f"per_gpu_stdev_GBps={format_stdev(per_gpu):.3f},"
        f"start_skew_max_us={max(skews):.3f},"
        f"cuda_median_ms={statistics.median(cuda_medians):.3f},"
        f"cuda_median_stdev_ms={format_stdev(cuda_medians):.3f}"
    )


if __name__ == "__main__":
    main()
