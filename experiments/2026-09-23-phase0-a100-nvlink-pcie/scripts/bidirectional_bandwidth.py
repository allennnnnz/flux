#!/usr/bin/env python3
"""Measure simultaneous H2D and D2H bandwidth with the Phase 0 v2 method."""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import multiprocessing as mp
import os
import statistics
import time


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--size-mib", type=int, default=1024)
    parser.add_argument("--copies", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--numa-mode", choices=("none", "local", "remote"), default="local")
    return parser.parse_args()


def cpus_for_node(node: int) -> set[int]:
    text = open(f"/sys/devices/system/node/node{node}/cpulist", encoding="ascii").read().strip()
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
        raise RuntimeError("libnuma is unavailable")
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
        raise RuntimeError(f"failed to parse NUMA node {node}")
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
    host_in = torch.empty(elements, dtype=torch.float32, pin_memory=True)
    host_out = torch.empty(elements, dtype=torch.float32, pin_memory=True)
    gpu = torch.empty(elements, dtype=torch.float32, device=f"cuda:{device}")
    h2d_stream = torch.cuda.Stream(device=device)
    d2h_stream = torch.cuda.Stream(device=device)

    def run_once(record_events: bool) -> tuple[float, float] | None:
        h2d_start = torch.cuda.Event(enable_timing=True) if record_events else None
        h2d_end = torch.cuda.Event(enable_timing=True) if record_events else None
        d2h_start = torch.cuda.Event(enable_timing=True) if record_events else None
        d2h_end = torch.cuda.Event(enable_timing=True) if record_events else None
        with torch.cuda.stream(h2d_stream):
            if record_events:
                h2d_start.record(h2d_stream)
            for _ in range(copies):
                gpu.copy_(host_in, non_blocking=True)
            if record_events:
                h2d_end.record(h2d_stream)
        with torch.cuda.stream(d2h_stream):
            if record_events:
                d2h_start.record(d2h_stream)
            for _ in range(copies):
                host_out.copy_(gpu, non_blocking=True)
            if record_events:
                d2h_end.record(d2h_stream)
        h2d_stream.synchronize()
        d2h_stream.synchronize()
        if not record_events:
            return None
        return h2d_start.elapsed_time(h2d_end), d2h_start.elapsed_time(d2h_end)

    for _ in range(warmup):
        run_once(False)

    rows = []
    for repeat in range(repeats):
        barrier.wait()
        start_time = time.perf_counter()
        h2d_ms, d2h_ms = run_once(True)
        barrier.wait()
        rows.append((repeat, device, node, start_time, h2d_ms, d2h_ms))
    result_queue.put(rows)


def stdev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def main() -> None:
    args = parse_args()
    devices = [int(item) for item in args.devices.split(",") if item.strip()]
    if not devices:
        raise ValueError("At least one device is required")
    if args.size_mib < 1024 and args.copies < 20:
        raise ValueError("Use either --size-mib >= 1024 or --copies >= 20")

    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(len(devices) + 1)
    result_queue = ctx.Queue()
    size_bytes = args.size_mib * 1024 * 1024
    bytes_per_direction = len(devices) * size_bytes * args.copies
    processes = [
        ctx.Process(
            target=worker,
            args=(
                device,
                size_bytes,
                args.copies,
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
        f"devices={devices},size_mib={args.size_mib},copies={args.copies},"
        f"warmup={args.warmup},repeats={args.repeats},numa_mode={args.numa_mode}"
    )
    print(
        "repeat,wall_seconds,h2d_GBps,d2h_GBps,h2d_per_gpu_GBps,d2h_per_gpu_GBps,"
        "start_skew_us,h2d_cuda_median_ms,d2h_cuda_median_ms"
    )
    for process in processes:
        process.start()

    walls = []
    for _ in range(args.repeats):
        barrier.wait()
        start = time.perf_counter()
        barrier.wait()
        walls.append(time.perf_counter() - start)

    for process in processes:
        process.join()
        if process.exitcode != 0:
            raise RuntimeError(f"worker exited with code {process.exitcode}")

    rows = []
    while not result_queue.empty():
        rows.extend(result_queue.get())
    by_repeat: dict[int, list[tuple[int, int | None, float, float, float]]] = {}
    for repeat, device, node, start_time, h2d_ms, d2h_ms in rows:
        by_repeat.setdefault(repeat, []).append((device, node, start_time, h2d_ms, d2h_ms))

    h2d_rates = []
    d2h_rates = []
    h2d_per_gpu = []
    d2h_per_gpu = []
    skews = []
    for repeat, seconds in enumerate(walls):
        worker_rows = by_repeat[repeat]
        h2d = bytes_per_direction / seconds / 1e9
        d2h = bytes_per_direction / seconds / 1e9
        starts = [item[2] for item in worker_rows]
        h2d_events = [item[3] for item in worker_rows]
        d2h_events = [item[4] for item in worker_rows]
        skew = (max(starts) - min(starts)) * 1e6
        h2d_rates.append(h2d)
        d2h_rates.append(d2h)
        h2d_per_gpu.append(h2d / len(devices))
        d2h_per_gpu.append(d2h / len(devices))
        skews.append(skew)
        print(
            f"{repeat},{seconds:.9f},{h2d:.3f},{d2h:.3f},"
            f"{h2d / len(devices):.3f},{d2h / len(devices):.3f},"
            f"{skew:.3f},{statistics.median(h2d_events):.3f},{statistics.median(d2h_events):.3f}"
        )

    print(
        "summary,"
        f"h2d_median_GBps={statistics.median(h2d_rates):.3f},"
        f"h2d_stdev_GBps={stdev(h2d_rates):.3f},"
        f"d2h_median_GBps={statistics.median(d2h_rates):.3f},"
        f"d2h_stdev_GBps={stdev(d2h_rates):.3f},"
        f"h2d_per_gpu_median_GBps={statistics.median(h2d_per_gpu):.3f},"
        f"d2h_per_gpu_median_GBps={statistics.median(d2h_per_gpu):.3f},"
        f"start_skew_max_us={max(skews):.3f}"
    )


if __name__ == "__main__":
    main()
