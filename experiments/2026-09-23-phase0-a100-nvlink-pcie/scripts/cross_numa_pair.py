#!/usr/bin/env python3
"""Measure GPU3 D2H and GPU4 H2D against one pinned buffer on a selected NUMA node."""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import statistics
import time


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
    import os

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--buffer-numa-node", type=int, choices=(0, 1), required=True)
    parser.add_argument("--size-mib", type=int, default=1024)
    parser.add_argument("--copies", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    return parser.parse_args()


def stdev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def main() -> None:
    args = parse_args()
    bind_to_numa_node(args.buffer_numa_node)

    import torch

    size_bytes = args.size_mib * 1024 * 1024
    elements = size_bytes // 4
    host = torch.empty(elements, dtype=torch.float32, pin_memory=True)
    with torch.cuda.device(3):
        gpu3 = torch.empty(elements, dtype=torch.float32, device="cuda:3")
        d2h_stream = torch.cuda.Stream(device=3)
    with torch.cuda.device(4):
        gpu4 = torch.empty(elements, dtype=torch.float32, device="cuda:4")
        h2d_stream = torch.cuda.Stream(device=4)

    def run_once(record_events: bool) -> tuple[float, float, float]:
        d2h_start = torch.cuda.Event(enable_timing=True) if record_events else None
        d2h_end = torch.cuda.Event(enable_timing=True) if record_events else None
        h2d_start = torch.cuda.Event(enable_timing=True) if record_events else None
        h2d_end = torch.cuda.Event(enable_timing=True) if record_events else None
        start = time.perf_counter()
        with torch.cuda.device(3), torch.cuda.stream(d2h_stream):
            if record_events:
                d2h_start.record(d2h_stream)
            for _ in range(args.copies):
                host.copy_(gpu3, non_blocking=True)
            if record_events:
                d2h_end.record(d2h_stream)
        with torch.cuda.device(4), torch.cuda.stream(h2d_stream):
            if record_events:
                h2d_start.record(h2d_stream)
            for _ in range(args.copies):
                gpu4.copy_(host, non_blocking=True)
            if record_events:
                h2d_end.record(h2d_stream)
        d2h_stream.synchronize()
        h2d_stream.synchronize()
        wall = time.perf_counter() - start
        if not record_events:
            return wall, 0.0, 0.0
        return wall, d2h_start.elapsed_time(d2h_end), h2d_start.elapsed_time(h2d_end)

    for _ in range(args.warmup):
        run_once(False)

    print(
        "meta,"
        f"gpu3_d2h_to_buffer,gpu4_h2d_from_buffer,buffer_numa={args.buffer_numa_node},"
        f"size_mib={args.size_mib},copies={args.copies},warmup={args.warmup},repeats={args.repeats}"
    )
    print("repeat,wall_seconds,d2h_GBps,h2d_GBps,d2h_cuda_ms,h2d_cuda_ms")
    d2h_rates = []
    h2d_rates = []
    bytes_per_direction = size_bytes * args.copies
    for repeat in range(args.repeats):
        wall, d2h_ms, h2d_ms = run_once(True)
        d2h = bytes_per_direction / wall / 1e9
        h2d = bytes_per_direction / wall / 1e9
        d2h_rates.append(d2h)
        h2d_rates.append(h2d)
        print(f"{repeat},{wall:.9f},{d2h:.3f},{h2d:.3f},{d2h_ms:.3f},{h2d_ms:.3f}")
    print(
        "summary,"
        f"d2h_median_GBps={statistics.median(d2h_rates):.3f},"
        f"d2h_stdev_GBps={stdev(d2h_rates):.3f},"
        f"h2d_median_GBps={statistics.median(h2d_rates):.3f},"
        f"h2d_stdev_GBps={stdev(h2d_rates):.3f}"
    )


if __name__ == "__main__":
    main()
