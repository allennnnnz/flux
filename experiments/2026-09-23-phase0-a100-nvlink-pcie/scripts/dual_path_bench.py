#!/usr/bin/env python3
"""Synthetic dual-path GPU copy benchmark.

Each source GPU sends a payload to the next GPU in a ring. A configurable
fraction of the payload is staged through pinned host memory while the
remainder uses peer GPU copies.
"""

from __future__ import annotations

import argparse
import math
import time

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--bytes-per-gpu", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--alphas",
        default="0,0.02,0.05,0.08,0.10,0.15,1.0",
        help="Comma-separated fraction of bytes staged through pinned host memory.",
    )
    return parser.parse_args()


def synchronize(devices: list[int]) -> None:
    for dev in devices:
        with torch.cuda.device(dev):
            torch.cuda.synchronize()


def time_once(
    devices: list[int],
    nbytes: int,
    alpha: float,
    src: list[torch.Tensor],
    dst: list[torch.Tensor],
    host: list[torch.Tensor],
    peer_streams: list[torch.cuda.Stream],
    host_streams: list[torch.cuda.Stream],
) -> tuple[float, float]:
    host_bytes = int(math.floor(nbytes * alpha))
    host_bytes -= host_bytes % 4
    peer_bytes = nbytes - host_bytes

    for dev, tensor in zip(devices, dst):
        with torch.cuda.device(dev):
            tensor.zero_()
    synchronize(devices)

    start = time.perf_counter()
    events: list[torch.cuda.Event] = []

    for idx, src_dev in enumerate(devices):
        dst_dev = devices[(idx + 1) % len(devices)]
        peer_elems = peer_bytes // 4
        host_elems = host_bytes // 4

        if peer_elems:
            with torch.cuda.device(dst_dev), torch.cuda.stream(peer_streams[idx]):
                dst[idx][:peer_elems].copy_(src[idx][:peer_elems], non_blocking=True)

        if host_elems:
            with torch.cuda.device(src_dev), torch.cuda.stream(host_streams[idx]):
                host[idx][:host_elems].copy_(src[idx][peer_elems:], non_blocking=True)
                event = torch.cuda.Event()
                event.record(host_streams[idx])
                events.append(event)

            with torch.cuda.device(dst_dev), torch.cuda.stream(host_streams[idx]):
                host_streams[idx].wait_event(event)
                dst[idx][peer_elems:].copy_(host[idx][:host_elems], non_blocking=True)

    synchronize(devices)
    seconds = time.perf_counter() - start

    max_error = 0.0
    for idx, dev in enumerate(devices):
        expected_dev = devices[idx]
        with torch.cuda.device(dev):
            err = (dst[idx] - float(expected_dev + 1)).abs().max().item()
            max_error = max(max_error, err)

    return seconds, max_error


def main() -> None:
    args = parse_args()
    devices = [int(item) for item in args.devices.split(",") if item.strip()]
    alphas = [float(item) for item in args.alphas.split(",") if item.strip()]

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if len(devices) < 2:
        raise ValueError("At least two devices are required")

    elem_count = args.bytes_per_gpu // 4
    nbytes = elem_count * 4

    src: list[torch.Tensor] = []
    dst: list[torch.Tensor] = []
    host: list[torch.Tensor] = []
    peer_streams: list[torch.cuda.Stream] = []
    host_streams: list[torch.cuda.Stream] = []

    for idx, dev in enumerate(devices):
        with torch.cuda.device(dev):
            src.append(torch.full((elem_count,), float(dev + 1), device="cuda", dtype=torch.float32))
            dst.append(torch.empty((elem_count,), device="cuda", dtype=torch.float32))
            host.append(torch.empty((elem_count,), device="cpu", dtype=torch.float32, pin_memory=True))
            peer_streams.append(torch.cuda.Stream(device=devices[(idx + 1) % len(devices)]))
            host_streams.append(torch.cuda.Stream(device=dev))

    synchronize(devices)

    print(f"devices={devices} bytes_per_gpu={nbytes} repeats={args.repeats}")
    print("alpha,seconds,payload_GBps,max_error")

    for alpha in alphas:
        timings = []
        max_error = 0.0
        for _ in range(args.repeats):
            seconds, err = time_once(
                devices,
                nbytes,
                alpha,
                src,
                dst,
                host,
                peer_streams,
                host_streams,
            )
            timings.append(seconds)
            max_error = max(max_error, err)

        best = min(timings)
        payload_gbps = (nbytes * len(devices) * 2) / best / 1e9
        print(f"{alpha:.3f},{best:.6f},{payload_gbps:.3f},{max_error:.1f}")


if __name__ == "__main__":
    main()
