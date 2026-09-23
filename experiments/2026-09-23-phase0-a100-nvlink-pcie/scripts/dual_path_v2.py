#!/usr/bin/env python3
"""Synthetic dual-path benchmark: NVLink all-peer ingress plus host staging ring."""

from __future__ import annotations

import argparse
import math
import statistics
import time

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--size-mib", type=int, default=1024)
    parser.add_argument("--copies", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--alphas", default="0,0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.50,1.0")
    return parser.parse_args()


def sync(devices: list[int]) -> None:
    for dev in devices:
        with torch.cuda.device(dev):
            torch.cuda.synchronize()


def make_streams(devices: list[int], count: int) -> dict[int, list[torch.cuda.Stream]]:
    streams: dict[int, list[torch.cuda.Stream]] = {}
    for dev in devices:
        with torch.cuda.device(dev):
            streams[dev] = [torch.cuda.Stream(device=dev) for _ in range(count)]
    return streams


def stdev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def main() -> None:
    args = parse_args()
    devices = [int(item) for item in args.devices.split(",") if item.strip()]
    alphas = [float(item) for item in args.alphas.split(",") if item.strip()]
    if len(devices) < 2:
        raise ValueError("At least two devices are required")
    if args.size_mib < 1024 and args.copies < 20:
        raise ValueError("Use either --size-mib >= 1024 or --copies >= 20")

    size_bytes = args.size_mib * 1024 * 1024
    elems = size_bytes // 4
    peer_count = len(devices) - 1

    src = {}
    dst = {}
    host = {}
    for dev in devices:
        with torch.cuda.device(dev):
            src[dev] = torch.full((elems,), float(dev + 1), dtype=torch.float32, device=f"cuda:{dev}")
            dst[dev] = torch.empty((elems,), dtype=torch.float32, device=f"cuda:{dev}")
            host[dev] = torch.empty((elems,), dtype=torch.float32, pin_memory=True)

    nv_streams = make_streams(devices, peer_count)
    d2h_streams = make_streams(devices, 1)
    h2d_streams = make_streams(devices, 1)
    sync(devices)

    def run_once(alpha: float) -> float:
        staging_elems = int(math.floor(elems * alpha))
        staging_elems -= staging_elems % 4
        nv_elems = elems - staging_elems
        per_peer = nv_elems // peer_count if peer_count else 0
        per_peer -= per_peer % 4

        start = time.perf_counter()
        if per_peer:
            for dst_idx, dst_dev in enumerate(devices):
                stream_idx = 0
                for src_dev in devices:
                    if src_dev == dst_dev:
                        continue
                    off = stream_idx * per_peer
                    with torch.cuda.device(dst_dev), torch.cuda.stream(nv_streams[dst_dev][stream_idx]):
                        dst[dst_dev][off : off + per_peer].copy_(
                            src[src_dev][off : off + per_peer], non_blocking=True
                        )
                    stream_idx += 1

        if staging_elems:
            for idx, src_dev in enumerate(devices):
                dst_dev = devices[(idx + 1) % len(devices)]
                off = elems - staging_elems
                with torch.cuda.device(src_dev), torch.cuda.stream(d2h_streams[src_dev][0]):
                    host[src_dev][:staging_elems].copy_(src[src_dev][off:], non_blocking=True)
                    ready = torch.cuda.Event()
                    ready.record(d2h_streams[src_dev][0])
                with torch.cuda.device(dst_dev), torch.cuda.stream(h2d_streams[dst_dev][0]):
                    h2d_streams[dst_dev][0].wait_event(ready)
                    dst[dst_dev][off:].copy_(host[src_dev][:staging_elems], non_blocking=True)

        sync(devices)
        return time.perf_counter() - start

    print(
        "meta,"
        f"devices={devices},peer_receivers_per_gpu={peer_count},size_mib={args.size_mib},"
        f"copies={args.copies},warmup={args.warmup},repeats={args.repeats}"
    )
    print(
        "alpha,median_seconds,stdev_seconds,payload_GBps,nvlink_payload_GBps,"
        "staging_payload_GBps,predicted_seconds,model_nvlink_GBps,model_staging_GBps"
    )

    b_nvlink = None
    b_staging = None
    measured: dict[float, float] = {}
    for alpha in alphas:
        for _ in range(args.warmup):
            for _copy in range(args.copies):
                run_once(alpha)
        timings = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            for _copy in range(args.copies):
                run_once(alpha)
            timings.append(time.perf_counter() - start)
        med = statistics.median(timings)
        measured[alpha] = med
        total_payload = len(devices) * size_bytes * args.copies
        nv_payload = total_payload * (1.0 - alpha)
        staging_payload = total_payload * alpha
        payload_gbps = total_payload / med / 1e9
        nv_gbps = nv_payload / med / 1e9 if nv_payload else 0.0
        staging_gbps = staging_payload / med / 1e9 if staging_payload else 0.0
        if alpha == 0:
            b_nvlink = total_payload / med
        if alpha == 1:
            b_staging = total_payload / med
        pred = 0.0
        if b_nvlink and b_staging:
            pred = max(nv_payload / b_nvlink if nv_payload else 0.0, staging_payload / b_staging if staging_payload else 0.0)
        print(
            f"{alpha:.3f},{med:.9f},{stdev(timings):.9f},{payload_gbps:.3f},"
            f"{nv_gbps:.3f},{staging_gbps:.3f},{pred:.9f},"
            f"{(b_nvlink / 1e9) if b_nvlink else 0.0:.3f},{(b_staging / 1e9) if b_staging else 0.0:.3f}"
        )


if __name__ == "__main__":
    main()
