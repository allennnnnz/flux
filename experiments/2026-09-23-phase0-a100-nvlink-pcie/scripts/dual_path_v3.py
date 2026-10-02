#!/usr/bin/env python3
"""Dual-path benchmark: NVLink all-peer ingress plus host-staging ring.

Replaces dual_path_v2.py, whose NVLink path measured 38.4 GB/s per GPU against a
demonstrated single-pair peer-copy capability of ~270 GB/s.

Root cause of the v2 defect: PyTorch executes a cross-device copy on the SOURCE
device's current stream (aten/src/ATen/native/cuda/Copy.cu), not the destination's.
v2 wrapped each copy in the destination device's stream and never set any stream on
the source, so all 7 outgoing copies from each GPU serialized on that GPU's default
stream. Measured directly: destination-stream placement 38.1 GB/s/GPU vs
source-stream placement 103.7 GB/s/GPU at 256 MiB, and 210.9 GB/s/GPU at 1024 MiB.

The transfer is also per-copy-overhead bound below ~1 GiB per GPU per iteration.
Stream count is irrelevant (1 stream and 7 streams measure identically); only the
size of each individual peer copy matters:
    64 MiB/GPU  ->  9.1 MiB per peer copy ->  34.7 GB/s/GPU
   256 MiB/GPU  -> 36.6 MiB per peer copy -> 129.6 GB/s/GPU
  1024 MiB/GPU  -> 146.3 MiB per peer copy -> 210.9 GB/s/GPU
Hence the 1024 MiB default; smaller sizes understate NVLink badly.

Changes from v2:
  * Peer copies are issued from the source side, each on its own source stream.
  * No synchronization inside the timed region. All `copies` iterations are issued
    back to back; the only sync is at the end of the whole measured interval.
  * Every work stream is chained to a per-device timing event, so the per-device
    CUDA-event elapsed time covers exactly that device's share of the interval.
    Note this times each device's EGRESS; in symmetric all-to-all egress bytes
    equal ingress bytes, so the per-GPU figure is unchanged either way.
  * Host staging is double buffered with explicit cross-stream events, so removing
    the per-copy sync does not introduce a write-after-read race on the pinned buffer.
  * Peer access is asserted up front rather than assumed.
  * Every result is reported per-GPU and as an 8-GPU aggregate.

Payload accounting: each GPU *receives* `size_mib` per copy iteration, split as
(1-alpha) over 7 NVLink peers and alpha through one host-staged ring hop.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
import time

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--devices", default="0,1,2,3,4,5,6,7")
    p.add_argument("--size-mib", type=int, default=1024)
    p.add_argument("--copies", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--alphas", default="0,0.01,0.02,0.03,0.05,0.08,0.1")
    p.add_argument(
        "--stage-ring",
        default="",
        help="explicit staging ring order, e.g. 0,2,4,6,1,3,5,7 to avoid same-switch hops. "
        "Default is the device order, which on this box pairs 0->1, 2->3, ... on shared "
        "PCIe switch uplinks and roughly halves staging bandwidth.",
    )
    p.add_argument("--out", default="")
    return p.parse_args()


def stdev(v: list[float]) -> float:
    return statistics.stdev(v) if len(v) > 1 else 0.0


def assert_peer_access(devices: list[int]) -> None:
    missing = [
        (i, j)
        for i in devices
        for j in devices
        if i != j and not torch.cuda.can_device_access_peer(i, j)
    ]
    if missing:
        raise RuntimeError(f"peer access unavailable for pairs: {missing}")


class Bench:
    def __init__(self, devices: list[int], size_mib: int, stage_ring: list[int] | None = None):
        self.devices = devices
        self.stage_ring = stage_ring or list(devices)
        self.n = len(devices)
        self.peers = self.n - 1
        self.size_bytes = size_mib * 1024 * 1024
        self.elems = self.size_bytes // 4

        # peer_order[s] = destinations s pushes to; region[(d, s)] = which of d's
        # 7 receive slots holds s's contribution, so slots stay disjoint per dst.
        self.peer_order = {d: [x for x in devices if x != d] for d in devices}
        self.region = {
            (d, s): self.peer_order[d].index(s) for d in devices for s in devices if s != d
        }

        self.src, self.dst, self.host = {}, {}, {}
        self.nv_streams, self.d2h_stream, self.h2d_stream, self.timing_stream = {}, {}, {}, {}
        self.start_ev, self.end_ev = {}, {}
        # double-buffered staging handshake events
        self.d2h_done, self.h2d_done = {}, {}

        for dev in devices:
            with torch.cuda.device(dev):
                self.src[dev] = torch.full(
                    (self.elems,), float(dev + 1), dtype=torch.float32, device=f"cuda:{dev}"
                )
                self.dst[dev] = torch.empty(
                    (self.elems,), dtype=torch.float32, device=f"cuda:{dev}"
                )
                # two pinned buffers per source => double buffering
                self.host[dev] = [
                    torch.empty((self.elems,), dtype=torch.float32, pin_memory=True)
                    for _ in range(2)
                ]
                self.nv_streams[dev] = [
                    torch.cuda.Stream(device=dev) for _ in range(self.peers)
                ]
                self.d2h_stream[dev] = torch.cuda.Stream(device=dev)
                self.h2d_stream[dev] = torch.cuda.Stream(device=dev)
                self.timing_stream[dev] = torch.cuda.Stream(device=dev)
                self.start_ev[dev] = torch.cuda.Event(enable_timing=True)
                self.end_ev[dev] = torch.cuda.Event(enable_timing=True)
                self.d2h_done[dev] = [torch.cuda.Event() for _ in range(2)]
                self.h2d_done[dev] = [torch.cuda.Event() for _ in range(2)]
        self.sync_all()

    def sync_all(self) -> None:
        for dev in self.devices:
            with torch.cuda.device(dev):
                torch.cuda.synchronize()

    def _work_streams(self, dev: int, staging: bool) -> list[torch.cuda.Stream]:
        s = list(self.nv_streams[dev])
        if staging:
            s += [self.d2h_stream[dev], self.h2d_stream[dev]]
        return s

    def _issue_copy(self, c: int, per_peer: int, staging_elems: int, nv_elems: int) -> None:
        """Issue one copy iteration across all GPUs. No synchronization."""
        if per_peer:
            # Placement matters: PyTorch runs a cross-device copy on the SOURCE
            # device's current stream (aten/src/ATen/native/cuda/Copy.cu). v2 set
            # the destination device's stream, so every source's 7 outgoing copies
            # serialized on that source's untouched default stream -- 38 GB/s/GPU
            # instead of 211. Issue from the source side.
            for src_dev in self.devices:
                for i, dst_dev in enumerate(self.peer_order[src_dev]):
                    off = self.region[(dst_dev, src_dev)] * per_peer
                    st = self.nv_streams[src_dev][i]
                    with torch.cuda.device(src_dev), torch.cuda.stream(st):
                        self.dst[dst_dev][off : off + per_peer].copy_(
                            self.src[src_dev][off : off + per_peer], non_blocking=True
                        )

        if staging_elems:
            b = c % 2
            off = self.elems - staging_elems
            for i, src_dev in enumerate(self.stage_ring):
                dst_dev = self.stage_ring[(i + 1) % len(self.stage_ring)]
                d2h = self.d2h_stream[src_dev]
                h2d = self.h2d_stream[dst_dev]
                with torch.cuda.device(src_dev), torch.cuda.stream(d2h):
                    # do not overwrite a pinned buffer the previous H2D may still read
                    if c >= 2:
                        d2h.wait_event(self.h2d_done[src_dev][b])
                    self.host[src_dev][b][:staging_elems].copy_(
                        self.src[src_dev][off:], non_blocking=True
                    )
                    self.d2h_done[src_dev][b].record(d2h)
                with torch.cuda.device(dst_dev), torch.cuda.stream(h2d):
                    h2d.wait_event(self.d2h_done[src_dev][b])
                    self.dst[dst_dev][off:].copy_(
                        self.host[src_dev][b][:staging_elems], non_blocking=True
                    )
                    self.h2d_done[src_dev][b].record(h2d)

    def run_interval(self, alpha: float, copies: int, timed: bool) -> tuple[float, dict[int, float]]:
        staging_elems = int(math.floor(self.elems * alpha))
        staging_elems -= staging_elems % 4
        nv_elems = self.elems - staging_elems
        per_peer = (nv_elems // self.peers) if self.peers else 0
        per_peer -= per_peer % 4
        staging = staging_elems > 0

        self.sync_all()

        if timed:
            for dev in self.devices:
                with torch.cuda.device(dev):
                    self.start_ev[dev].record(self.timing_stream[dev])
                    for st in self._work_streams(dev, staging):
                        st.wait_event(self.start_ev[dev])

        wall0 = time.perf_counter()
        for c in range(copies):
            self._issue_copy(c, per_peer, staging_elems, nv_elems)

        if timed:
            for dev in self.devices:
                with torch.cuda.device(dev):
                    for st in self._work_streams(dev, staging):
                        self.timing_stream[dev].wait_stream(st)
                    self.end_ev[dev].record(self.timing_stream[dev])

        self.sync_all()
        wall = time.perf_counter() - wall0

        per_dev = {}
        if timed:
            for dev in self.devices:
                per_dev[dev] = self.start_ev[dev].elapsed_time(self.end_ev[dev]) / 1e3
        return wall, per_dev


def main() -> None:
    args = parse_args()
    devices = [int(x) for x in args.devices.split(",") if x.strip()]
    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
    if len(devices) < 2:
        raise ValueError("at least two devices required")
    assert_peer_access(devices)

    stage_ring = [int(x) for x in args.stage_ring.split(",") if x.strip()] or None
    if stage_ring and sorted(stage_ring) != sorted(devices):
        raise ValueError("--stage-ring must be a permutation of --devices")
    b = Bench(devices, args.size_mib, stage_ring)
    n = len(devices)
    # each GPU receives size_bytes per copy iteration
    payload_per_gpu = b.size_bytes * args.copies
    payload_total = payload_per_gpu * n

    print(
        f"meta,devices={devices},peers_per_gpu={b.peers},size_mib={args.size_mib},"
        f"copies={args.copies},warmup={args.warmup},repeats={args.repeats},"
        f"stage_ring={b.stage_ring}",
        file=sys.stderr,
    )
    hdr = (
        f"{'alpha':>6} {'wall_s':>10} {'agg_GBps':>10} {'perGPU_GBps':>12} "
        f"{'cuda_perGPU_GBps':>17} {'nv_perGPU':>10} {'stg_perGPU':>11} {'stdev_GBps':>11}"
    )
    print(hdr, file=sys.stderr)
    print("-" * len(hdr), file=sys.stderr)

    rows = []
    for alpha in alphas:
        for _ in range(args.warmup):
            b.run_interval(alpha, args.copies, timed=False)

        walls, cuda_meds = [], []
        for _ in range(args.repeats):
            wall, per_dev = b.run_interval(alpha, args.copies, timed=True)
            walls.append(wall)
            cuda_meds.append(statistics.median(per_dev.values()))

        wall_med = statistics.median(walls)
        cuda_med = statistics.median(cuda_meds)
        agg = payload_total / wall_med / 1e9
        per_gpu = payload_per_gpu / wall_med / 1e9
        cuda_per_gpu = payload_per_gpu / cuda_med / 1e9
        agg_all = [payload_total / w / 1e9 for w in walls]

        row = {
            "alpha": alpha,
            "wall_median_s": wall_med,
            "aggregate_GBps": agg,
            "per_gpu_GBps": per_gpu,
            "cuda_per_gpu_GBps": cuda_per_gpu,
            "nvlink_per_gpu_GBps": per_gpu * (1.0 - alpha),
            "staging_per_gpu_GBps": per_gpu * alpha,
            "aggregate_stdev_GBps": stdev(agg_all),
        }
        rows.append(row)
        print(
            f"{alpha:>6.3f} {wall_med:>10.6f} {agg:>10.2f} {per_gpu:>12.2f} "
            f"{cuda_per_gpu:>17.2f} {row['nvlink_per_gpu_GBps']:>10.2f} "
            f"{row['staging_per_gpu_GBps']:>11.2f} {row['aggregate_stdev_GBps']:>11.3f}",
            file=sys.stderr,
        )

    base = next((r for r in rows if r["alpha"] == 0.0), None)
    if base:
        print(file=sys.stderr)
        print(
            f"{'alpha':>6} {'per-GPU GB/s':>13} {'vs alpha=0':>11}",
            file=sys.stderr,
        )
        for r in rows:
            d = (r["per_gpu_GBps"] / base["per_gpu_GBps"] - 1.0) * 100.0
            print(
                f"{r['alpha']:>6.3f} {r['per_gpu_GBps']:>13.2f} {d:>+10.2f}%",
                file=sys.stderr,
            )

    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"\nwrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
