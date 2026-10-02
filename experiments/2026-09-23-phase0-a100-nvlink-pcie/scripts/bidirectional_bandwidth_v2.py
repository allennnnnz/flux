#!/usr/bin/env python3
"""Simultaneous H2D + D2H bandwidth, timed independently per direction.

Replaces bidirectional_bandwidth.py, which had two defects:

  1. It reported H2D and D2H as the same number by construction --
         h2d = bytes_per_direction / seconds / 1e9
         d2h = bytes_per_direction / seconds / 1e9
     both derived from the same wall clock. Every row in CDE_REPORT.md where
     "H2D" equals "D2H" exactly is this artifact, not a measurement.

  2. Its own CUDA-event columns contradicted that and were never used. For GPU0
     they read h2d 1687.96 ms against d2h 963.80 ms for the same payload -- the
     directions differ by 1.75x. Because D2H finishes in 964 ms while H2D runs
     1688 ms, the two directions are only truly concurrent for the first 964 ms;
     the remaining 724 ms is H2D alone. Neither reported number describes
     simultaneous bidirectional bandwidth.

This version records an event after every individual copy in each direction, all
offset from one common reference event per device, which gives:

  * each direction's own steady-state bandwidth,
  * the exact window in which both directions were in flight,
  * each direction's bandwidth restricted to that overlap window, which is the
    only number that legitimately describes simultaneous bidirectional transfer.

Separate GPU buffers are used per direction so the two do not contend on the same
device memory.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import multiprocessing as mp
import os
import statistics

GPU_NUMA_NODE = {0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 1}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--devices", default="0,1,2,3,4,5,6,7")
    p.add_argument("--size-mib", type=int, default=256)
    p.add_argument("--copies", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--numa-mode", choices=("none", "local", "remote"), default="local")
    p.add_argument("--direction", choices=("both", "h2d", "d2h"), default="both")
    p.add_argument("--out", default="")
    return p.parse_args()


def cpus_for_node(node: int) -> set[int]:
    text = open(f"/sys/devices/system/node/node{node}/cpulist", encoding="ascii").read().strip()
    cpus: set[int] = set()
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-", 1)
            cpus.update(range(int(a), int(b) + 1))
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
    if numa.numa_run_on_node(node) != 0:
        raise OSError(ctypes.get_errno(), f"numa_run_on_node({node}) failed")
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
    return local if mode == "local" else 1 - local


def worker(device, size_bytes, copies, warmup, repeats, numa_mode, direction, barrier, q):
    node = numa_node_for(device, numa_mode)
    if node is not None:
        bind_to_numa_node(node)

    import torch

    torch.cuda.set_device(device)
    n = size_bytes // 4
    host_in = torch.empty(n, dtype=torch.float32, pin_memory=True)
    host_out = torch.empty(n, dtype=torch.float32, pin_memory=True)
    # separate device buffers: the two directions must not contend on one allocation
    gpu_in = torch.empty(n, dtype=torch.float32, device=f"cuda:{device}")
    gpu_out = torch.empty(n, dtype=torch.float32, device=f"cuda:{device}")

    h2d_s = torch.cuda.Stream(device=device)
    d2h_s = torch.cuda.Stream(device=device)
    ref_s = torch.cuda.Stream(device=device)

    do_h2d = direction in ("both", "h2d")
    do_d2h = direction in ("both", "d2h")

    def run(record: bool):
        ref = torch.cuda.Event(enable_timing=True)
        h2d_ev = [torch.cuda.Event(enable_timing=True) for _ in range(copies + 1)] if record else None
        d2h_ev = [torch.cuda.Event(enable_timing=True) for _ in range(copies + 1)] if record else None

        torch.cuda.synchronize(device)
        with torch.cuda.stream(ref_s):
            ref.record(ref_s)
        h2d_s.wait_event(ref)
        d2h_s.wait_event(ref)

        if do_h2d:
            with torch.cuda.stream(h2d_s):
                if record:
                    h2d_ev[0].record(h2d_s)
                for i in range(copies):
                    gpu_in.copy_(host_in, non_blocking=True)
                    if record:
                        h2d_ev[i + 1].record(h2d_s)
        if do_d2h:
            with torch.cuda.stream(d2h_s):
                if record:
                    d2h_ev[0].record(d2h_s)
                for i in range(copies):
                    host_out.copy_(gpu_out, non_blocking=True)
                    if record:
                        d2h_ev[i + 1].record(d2h_s)

        h2d_s.synchronize()
        d2h_s.synchronize()
        if not record:
            return None
        # every timestamp expressed as ms after the common reference event
        h2d_t = [ref.elapsed_time(e) for e in h2d_ev] if do_h2d else []
        d2h_t = [ref.elapsed_time(e) for e in d2h_ev] if do_d2h else []
        return h2d_t, d2h_t

    for _ in range(warmup):
        run(False)

    rows = []
    for r in range(repeats):
        barrier.wait()
        h2d_t, d2h_t = run(True)
        barrier.wait()
        rows.append((r, device, node, h2d_t, d2h_t))
    q.put(rows)


def bytes_in_window(ts: list[float], size_bytes: int, lo: float, hi: float) -> float:
    """Bytes completed strictly inside [lo, hi] -- one copy per consecutive pair."""
    if not ts:
        return 0.0
    total = 0.0
    for i in range(len(ts) - 1):
        a, b = ts[i], ts[i + 1]
        if b <= lo or a >= hi:
            continue
        frac = (min(b, hi) - max(a, lo)) / (b - a)
        total += size_bytes * max(0.0, frac)
    return total


def stdev(v):
    return statistics.stdev(v) if len(v) > 1 else 0.0


def main() -> None:
    args = parse_args()
    devices = [int(x) for x in args.devices.split(",") if x.strip()]
    size_bytes = args.size_mib * 1024 * 1024

    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(len(devices) + 1)
    q = ctx.Queue()
    procs = [
        ctx.Process(
            target=worker,
            args=(d, size_bytes, args.copies, args.warmup, args.repeats,
                  args.numa_mode, args.direction, barrier, q),
        )
        for d in devices
    ]
    print(
        f"meta,devices={devices},size_mib={args.size_mib},copies={args.copies},"
        f"warmup={args.warmup},repeats={args.repeats},numa_mode={args.numa_mode},"
        f"direction={args.direction}"
    )
    for p in procs:
        p.start()
    for _ in range(args.repeats):
        barrier.wait()
        barrier.wait()

    rows = []
    for _ in procs:
        rows.extend(q.get())
    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"worker exited {p.exitcode}")

    # per device, per repeat
    h2d_own, d2h_own, h2d_ov, d2h_ov, ov_frac = [], [], [], [], []
    for r, dev, node, h2d_t, d2h_t in rows:
        if h2d_t:
            own = (h2d_t[-1] - h2d_t[0]) / 1e3
            h2d_own.append(size_bytes * args.copies / own / 1e9)
        if d2h_t:
            own = (d2h_t[-1] - d2h_t[0]) / 1e3
            d2h_own.append(size_bytes * args.copies / own / 1e9)
        if h2d_t and d2h_t:
            lo = max(h2d_t[0], d2h_t[0])
            hi = min(h2d_t[-1], d2h_t[-1])
            if hi > lo:
                w = (hi - lo) / 1e3
                h2d_ov.append(bytes_in_window(h2d_t, size_bytes, lo, hi) / w / 1e9)
                d2h_ov.append(bytes_in_window(d2h_t, size_bytes, lo, hi) / w / 1e9)
                span = max(h2d_t[-1], d2h_t[-1]) - min(h2d_t[0], d2h_t[0])
                ov_frac.append((hi - lo) / span)

    def s(v):
        return f"{statistics.median(v):.3f}" if v else "n/a"

    n = len(devices)
    print()
    print(f"{'metric':<42} {'per-GPU GB/s':>13} {'aggregate GB/s':>15}")
    print("-" * 72)
    if h2d_own:
        m = statistics.median(h2d_own)
        print(f"{'H2D, own event window':<42} {m:>13.3f} {m*n:>15.3f}")
    if d2h_own:
        m = statistics.median(d2h_own)
        print(f"{'D2H, own event window':<42} {m:>13.3f} {m*n:>15.3f}")
    if h2d_ov:
        m = statistics.median(h2d_ov)
        print(f"{'H2D, restricted to overlap window':<42} {m:>13.3f} {m*n:>15.3f}")
    if d2h_ov:
        m = statistics.median(d2h_ov)
        print(f"{'D2H, restricted to overlap window':<42} {m:>13.3f} {m*n:>15.3f}")
    if h2d_ov and d2h_ov:
        tot = statistics.median(h2d_ov) + statistics.median(d2h_ov)
        print(f"{'combined during overlap':<42} {tot:>13.3f} {tot*n:>15.3f}")
        print()
        print(f"overlap window as fraction of total span: {statistics.median(ov_frac)*100:.1f}%")
    print()
    print(f"stdev  h2d_own={stdev(h2d_own):.3f}  d2h_own={stdev(d2h_own):.3f}  "
          f"h2d_ov={stdev(h2d_ov):.3f}  d2h_ov={stdev(d2h_ov):.3f}")

    if args.out:
        import csv
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["metric", "per_gpu_GBps", "aggregate_GBps", "stdev_GBps"])
            for name, vals in [
                ("h2d_own", h2d_own), ("d2h_own", d2h_own),
                ("h2d_overlap", h2d_ov), ("d2h_overlap", d2h_ov),
            ]:
                if vals:
                    m = statistics.median(vals)
                    w.writerow([name, f"{m:.3f}", f"{m*n:.3f}", f"{stdev(vals):.3f}"])
            if ov_frac:
                w.writerow(["overlap_fraction", f"{statistics.median(ov_frac):.4f}", "", ""])
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
