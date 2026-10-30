"""T2 helper (2026-10-30): continuous GPU->host (d2h) or host->GPU (h2d) copies on the given GPUs for a fixed time.
Prints each GPU's rate over the middle of the run (the RDMA test runs inside that window).
    python t2_copy_bg_v1.py --gpus 0,1 --mode d2h --seconds 9 --numa 0
"""
import argparse
import os
import threading
import time

import torch

MiB = 1 << 20
NUMA_CPUS = {0: list(range(0, 32)) + list(range(64, 96)), 1: list(range(32, 64)) + list(range(96, 128))}


def worker(g, mode, seconds, t_start, res):
    torch.cuda.set_device(g)
    host = torch.empty(256 * MiB, dtype=torch.uint8, pin_memory=True)
    host.fill_(1)
    dev = torch.empty(256 * MiB, dtype=torch.uint8, device=g)
    s = torch.cuda.Stream()
    n = 0
    marks = []
    while time.time() < t_start:
        time.sleep(0.001)
    t_end = t_start + seconds
    with torch.cuda.stream(s):
        while time.time() < t_end:
            for k in range(16):
                sl = slice(k * 16 * MiB, (k + 1) * 16 * MiB)
                if mode == "d2h":
                    host[sl].copy_(dev[sl], non_blocking=True)
                else:
                    dev[sl].copy_(host[sl], non_blocking=True)
            s.synchronize()
            n += 256 * MiB
            marks.append((time.time(), n))
    # rate over the middle (25%..75% of the run)
    a, b = t_start + 0.25 * seconds, t_start + 0.75 * seconds
    mid = [(t, x) for t, x in marks if a <= t <= b]
    res[g] = (mid[-1][1] - mid[0][1]) / (mid[-1][0] - mid[0][0]) / 1e9 if len(mid) >= 2 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--mode", default="d2h")
    ap.add_argument("--seconds", type=float, default=9.0)
    ap.add_argument("--numa", type=int, default=0)
    a = ap.parse_args()
    os.sched_setaffinity(0, NUMA_CPUS[a.numa])
    gpus = [int(x) for x in a.gpus.split(",")]
    for g in gpus:  # create contexts before the timed start
        torch.cuda.set_device(g)
        torch.zeros(1, device=g)
    t_start = time.time() + 1.0
    res = {}
    th = [threading.Thread(target=worker, args=(g, a.mode, a.seconds, t_start, res)) for g in gpus]
    for t in th:
        t.start()
    for t in th:
        t.join()
    print(" ".join(f"gpu{g}_{a.mode}_GBps={res[g]:.2f}" for g in gpus), f"total={sum(res.values()):.2f}", flush=True)


if __name__ == "__main__":
    main()
