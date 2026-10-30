"""T1 (2026-10-30, topic search): do background PCIe copies slow TRAINING-size NCCL collectives?

X1d measured decode-size (KB) all-reduce inside CUDA graphs. Training (FSDP / ZeRO-Offload / activation
offload) runs MB-size reduce-scatter / all-gather / all-reduce eagerly while host<->GPU copies are in flight.
This probe answers: at 4 / 32 / 256 MB, how much slower is each collective while all 8 GPUs copy to / from
pinned host memory on a side stream, and how much slower is the copy?

Per round (>= 30 rounds), every (op, size, mode) window runs once in a shuffled order (same order on all
ranks):  mode = none | h2d | d2h;  plus two bg-only windows (h2d, d2h) for the copy's own rate.
A window: barrier -> event t0 on the main stream -> side stream waits t0 and enqueues enough 16 MiB copies
to outlast the collectives (sized from the 'none' calibration) -> main stream: 2 warm-up + ITERS collectives
between events -> bg_end event. A window counts only if the background outlasts the collectives (covered).
Output: per-rank per-window times -> rank-max per-call time; ratio to the same round's 'none' window.
Run under torchrun (8 ranks) inside common/measure/exclusive_guard_v2.py.
"""
import argparse
import json
import os
import random
import statistics
import sys
import time

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../common/measure"))
from clock_logger import ClockLogger  # noqa: E402

MiB = 1 << 20
CHUNK = 16 * MiB
NUMA_CPUS = {0: list(range(0, 32)) + list(range(64, 96)), 1: list(range(32, 64)) + list(range(96, 128))}


def busbw(op, nbytes, t, n):
    f = 2 * (n - 1) / n if op == "all_reduce" else (n - 1) / n
    return f * nbytes / t / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--sizes_mb", default="4,32,256")
    ap.add_argument("--ops", default="all_reduce,reduce_scatter,all_gather")
    ap.add_argument("--bg_ranks", default="all", help="ranks that run the background copy, e.g. 0 or 0,1 (default all)")
    args = ap.parse_args()

    dist.init_process_group("nccl")
    rank, n = dist.get_rank(), dist.get_world_size()
    lr = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(lr)
    os.sched_setaffinity(0, NUMA_CPUS[0 if lr < 4 else 1])  # pinned pages land on the GPU's NUMA node (first touch)
    dev = torch.device("cuda", lr)
    sizes = [int(s) * MiB for s in args.sizes_mb.split(",")]
    ops = args.ops.split(",")
    maxs = max(sizes)
    bg_on = args.bg_ranks == "all" or str(rank) in args.bg_ranks.split(",")

    a = torch.randn(maxs // 2, dtype=torch.bfloat16, device=dev)
    b = torch.empty_like(a)
    host = torch.empty(256 * MiB, dtype=torch.uint8, pin_memory=True)
    host.fill_(1)
    gbuf = torch.empty(256 * MiB, dtype=torch.uint8, device=dev)
    side = torch.cuda.Stream()

    def coll(op, nb):
        e = nb // 2
        if op == "all_reduce":
            dist.all_reduce(a[:e])
        elif op == "reduce_scatter":
            dist.reduce_scatter_tensor(b[: e // n], a[:e])
        else:
            dist.all_gather_into_tensor(b[:e], a[: e // n])

    def bg_enqueue(mode, nchunks, t0):
        side.wait_event(t0)
        with torch.cuda.stream(side):
            s = torch.cuda.Event(enable_timing=True)
            s.record()
            nh = host.numel() // CHUNK
            evs = []
            for i in range(nchunks):
                k = i % nh
                if mode == "h2d":
                    gbuf[k * CHUNK:(k + 1) * CHUNK].copy_(host[k * CHUNK:(k + 1) * CHUNK], non_blocking=True)
                else:
                    host[k * CHUNK:(k + 1) * CHUNK].copy_(gbuf[k * CHUNK:(k + 1) * CHUNK], non_blocking=True)
                ev = torch.cuda.Event(enable_timing=True)
                ev.record()
                evs.append(ev)
        return s, evs

    def barrier():
        torch.cuda.synchronize()
        dist.barrier()
        torch.cuda.synchronize()

    def window(op, nb, mode, nchunks):
        """returns (per-call ms, bg GB/s, covered, coll_total_ms)"""
        barrier()
        t0 = torch.cuda.Event(enable_timing=True)
        t0.record()
        bs = bevs = None
        if mode != "none" and bg_on:
            bs, bevs = bg_enqueue(mode, nchunks, t0)
        if op is not None:
            for _ in range(2):
                coll(op, nb)
            cs = torch.cuda.Event(enable_timing=True)
            ce = torch.cuda.Event(enable_timing=True)
            cs.record()
            for _ in range(args.iters):
                coll(op, nb)
            ce.record()
        torch.cuda.synchronize()
        per = cs.elapsed_time(ce) / args.iters if op is not None else None
        bg = cov = None
        if mode != "none" and not bg_on:
            cov = True  # this rank copies nothing; coverage is judged on the copying ranks
        elif mode != "none":
            # copy rate while the timed collectives run: chunks finished between cs and ce
            # (bg-only windows: whole run)
            if op is None:
                # middle half of the chunks only: all 8 GPUs are still copying (whole-run average is inflated
                # by GPUs that finish early and stop competing)
                ends = [bs.elapsed_time(ev) for ev in bevs]
                q1, q3 = len(ends) // 4, (3 * len(ends)) // 4
                bg = (q3 - q1) * CHUNK / ((ends[q3] - ends[q1]) / 1e3) / 1e9
                cov = True
            else:
                t_cs, t_ce = t0.elapsed_time(cs), t0.elapsed_time(ce)
                ends = [t0.elapsed_time(ev) for ev in bevs]
                inside = [x for x in ends if t_cs <= x <= t_ce]
                bg = (len(inside) - 1) * CHUNK / ((inside[-1] - inside[0]) / 1e3) / 1e9 if len(inside) >= 3 else None
                cov = ends[-1] >= t_ce
        return per, bg, cov

    # calibration: 'none' per-call time -> number of bg chunks to outlast 2+ITERS calls at <= 12 GB/s
    nchunks = {}
    for op in ops:
        for nb in sizes:
            ts = [window(op, nb, "none", 0)[0] for _ in range(3)]
            tot_s = max(ts) * (args.iters + 2) / 1e3
            loc = torch.tensor([tot_s], device=dev)
            dist.all_reduce(loc, op=dist.ReduceOp.MAX)
            need = int(3.0 * loc.item() * 12e9 / CHUNK) + 4
            nchunks[(op, nb)] = need
    bg_only_chunks = 64  # 1 GiB

    combos = [(op, nb, m) for op in ops for nb in sizes for m in ("none", "h2d", "d2h")]
    combos += [(None, 0, "h2d"), (None, 0, "d2h")]
    rec = []
    with ClockLogger(lr, period_s=0.25) as clk:
        for r in range(args.rounds):
            order = combos[:]
            random.Random(1000 + r).shuffle(order)
            for op, nb, m in order:
                w0 = time.perf_counter()
                nc = nchunks[(op, nb)] if op is not None else bg_only_chunks
                per, bg, cov = window(op, nb, m, nc)
                rec.append({"round": r, "op": op, "bytes": nb, "mode": m, "per_ms": per, "bg_GBps": bg,
                            "covered": cov, "w": (w0, time.perf_counter())})
        for x in rec:
            x["clock"] = clk.min_clock_between(*x["w"])
            del x["w"]

    allrec = [None] * n
    dist.all_gather_object(allrec, rec)
    if rank == 0:
        os.makedirs(args.out, exist_ok=True)
        meta = {"world": n, "bg_ranks": args.bg_ranks, "iters": args.iters, "rounds": args.rounds, "chunk": CHUNK, "nchunks": {f"{k[0]}|{k[1]}": v for k, v in nchunks.items()},
                "env": {k: v for k, v in os.environ.items() if k.startswith(("NCCL", "CUDA"))},
                "torch": torch.__version__, "nccl": ".".join(map(str, torch.cuda.nccl.version()))}
        with open(os.path.join(args.out, "t1_raw.json"), "w") as f:
            json.dump({"meta": meta, "ranks": allrec}, f)
        global BG_RANKS
        BG_RANKS = args.bg_ranks
        summarize(allrec, ops, sizes, n, args.out)
    dist.barrier()
    dist.destroy_process_group()


BG_RANKS = "all"


def summarize(allrec, ops, sizes, n, out):
    """rank-max per-call time per window; paired ratio to the same round's 'none'; drop down-clocked windows."""
    nr = len(allrec)
    keyed = {}
    for rk, recs in enumerate(allrec):
        for x in recs:
            keyed.setdefault((x["round"], x["op"], x["bytes"], x["mode"]), []).append(x)
    clocks = [x["clock"] for recs in allrec for x in recs if x["clock"]]
    modal = statistics.mode(clocks) if clocks else None
    lines = [f"T1 summary (rank-max per-call time; ratio = same-round paired; modal SM clock {modal} MHz; windows with any rank < 95% modal dropped)"]
    rows = []
    for op in ops:
        for nb in sizes:
            base = {}
            per = {"h2d": [], "d2h": []}
            bgr = {"h2d": [], "d2h": []}
            uncovered = 0
            dropped = 0
            for (r, o, b, m), xs in keyed.items():
                if o != op or b != nb:
                    continue
                if len(xs) != nr:
                    continue
                if modal and any(x["clock"] and x["clock"] < 0.95 * modal for x in xs):
                    dropped += 1
                    continue
                t = max(x["per_ms"] for x in xs)
                if m == "none":
                    base[r] = t
                else:
                    if not all(x["covered"] for x in xs):
                        uncovered += 1
                        continue
                    per[m].append((r, t))
                    bv = [x["bg_GBps"] for x in xs if x["bg_GBps"]]
                    if bv:
                        bgr[m].append(statistics.median(bv))
            tb = statistics.median(base.values())
            line = f"{op:15s} {nb // MiB:4d} MiB  none {tb:8.3f} ms ({busbw(op, nb, tb / 1e3, n):6.1f} GB/s busbw)"
            for m in ("h2d", "d2h"):
                ratios = sorted(t / base[r] for r, t in per[m] if r in base)
                if not ratios:
                    line += f" | {m}: n/a"
                    continue
                med = statistics.median(ratios)
                p10, p90 = ratios[len(ratios) // 10], ratios[(9 * len(ratios)) // 10]
                bgm = statistics.median(bgr[m]) if bgr[m] else float("nan")
                line += f" | {m} x{med:.3f} [p10 {p10:.3f} p90 {p90:.3f}] n={len(ratios)} copy {bgm:.1f} GB/s/GPU"
                rows.append({"op": op, "MiB": nb // MiB, "mode": m, "none_ms": tb, "ratio_med": med, "p10": p10, "p90": p90,
                             "n": len(ratios), "copy_GBps_per_gpu_during_med": bgm})
            line += f"  (uncovered {uncovered}, clock-dropped {dropped})"
            lines.append(line)
    for m in ("h2d", "d2h"):
        xs = [statistics.median([x["bg_GBps"] for x in v if x["bg_GBps"]]) for (r, o, b, mm), v in keyed.items() if o is None and mm == m and len(v) == nr]
        if xs:
            lines.append(f"bg-only {m}: per-GPU rate (middle half of chunks) median {statistics.median(xs):.2f} GB/s (copying ranks: {BG_RANKS}; n={len(xs)})")
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(out, "t1_summary.txt"), "w") as f:
        f.write(txt + "\n")
    with open(os.path.join(out, "t1_rows.json"), "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
