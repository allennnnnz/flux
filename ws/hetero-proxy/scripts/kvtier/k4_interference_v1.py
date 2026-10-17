# WITHDRAWN 2026-10-17: hung inside capture_all (warm-ups on the graph communicator, results/kvtier/k4/
# interference_v1_hang_capture). Replaced by ws/hetero-proxy/scripts/kvtier/k4_interference_v2.py.
################################################################################
# hetero-proxy KV tier K4.1 (2026-10-17): does PCIe KV traffic slow the TP forward passes that run at the same time?
# Direction 1 (STATUS section 0): restore evicted KV in the BACKGROUND while the other users keep decoding, instead of
# K2's "everybody waits for the restore". Whether that needs Flux-style scheduling (copies only while NVLink is idle)
# depends on how much the copies slow the passes - results/e2e/c_diag measured 1.7x on 10-80 MiB NCCL all-reduces
# overlapped by PCIe copies (nsys), but a decode step's all-reduces are ~0.1-0.3 MiB and latency-bound.
# K2 engine (k2_engine_v1.py, CUDA graphs, own NCCL communicator), all 8 ranks. Cases, interleaved and shuffled per
# round (same order on every rank):
#   ops   dec8 / dec16 / dec32   one decode step, B rows at positions ~2000 (graph replay)
#         new2048                prefill of 2048 tokens from position 0 (a recompute-sized pass; 20 MiB all-reduces)
#         ext128                 128 new tokens after a 2000-token history (the new turn after a restore)
#   bg    none | h2d | d2h       --bg_mb MiB per GPU in --chunk_mb copies on a side stream, enqueued just before the
#                                op, sized to outlast it (4x for new2048; all 8 GPUs at once, as a real restore is)
# Per round and case: the op's CUDA-event time on every rank (rank-max) and the background's duration; the analysis
# keeps rounds where the background outlasted the op on every rank (it really overlapped the whole op).
# PRE-REGISTERED [推論]:
#   I1 decode step with h2d background: +0 .. +10% at every B (small all-reduces; K2 saw no slowdown from saves)
#   I2 new2048 with h2d background: +5 .. +30% (c_diag 1.7x on overlapped large all-reduces, ~20% of a prefill)
#   I3 d2h within 5 points of h2d for every op
#   I4 the background copies themselves run at 9-12 GB/s per GPU whatever the op (copy engine, not SMs)
# Usage: (repo root, exclusive guard, tmux) ws/hetero-proxy/scripts/kvtier/run_k2_v1.sh is for k2_serve; this one:
#   env -u CUDA_DEVICE_MAX_CONNECTIONS CUDA_MODULE_LOADING=EAGER pixi run --manifest-path pixi.toml \
#     ~/.venvs/hetero-proxy-hf/bin/python -m torch.distributed.run --nproc_per_node 8 \
#     ws/hetero-proxy/scripts/kvtier/k4_interference_v1.py --out <dir> [--rounds 200]
################################################################################
import argparse
import json
import os
import random
import statistics
import sys
import time

import torch
import torch.distributed as dist

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "e2e"))
sys.path.insert(0, os.path.join(HERE, ".."))
from qwen_tp_v1 import init  # noqa: E402
from qwen_tp_v5 import pin  # noqa: E402
from k2_engine_v1 import K2Engine  # noqa: E402
from kvtier_v2 import node_cpus, numa_pages  # noqa: E402
from hxb import cudamem as cm  # noqa: E402

MiB = 2 ** 20


def main():
    import faulthandler
    faulthandler.dump_traceback_later(int(os.environ.get("HXB_STACK_DUMP_S", "0")) or 10 ** 7, repeat=True, exit=False)
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--bg_mb", type=int, default=384)
    ap.add_argument("--chunk_mb", type=int, default=4)
    args = ap.parse_args()
    rank, world, dev = init()
    pin(rank)
    os.makedirs(args.out, exist_ok=True)
    m = K2Engine(rank, world, dev, 40, 4608, 32)
    t0 = time.time()
    m.capture_all(max_ext_t=128)
    # background buffers: pinned host memory on this GPU's NUMA node (first touch there), device scratch
    nb = args.bg_mb * MiB
    node = 0 if dev.index < 4 else 1
    old, nt = os.sched_getaffinity(0), torch.get_num_threads()
    os.sched_setaffinity(0, node_cpus(node))
    torch.set_num_threads(1)
    host = torch.zeros(nb, dtype=torch.uint8)
    os.sched_setaffinity(0, old)
    torch.set_num_threads(nt)
    cm.register(host.data_ptr(), nb)
    devbuf = torch.empty(nb, dtype=torch.uint8, device=dev)
    bgs = torch.cuda.Stream(dev)
    ch = args.chunk_mb * MiB
    numa = numa_pages(host.data_ptr())

    # fixed op inputs: decode rows in slots 0..B-1 at positions 2000 + j; ext / new in slot 39
    def prep(op):
        if op.startswith("dec"):
            B = int(op[3:])
            m.d_in_h[0, :B] = torch.arange(B)
            m.d_in_h[1, :B] = 2000 + torch.arange(B)
            m.d_in[:, :B].copy_(m.d_in_h[:, :B])
        elif op == "new2048":
            m.set_prefill([1000] * 2048, 0, 39)
        else:
            m.set_prefill([1000] * 128, 2000, 39)
        torch.cuda.synchronize()

    def replay(op):
        if op.startswith("dec"):
            m.gs[("dec", int(op[3:]))].replay()
        elif op == "new2048":
            m.gs[("new", 2048)].replay()
        else:
            m.gate.fill_(1)
            m.gs[("ext", 128, m.ext_window(2000, 128))].replay()

    ops = ["dec8", "dec16", "dec32", "new2048", "ext128"]
    bgk = ["none", "h2d", "d2h"]
    cases = [(o, b) for o in ops for b in bgk]
    raw = {f"{o}|{b}": [] for o, b in cases}
    bgms = {f"{o}|{b}": [] for o, b in cases if b != "none"}

    @torch.inference_mode()
    def one(op, b):
        prep(op)
        dist.barrier()
        torch.cuda.synchronize()
        cs = torch.cuda.current_stream()
        b0, b1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        if b != "none":
            bgs.wait_stream(cs)
            b0.record(bgs)
            total = nb * (4 if op == "new2048" else 1)  # outlast the ~0.1 s prefill too
            for o2 in range(0, total, ch):
                off = o2 % nb
                if b == "h2d":
                    cm.memcpy(bgs, devbuf.data_ptr() + off, host.data_ptr() + off, ch)
                else:
                    cm.memcpy(bgs, host.data_ptr() + off, devbuf.data_ptr() + off, ch)
            b1.record(bgs)
        e0.record(cs)
        replay(op)
        e1.record(cs)
        torch.cuda.synchronize()
        # did the background start before the op and outlast it? (both measured from b0)
        if b != "none":
            outlast = b0.elapsed_time(b1) >= b0.elapsed_time(e1)
            return e0.elapsed_time(e1), b0.elapsed_time(b1), outlast
        return e0.elapsed_time(e1), None, True

    for _ in range(3):
        for c in cases:
            one(*c)
    if rank == 0:
        print(f"[setup] {time.time() - t0:.0f} s, host NUMA pages {numa}", flush=True)
    rng = random.Random(1)
    outl = {k: [] for k in raw}
    t1 = time.time()
    for r in range(args.rounds):
        order = cases[:]
        rng.shuffle(order)
        for o, b in order:
            ms, bm, ok = one(o, b)
            raw[f"{o}|{b}"].append(ms)
            outl[f"{o}|{b}"].append(ok)
            if bm is not None:
                bgms[f"{o}|{b}"].append(bm)
        if rank == 0 and (r + 1) % 25 == 0:
            print(f"[progress] round {r + 1}/{args.rounds}, {time.time() - t1:.0f} s", flush=True)
    allr = [None] * world
    dist.all_gather_object(allr, dict(raw=raw, bg=bgms, outlast=outl))
    if rank == 0:
        K = list(raw)
        rmax = {k: [max(allr[q]["raw"][k][j] for q in range(world)) for j in range(len(raw[k]))] for k in K}
        ok_all = {k: [all(allr[q]["outlast"][k][j] for q in range(world)) for j in range(len(raw[k]))] for k in K}
        summ = {}
        for k in K:
            kept = [v for v, ok in zip(rmax[k], ok_all[k]) if ok]
            summ[k] = dict(median_ms=statistics.median(kept) if kept else None, kept=len(kept), rounds=len(rmax[k]))
            if k in bgms:
                bgv = [statistics.median(allr[q]["bg"][k]) for q in range(world)]
                tot = nb * (4 if k.startswith("new2048") else 1)
                summ[k]["bg_GBps_per_rank"] = [tot / (v / 1e3) / 1e9 for v in bgv]
        for o in ops:
            base = summ[f"{o}|none"]["median_ms"]
            line = f"[interf] {o:<8} none {base:8.3f} ms"
            for b in ("h2d", "d2h"):
                s = summ[f"{o}|{b}"]
                if s["median_ms"] is not None:
                    line += (f" | {b} {s['median_ms']:8.3f} ms ({100 * (s['median_ms'] / base - 1):+5.1f}%, "
                             f"kept {s['kept']}/{s['rounds']}, bg {min(s['bg_GBps_per_rank']):.1f}-"
                             f"{max(s['bg_GBps_per_rank']):.1f} GB/s)")
            print(line, flush=True)
        with open(os.path.join(args.out, "k4_interference_v1.json"), "w") as f:
            json.dump(dict(args=vars(args), rank_max_ms=rmax, outlast=ok_all, per_rank=allr, summary=summ,
                           numa_rank0=numa), f)
    dist.barrier()
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
