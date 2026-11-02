################################################################################
# hetero-proxy pflux V3 (2026-11-02): can a GPU send to and receive from pinned host memory at the same time without
# the copy-engine H2D collapse? V2 (results/pflux/v2_full/analysis.txt) found the board reduce-scatter limited to
# 6.45 GB/s per direction (= the Phase 0 contended-H2D anchor 6.46) and the Triton SM copy (32 CTAs) even slower,
# while nvbandwidth issue #53 (same CPU / topology) reports SM bidirectional ~21.5 / 21.5 GB/s. This script measures,
# per GPU, H2D only / D2H only / both at once, with each direction done by the copy engine (cuMemcpyAsync) or by a
# Triton copy kernel (CTAs x block x warps swept), alone on GPU0 and on all 8 GPUs at once.
# Each direction is timed on its own stream (CUDA events after a shared host-memory gate to end of that stream);
# bidirectional rates are bytes / own-direction time and the overlap window (min of the two times) is reported.
# Host buffers are NUMA-local to each GPU (first touch, verified with move_pages). One process, no NCCL.
#
# Predictions (registered 2026-11-02 before any run):
#   Q1 CE on GPU0: H2D alone ~22 GB/s, D2H alone ~22.5, bidirectional H2D 6-9 / D2H ~22 (Phase 0: 6.46 / 22.52;
#      nvbandwidth 8.88 / 22.13).
#   Q2 SM with enough CTAs on GPU0: H2D alone >= 18 GB/s; bidirectional >= 15 GB/s each way (if issue #53 carries over).
#   Q3 SM with 32 CTAs, block 2048 (the V2 setting): bidirectional <= 12 GB/s per direction (V2 rs_sm was slow).
#   Q4 all 8 GPUs, best SM (or mixed) bidirectional setting: >= 9 GB/s per GPU per direction (CE: ~6.45, V2).
# Decision rule: Q4 holds (both directions >= 9 GB/s per GPU on g8) -> rerun the V2 reduce-scatter with that setting
# (V2c) and recompute R2; otherwise try CPU-pushed H2D (GDRCopy) or accept the all-gather-only gain.
# Usage (repo root): scripts/pflux/run_v3_v1.sh (guard + tmux)
################################################################################
import argparse
import ctypes
import json
import mmap
import os
import random
import statistics
import sys

import torch
import triton
import triton.language as tl

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from hxb import cudamem as cm  # noqa: E402

NODE_CPUS = {0: list(range(0, 32)) + list(range(64, 96)), 1: list(range(32, 64)) + list(range(96, 128))}
MIB = 2 ** 20
NB = 32 * MIB          # per direction per repetition (V2 reduce-scatter moves 35 MiB each way at M=4096)
REPS = 4               # copies per direction per item -> 128 MiB per direction
SM_SETS = [(32, 2048, 4), (64, 2048, 4), (108, 2048, 4), (216, 2048, 4), (108, 8192, 8), (216, 8192, 8)]


@triton.jit(do_not_specialize=["n"])
def _copy_k(dst, src, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    step = tl.num_programs(0) * BLOCK
    for start in range(pid * BLOCK, n, step):
        offs = start + tl.arange(0, BLOCK)
        m = offs < n
        tl.store(dst + offs, tl.load(src + offs, mask=m), mask=m)


class Ptr:
    dtype = torch.int64

    def __init__(self, addr):
        self.addr = addr

    def data_ptr(self):
        return self.addr


def numa_pages(addr, nbytes, samples=64):
    lib = ctypes.CDLL("libnuma.so.1")
    lib.numa_move_pages.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p,
                                    ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    pages = (ctypes.c_void_p * samples)(*[(addr + (nbytes - 1) * i // (samples - 1)) & ~4095 for i in range(samples)])
    status = (ctypes.c_int * samples)()
    assert lib.numa_move_pages(0, samples, pages, None, status, 0) == 0
    out = {}
    for v in status:
        out[f"N{v}"] = out.get(f"N{v}", 0) + 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=50)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.sched_setaffinity(0, [0, 1])
    torch.set_num_threads(1)

    go_mm = mmap.mmap(-1, 4096)  # page-aligned host gate, registered below
    go_addr = ctypes.addressof(ctypes.c_char.from_buffer(go_mm))
    go_val = ctypes.c_int64.from_address(go_addr)
    st, dev, host, numa = {}, {}, {}, {}
    for g in range(8):
        node = g // 4
        with torch.cuda.device(g):
            st[g] = (torch.cuda.Stream(), torch.cuda.Stream())  # (d2h stream, h2d stream)
            dev[g] = (torch.empty(NB, dtype=torch.uint8, device=g).fill_(g), torch.empty(NB, dtype=torch.uint8, device=g))
        os.sched_setaffinity(0, NODE_CPUS[node][:2])
        hs = (torch.empty(NB, dtype=torch.uint8).fill_(1), torch.empty(NB, dtype=torch.uint8).fill_(2))  # first touch
        os.sched_setaffinity(0, [0, 1])
        for h in hs:
            cm.register(h.data_ptr(), NB)
        host[g] = hs
        numa[g] = [numa_pages(h.data_ptr(), NB) for h in hs]
    print(f"[v3] NUMA pages of host buffers per GPU: {json.dumps(numa)}", flush=True)
    cm.register(go_addr, 4096)

    def sm_copy(g, stream, dst, src, nbytes, cfg):
        ctas, block, warps = cfg
        with torch.cuda.device(g), torch.cuda.stream(stream):
            _copy_k[(ctas,)](Ptr(dst), Ptr(src), nbytes // 8, BLOCK=block, num_warps=warps)

    def ce_copy(g, stream, dst, src, nbytes):
        with torch.cuda.device(g):
            cm.memcpy(stream, dst, src, nbytes)

    for g in range(8):  # compile / load every kernel variant on every device before any gated launch
        for cfg in SM_SETS:
            sm_copy(g, st[g][0], dev[g][1].data_ptr(), dev[g][0].data_ptr(), MIB, cfg)
        torch.cuda.synchronize(g)

    epoch = [0]

    def item(gpus, d2h, h2d):
        """d2h / h2d: None, 'ce' or an SM cfg tuple. Returns {g: (d2h_ms, h2d_ms)}."""
        epoch[0] += 1
        e = epoch[0]
        ev = {}
        for g in gpus:
            s_up, s_dn = st[g]
            evs = []
            for s, eng, dst, src in ((s_up, d2h, host[g][0].data_ptr(), dev[g][0].data_ptr()),
                                     (s_dn, h2d, dev[g][1].data_ptr(), host[g][1].data_ptr())):
                if eng is None:
                    evs.append(None)
                    continue
                with torch.cuda.device(g):
                    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    cm.wait_geq(s, go_addr, e)
                    a.record(s)
                    for _ in range(REPS):
                        if eng == "ce":
                            ce_copy(g, s, dst, src, NB)
                        else:
                            sm_copy(g, s, dst, src, NB, eng)
                    b.record(s)
                evs.append((a, b))
            ev[g] = evs
        go_val.value = e  # release every stream at once
        out = {}
        for g in gpus:
            torch.cuda.synchronize(g)
            out[g] = tuple(None if x is None else x[0].elapsed_time(x[1]) for x in ev[g])
        return out

    items = []
    for gs_name, gpus in (("g0", [0]), ("g8", list(range(8)))):
        items += [(gs_name, gpus, "ce", None), (gs_name, gpus, None, "ce"), (gs_name, gpus, "ce", "ce")]
        sets = SM_SETS if gs_name == "g0" else [SM_SETS[2], SM_SETS[3], SM_SETS[5]]
        for cfg in sets:
            items += [(gs_name, gpus, cfg, None), (gs_name, gpus, None, cfg), (gs_name, gpus, cfg, cfg)]
        for cfg in (SM_SETS[2], SM_SETS[5]):
            items += [(gs_name, gpus, cfg, "ce"), (gs_name, gpus, "ce", cfg)]

    def name(it):
        f = lambda x: "-" if x is None else ("ce" if x == "ce" else "sm%dx%dw%d" % x)
        return f"{it[0]}|d2h={f(it[2])}|h2d={f(it[3])}"

    for it in items:  # warm (gated path, released immediately)
        item(it[1], it[2], it[3])
    rng = random.Random(7)
    rec = {name(it): {g: [] for g in it[1]} for it in items}
    for r in range(args.rounds):
        order = list(range(len(items)))
        rng.shuffle(order)
        for i in order:
            it = items[i]
            res = item(it[1], it[2], it[3])
            for g, v in res.items():
                rec[name(it)][g].append(v)
    total = NB * REPS
    summary = {}
    for k, per in rec.items():
        rows = {}
        for g, vs in per.items():
            up = [v[0] for v in vs if v[0] is not None]
            dn = [v[1] for v in vs if v[1] is not None]
            rows[g] = dict(d2h_ms=statistics.median(up) if up else None, h2d_ms=statistics.median(dn) if dn else None)
            for d in ("d2h", "h2d"):
                t = rows[g][f"{d}_ms"]
                rows[g][f"{d}_GBps"] = None if t is None else total / (t / 1e3) / 1e9
        summary[k] = rows
    json.dump(dict(args=vars(args), nb=NB, reps=REPS, numa=numa, summary=summary,
                   raw={k: {g: v for g, v in per.items()} for k, per in rec.items()}),
              open(os.path.join(args.out, "v3_bidir.json"), "w"), indent=1)
    for k, rows in summary.items():
        d2h = [r["d2h_GBps"] for r in rows.values() if r["d2h_GBps"]]
        h2d = [r["h2d_GBps"] for r in rows.values() if r["h2d_GBps"]]
        f = lambda xs: "    -    " if not xs else f"{min(xs):5.1f}-{max(xs):5.1f}"
        print(f"[v3] {k:40s} per-GPU GB/s  D2H {f(d2h)}  H2D {f(h2d)}"
              + (f"  (sum {sum(d2h):6.1f} / {sum(h2d):6.1f})" if len(rows) > 1 else ""), flush=True)


if __name__ == "__main__":
    main()
