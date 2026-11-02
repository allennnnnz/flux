################################################################################
# hetero-proxy pflux V3c (2026-11-02): does a CPU push into GPU memory (GDRCopy: CPU write-combining stores through the
# GPU BAR1 mapping, HFReduce SC24's H2D path) avoid the same-GPU H2D collapse that copy-engine H2D suffers while the
# GPU uploads (D2H)? V2 / V3: copy-engine H2D drops to ~6.5 GB/s per GPU on 8 GPUs when D2H runs at the same time;
# SM copies were worse. Here the H2D direction is done by T CPU threads per GPU (pinned to the GPU's NUMA node, each
# copying a slice with gdr_copy_to_mapping), optionally while the copy engine of the same GPU does D2H.
# Push time = CPU wall time from a shared release to the last thread's return (stores are posted; an sfence ends
# gdr_copy_to_mapping); D2H time = CUDA events after a host-memory gate released at the same moment.
# Correctness: device buffer equals the host source after a push (checked once per GPU).
#
# Predictions (registered 2026-11-02 before any run; no prior measurement of CPU->GPU BAR writes on this machine):
#   G1 one thread to one GPU: 4-10 GB/s.
#   G2 four threads to GPU0: >= 15 GB/s.
#   G3 GPU0, 4 threads + concurrent CE D2H: push drops <= 20% vs alone and D2H stays >= 18 GB/s (no collapse: both
#      directions are posted writes, no GPU read requests).
#   G4 all 8 GPUs, 4 threads each + CE D2H on every GPU: >= 9 GB/s per GPU in both directions.
# Decision: G4 holds -> build the reduce-scatter with CPU-pushed H2D (V2c) and recompute R2; otherwise stop at the
# all-gather-only result and report to the user.
# Usage (repo root): scripts/pflux/run_v3c_v1.sh (guard + tmux)
################################################################################
import argparse
import ctypes
import json
import mmap
import os
import random
import statistics
import sys
import threading
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from hxb import cudamem as cm  # noqa: E402

NODE_CPUS = {0: list(range(0, 32)), 1: list(range(32, 64))}  # physical cores only (HT siblings 64-127 unused)
MIB = 2 ** 20
NB = 32 * MIB
REPS = 4
GPU_PAGE = 1 << 16


class MH(ctypes.Structure):
    _fields_ = [("h", ctypes.c_ulong)]


class Info(ctypes.Structure):
    _fields_ = [("va", ctypes.c_uint64), ("mapped_size", ctypes.c_uint64), ("page_size", ctypes.c_uint32),
                ("tm_cycles", ctypes.c_uint64), ("cycles_per_ms", ctypes.c_uint32), ("bits", ctypes.c_uint32),
                ("mapping_type", ctypes.c_int)]


gdr = ctypes.CDLL("libgdrapi.so.2")
gdr.gdr_open.restype = ctypes.c_void_p
gdr.gdr_pin_buffer.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_size_t, ctypes.c_uint64, ctypes.c_uint32,
                               ctypes.POINTER(MH)]
gdr.gdr_map.argtypes = [ctypes.c_void_p, MH, ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
gdr.gdr_get_info_v2.argtypes = [ctypes.c_void_p, MH, ctypes.POINTER(Info)]
gdr.gdr_copy_to_mapping.argtypes = [MH, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
cu = ctypes.CDLL("libcuda.so.1")
cu.cuPointerSetAttribute.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64]
CU_POINTER_ATTRIBUTE_SYNC_MEMOPS = 6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=30)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    main_cpus = [64, 65]
    os.sched_setaffinity(0, main_cpus)
    torch.set_num_threads(1)
    g_h = gdr.gdr_open()
    assert g_h, "gdr_open failed"

    go_mm = mmap.mmap(-1, 4096)
    go_addr = ctypes.addressof(ctypes.c_char.from_buffer(go_mm))
    go_val = ctypes.c_int64.from_address(go_addr)
    cm.register(go_addr, 4096)

    dev_keep, dmap, mh, src, d2h_src, d2h_dst, st = {}, {}, {}, {}, {}, {}, {}
    info_out = {}
    for g in range(8):
        node = g // 4
        with torch.cuda.device(g):
            raw = torch.empty(NB + GPU_PAGE, dtype=torch.uint8, device=g)
            base = (raw.data_ptr() + GPU_PAGE - 1) & ~(GPU_PAGE - 1)
            one = ctypes.c_uint32(1)
            r = cu.cuPointerSetAttribute(ctypes.byref(one), CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, base)
            assert r == 0, f"cuPointerSetAttribute {r}"
            h = MH()
            r = gdr.gdr_pin_buffer(g_h, base, NB, 0, 0, ctypes.byref(h))
            assert r == 0, f"gdr_pin_buffer {r}"
            va = ctypes.c_void_p()
            r = gdr.gdr_map(g_h, h, ctypes.byref(va), NB)
            assert r == 0, f"gdr_map {r}"
            inf = Info()
            gdr.gdr_get_info_v2(g_h, h, ctypes.byref(inf))
            dmap[g] = va.value + (base - inf.va)
            info_out[g] = dict(va=inf.va, base=base, mapped_size=inf.mapped_size, mapping_type=inf.mapping_type)
            mh[g], dev_keep[g] = h, (raw, base)
            st[g] = torch.cuda.Stream()
            d2h_src[g] = torch.empty(NB, dtype=torch.uint8, device=g).fill_(g)
        os.sched_setaffinity(0, NODE_CPUS[node][:2])
        src[g] = torch.randint(0, 255, (NB,), dtype=torch.uint8)            # push source, first touch on `node`
        d2h_dst[g] = torch.empty(NB, dtype=torch.uint8).fill_(0)
        os.sched_setaffinity(0, main_cpus)
        cm.register(d2h_dst[g].data_ptr(), NB)
    print(f"[v3c] gdr mappings: {json.dumps(info_out)}", flush=True)

    def push_job(g, T, t, start_evt, done):
        os.sched_setaffinity(0, [NODE_CPUS[g // 4][(g % 4) * 8 + t]])
        sl = NB // T
        off = t * sl
        start_evt.wait()
        for _ in range(REPS):
            gdr.gdr_copy_to_mapping(mh[g], dmap[g] + off, src[g].data_ptr() + off, sl)
        done[(g, t)] = time.perf_counter()

    epoch = [0]

    def item(gpus, T, d2h):
        epoch[0] += 1
        e = epoch[0]
        evs = {}
        if d2h:
            for g in gpus:
                with torch.cuda.device(g):
                    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    cm.wait_geq(st[g], go_addr, e)
                    a.record(st[g])
                    for _ in range(REPS):
                        cm.memcpy(st[g], d2h_dst[g].data_ptr(), d2h_src[g].data_ptr(), NB)
                    b.record(st[g])
                    evs[g] = (a, b)
        start_evt, done, ths = threading.Event(), {}, []
        if T:
            for g in gpus:
                for t in range(T):
                    th = threading.Thread(target=push_job, args=(g, T, t, start_evt, done))
                    th.start()
                    ths.append(th)
            time.sleep(0.02)  # let every thread pin itself and park on the event
        t0 = time.perf_counter()
        go_val.value = e
        start_evt.set()
        for th in ths:
            th.join()
        out = {}
        for g in gpus:
            push_ms = (max(done[(g, t)] for t in range(T)) - t0) * 1e3 if T else None
            d2h_ms = None
            if d2h:
                torch.cuda.synchronize(g)
                d2h_ms = evs[g][0].elapsed_time(evs[g][1])
            out[g] = (push_ms, d2h_ms)
        return out

    # correctness: one push per GPU, compare
    ok = {}
    for g in range(8):
        item([g], 4, False)
        raw, base = dev_keep[g]
        off = base - raw.data_ptr()
        ok[g] = bool(torch.equal(raw[off:off + NB].cpu(), src[g]))
    print(f"[v3c] push correctness per GPU: {ok}", flush=True)

    items = [("g0", [0], T, False) for T in (1, 2, 4, 8)] + [("g0", [0], 4, True), ("g0", [0], 0, True)]
    items += [("g8", list(range(8)), T, d) for T in (2, 4) for d in (False, True)] + [("g8", list(range(8)), 0, True)]
    name = lambda it: f"{it[0]}|push_T={it[2]}|ce_d2h={int(it[3])}"
    for it in items:
        item(it[1], it[2], it[3])
    rec = {name(it): {g: [] for g in it[1]} for it in items}
    rng = random.Random(11)
    for r in range(args.rounds):
        order = list(range(len(items)))
        rng.shuffle(order)
        for i in order:
            it = items[i]
            for g, v in item(it[1], it[2], it[3]).items():
                rec[name(it)][g].append(v)
    total = NB * REPS
    summary = {}
    for k, per in rec.items():
        rows = {}
        for g, vs in per.items():
            p = [v[0] for v in vs if v[0] is not None]
            d = [v[1] for v in vs if v[1] is not None]
            rows[g] = dict(push_ms=statistics.median(p) if p else None, d2h_ms=statistics.median(d) if d else None)
            rows[g]["push_GBps"] = None if not p else total / (rows[g]["push_ms"] / 1e3) / 1e9
            rows[g]["d2h_GBps"] = None if not d else total / (rows[g]["d2h_ms"] / 1e3) / 1e9
        summary[k] = rows
        ps = [x["push_GBps"] for x in rows.values() if x["push_GBps"]]
        ds = [x["d2h_GBps"] for x in rows.values() if x["d2h_GBps"]]
        f = lambda xs: "    -     " if not xs else f"{min(xs):5.1f}-{max(xs):5.1f}"
        print(f"[v3c] {k:28s} per-GPU GB/s  push(H2D) {f(ps)}  CE D2H {f(ds)}"
              + (f"  (sum {sum(ps):6.1f} / {sum(ds):6.1f})" if len(rows) > 1 else ""), flush=True)
    json.dump(dict(args=vars(args), nb=NB, reps=REPS, gdr=info_out, correct=ok, summary=summary,
                   raw={k: {g: v for g, v in per.items()} for k, per in rec.items()}),
              open(os.path.join(args.out, "v3c_gdrpush.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
