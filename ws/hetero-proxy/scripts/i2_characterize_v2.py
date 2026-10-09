################################################################################
# hetero-proxy I2 v2 (2026-10-09): characterise the CPU backend and the Bridge's GPU copies, in isolation and under
# contention, as inputs to the I3 overlap prediction. Shapes: activation M=4096 x H=5120 bf16 (40 MiB), row chunks
# of M/n rows. Device = NUMA node 1 (CpuBackend), staging = NUMA node 0 (host process pinned to cpus 0-31).
# v2 vs v1: v1's section A copied one cache-hot host buffer every rep and overstated the device DMA ~2x against the
# pipeline (I3 dry run). v2 moves data the way the pipeline does:
#   A  device DMA (C engine) on pipeline-like data: per chunk, GPU0 D2H into one of 4 rotating staging slots, then
#      copy_in from it; copy_out into a slot, then GPU2 H2D from it (each step waited for, so each copy is alone);
#      dma_threads 4 / 8. Also both directions together through the one engine (in + out per chunk).
#   B  device compute alone: lowrank_gelu (R = 64, 256) per chunk rows
#   C  contention: compute while a feeder keeps copy_in and copy_out busy (rotating 4-slot staging, cold data)
#   D  GPU copies (cuMemcpyAsync, registered staging): D2H GPU0, H2D GPU0 / GPU2, alone and concurrent
#      (same GPU both ways, GPU0 D2H + GPU2 H2D), and GPU0 D2H + GPU2 H2D while the device DMA runs both ways.
#      Concurrent cases count only copies inside the window where all directions are in flight (CLAUDE.md 5.1.7).
#   E  per-op overhead: back-to-back tiny ops on one queue (gap between ops), host signal -> op start latency
#   F  (added after the I3 dry runs) device DMA engine per chunk (copy_in + copy_out, GPU-staged slots) while GPU0
#      D2H and GPU2 H2D stream continuously through other NUMA-0 staging memory: the pipeline's condition
# Times: device ops from the device trace (time.perf_counter in the device process), GPU copies from CUDA events.
# Usage: CUDA_MODULE_LOADING=EAGER python i2_characterize_v2.py --out <dir>   (inside exclusive_guard_v2)
################################################################################
import argparse
import csv
import os
import statistics
import sys
import threading
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hxb import cudamem as cm  # noqa: E402
from hxb.cpu_backend import CpuBackend  # noqa: E402

M, H = 4096, 5120
NS = (1, 2, 4, 8, 16, 32)
ROWS = []
BE_KW = {}


def emit(log, section, case, n, value_ms, nbytes=0, extra=""):
    gbps = nbytes / (value_ms / 1e3) / 1e9 if nbytes and value_ms > 0 else 0.0
    ROWS.append([section, case, n, f"{value_ms:.4f}", nbytes, f"{gbps:.2f}", extra])
    s = f"[{section}] {case:<44} n={n:<3} {value_ms:8.3f} ms" + (f"  {gbps:6.2f} GB/s" if gbps else "") + \
        (f"  {extra}" if extra else "")
    print(s, flush=True)
    log.write(s + "\n")


def dur(tr, kind=None, tag=None):
    return [r["t_end"] - r["t_start"] for r in tr if (kind is None or r["kind"] == kind)
            and (tag is None or r["tag"].startswith(tag))]


def med_ms(xs):
    return statistics.median(xs) * 1e3


def section_a(log, reps):
    g0 = torch.randn(M, H, device="cuda:0").to(torch.bfloat16)
    g2 = torch.empty(M, H, dtype=torch.bfloat16, device="cuda:2")
    s0, s2 = torch.cuda.Stream(0), torch.cuda.Stream(2)
    for d in (4, 8):
        be = CpuBackend(dma_threads=d, **BE_KW)
        try:
            q = be.queue()
            s = be.signal()
            v = 0
            dev = be.alloc((M, H), torch.bfloat16)
            for n in NS:
                cr = M // n
                nb = cr * H * 2
                slots = [be.host_alloc((cr, H), torch.bfloat16) for _ in range(4)]
                for h in slots:
                    cm.register(h.addr, h.nbytes)
                be.trace()
                k = max(reps, 2 * n)
                for i in range(k):  # GPU-staged data, each copy alone
                    h, r = slots[i % 4], ((i % n) * cr, (i % n + 1) * cr)
                    cm.memcpy(s0, h.addr, g0.data_ptr() + r[0] * H * 2, nb)
                    s0.synchronize()
                    v += 1
                    be.copy_in(q, dev, r, h, None, done=(s, v), tag="in")
                    s.wait(v)
                    h = slots[(i + 2) % 4]
                    v += 1
                    be.copy_out(q, h, None, dev, r, done=(s, v), tag="out")
                    s.wait(v)
                    cm.memcpy(s2, g2.data_ptr() + r[0] * H * 2, h.addr, nb)
                    s2.synchronize()
                tr = be.trace()
                emit(log, "A", f"copy_in  dma_threads={d} staged", n, med_ms(dur(tr, "copy_in")), nb)
                emit(log, "A", f"copy_out dma_threads={d} staged", n, med_ms(dur(tr, "copy_out")), nb)
                # one engine, both directions: in and out submitted together per chunk (pipeline order)
                be.trace()
                for i in range(k):
                    h_in, h_out, r = slots[i % 4], slots[(i + 2) % 4], ((i % n) * cr, (i % n + 1) * cr)
                    cm.memcpy(s0, h_in.addr, g0.data_ptr() + r[0] * H * 2, nb)
                    s0.synchronize()
                    be.copy_in(q, dev, r, h_in, None, tag="in")
                    v += 1
                    be.copy_out(q, h_out, None, dev, r, done=(s, v), tag="out")
                    s.wait(v)
                tr = be.trace()
                pair = [b["t_end"] - a["t_start"] for a, b in zip(
                    [x for x in tr if x["kind"] == "copy_in"], [x for x in tr if x["kind"] == "copy_out"])]
                emit(log, "A", f"copy_in+copy_out dma_threads={d} staged (engine per chunk)", n, med_ms(pair), 2 * nb)
                for h in slots:
                    cm.unregister(h.addr)
        finally:
            be.close()


def lowrank_args(be, q, R, s, v):
    g = torch.Generator().manual_seed(0)
    u = be.alloc((H, R), torch.float32)
    w = be.alloc((R, H), torch.float32)
    for buf, t in ((u, torch.randn(H, R, generator=g) / H ** 0.5), (w, torch.randn(R, H, generator=g) / R ** 0.5)):
        h = be.host_alloc(t.shape, t.dtype)
        h.tensor.copy_(t)
        be.copy_in(q, buf, None, h, None)
    x, y = be.alloc((M, H), torch.bfloat16), be.alloc((M, H), torch.bfloat16)
    return dict(x=x, u=u, v=w, y=y)


def section_b_c(log, reps):
    be = CpuBackend(dma_threads=4, **BE_KW)
    try:
        qc, qi, qo = be.queue(), be.queue(), be.queue()
        s = be.signal()
        v = 0
        for R in (64, 256):
            a = lowrank_args(be, qc, R, s, v)
            for n in NS:
                cr = M // n
                be.trace()
                for i in range(max(reps // 2, 4) + 2):
                    r = ((i % n) * cr, (i % n + 1) * cr)
                    be.launch(qc, "lowrank_gelu", dict(a, rows=r), tag="op")
                be.sync()
                t = med_ms(dur(sorted(be.trace(), key=lambda x: x["t_start"])[2:], "launch"))  # first 2 = warm-up
                emit(log, "B", f"lowrank_gelu R={R} alone", n, t, 0,
                     f"{4 * cr * H * R / (t / 1e3) / 1e12:.3f} TFLOPS")
            # C: the same compute while a feeder keeps copy_in and copy_out busy on two other queues
            dev = be.alloc((M, H), torch.bfloat16)
            for n in (1, 8):
                cr = M // n
                hi = [be.host_alloc((cr, H), torch.bfloat16) for _ in range(4)]
                ho = [be.host_alloc((cr, H), torch.bfloat16) for _ in range(4)]
                k = max(reps // 2, 4)
                be.trace()
                with Feeder(be, qi, qo, dev, hi, ho, cr, n):
                    time.sleep(0.05)
                    fin = be.signal("fin")
                    for i in range(k):
                        r = ((i % n) * cr, (i % n + 1) * cr)
                        be.launch(qc, "lowrank_gelu", dict(a, rows=r), tag="op")
                    be.launch(qc, "sleep", {"seconds": 0}, done=(fin, 1))
                    fin.wait(1, timeout=600)
                    t_fin = time.perf_counter()
                be.sync(timeout=600)
                tr = be.trace()
                ops = [r for r in tr if r["tag"] == "op"]
                t_last = max(r["t_end"] for r in ops)
                cps = [r for r in tr if r["kind"] in ("copy_in", "copy_out") and r["t_end"] <= t_last]
                nb = cr * H * 2
                emit(log, "C", f"lowrank_gelu R={R} | copies both ways", n, med_ms(dur(ops)), 0,
                     f"copies until {1e3 * (max(r['t_end'] for r in tr if r['kind'] == 'copy_in') - t_last):.1f} "
                     f"ms after the last op")
                emit(log, "C", f"copy_in  | with compute R={R} + copy_out", n,
                     med_ms([r["t_end"] - r["t_start"] for r in cps if r["kind"] == "copy_in"]), nb)
                emit(log, "C", f"copy_out | with compute R={R} + copy_in", n,
                     med_ms([r["t_end"] - r["t_start"] for r in cps if r["kind"] == "copy_out"]), nb)
        # E: per-op overhead and host-signal latency
        tiny = be.alloc((1, 16), torch.float32)
        be.trace()
        for i in range(200):
            be.launch(qc, "axpb", {"x": tiny, "y": tiny, "rows": (0, 1), "a": 1.0, "b": 0.0}, tag="tiny")
        be.sync()
        tr = sorted(be.trace(), key=lambda r: r["t_start"])
        gaps = [b["t_start"] - a["t_end"] for a, b in zip(tr, tr[1:])]
        emit(log, "E", "tiny op duration", 1, med_ms(dur(tr)))
        emit(log, "E", "gap between back-to-back ops (same queue)", 1, med_ms(gaps))
        lat = []
        for i in range(50):
            g = be.signal("gate")
            be.launch(qc, "axpb", {"x": tiny, "y": tiny, "rows": (0, 1), "a": 1.0, "b": 0.0}, wait=[(g, 1)],
                      tag="gated")
            time.sleep(0.01)
            t0 = time.perf_counter()
            g.set(1)
            be.sync()
            lat.append([r for r in be.trace() if r["tag"] == "gated"][0]["t_start"] - t0)
        emit(log, "E", "host signal -> waiting op starts", 1, med_ms(lat))
    finally:
        be.close()


class Feeder:
    """Keeps copy_in (queue qi) and copy_out (queue qo) busy, at most two outstanding copies per queue; hi / ho may
    be lists of staging slots used in rotation (cold data, as in the pipeline)."""

    def __init__(self, be, qi, qo, dev, hi, ho, cr, n):
        self.be, self.qi, self.qo, self.dev, self.hi, self.ho, self.cr, self.n = be, qi, qo, dev, hi, ho, cr, n
        self.stop = threading.Event()
        self.si, self.so = be.signal("feed_in"), be.signal("feed_out")

    def _run(self):
        i = 0
        while not self.stop.is_set():
            r = ((i % self.n) * self.cr, (i % self.n + 1) * self.cr)
            hi = self.hi[i % len(self.hi)] if isinstance(self.hi, list) else self.hi
            ho = self.ho[i % len(self.ho)] if isinstance(self.ho, list) else self.ho
            self.be.copy_in(self.qi, self.dev, r, hi, None, done=(self.si, i + 1), tag="in")
            self.be.copy_out(self.qo, ho, None, self.dev, r, done=(self.so, i + 1), tag="out")
            if i >= 2:
                self.si.wait(i - 1, timeout=600)
                self.so.wait(i - 1, timeout=600)
            i += 1

    def __enter__(self):
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()
        return self

    def __exit__(self, *a):
        self.stop.set()
        self.th.join()


def gpu_copy_times(be, jobs, reps):
    """jobs: list of (device, kind, gpu_tensor, host_addr, nbytes), each on its own stream, all released by one host
    gate counter, `reps` copies back to back. CLAUDE.md 5.1.7: per job, only copies that END before the first job
    to finish has finished are counted (all directions in flight). Returns [(median ms, copies counted)]."""
    gate = be.signal("gate")
    streams = [torch.cuda.Stream(d) for d, *_ in jobs]
    evs = []
    for (d, kind, t, haddr, nb), st in zip(jobs, streams):
        with torch.cuda.device(d):
            cm.wait_geq(st, gate.addr, 1)
            e = [torch.cuda.Event(enable_timing=True)]
            e[0].record(st)
            for _ in range(reps):
                if kind == "d2h":
                    cm.memcpy(st, haddr, t.data_ptr(), nb)
                else:
                    cm.memcpy(st, t.data_ptr(), haddr, nb)
                x = torch.cuda.Event(enable_timing=True)
                x.record(st)
                e.append(x)
            evs.append(e)
    time.sleep(0.01)
    gate.set(1)
    for d in {j[0] for j in jobs}:
        torch.cuda.synchronize(d)
    ends = [[e[0].elapsed_time(x) for x in e] for e in evs]  # ms since each stream's gate event
    window = min(t[-1] for t in ends)
    out = []
    for t in ends:
        per = [t[i + 1] - t[i] for i in range(len(t) - 1) if t[i + 1] <= window + 1e-3]
        out.append((statistics.median(per), len(per)))
    return out


def section_d(log, reps):
    be = CpuBackend(dma_threads=4, **BE_KW)
    try:
        hosts = {k: be.host_alloc((M, H), torch.bfloat16) for k in ("a", "b", "c", "d")}
        for h in hosts.values():
            cm.register(h.addr, h.nbytes)
        cm.register(be.table.addr, be.table.nbytes)
        g0 = torch.randn(M, H, device="cuda:0").to(torch.bfloat16)
        g0b = torch.empty_like(g0)
        g2 = torch.empty(M, H, dtype=torch.bfloat16, device="cuda:2")
        gpu_copy_times(be, [(0, "d2h", g0, hosts["a"].addr, 1 << 20), (2, "h2d", g2, hosts["b"].addr, 1 << 20)], 3)

        def one(n, label, jobs, names):
            nb = (M // n) * H * 2
            r = max(reps, 24, 3 * n)
            res = gpu_copy_times(be, [(d, k, t, hosts[h].addr, nb) for d, k, t, h in jobs], r)
            for name, (ms, cnt) in zip(names, res):
                emit(log, "D", f"{name}{label}", n, ms, nb, f"copies in window {cnt}/{r}")
        for n in NS:
            one(n, " alone", [(0, "d2h", g0, "a")], ["D2H GPU0"])
            one(n, " alone", [(0, "h2d", g0b, "b")], ["H2D GPU0"])
            one(n, " alone", [(2, "h2d", g2, "b")], ["H2D GPU2"])
            one(n, "", [(0, "d2h", g0, "a"), (0, "h2d", g0b, "b")],
                ["D2H GPU0 | with H2D GPU0 (same GPU)", "H2D GPU0 | with D2H GPU0 (same GPU)"])
            one(n, "", [(0, "d2h", g0, "a"), (2, "h2d", g2, "b")],
                ["D2H GPU0 | with H2D GPU2", "H2D GPU2 | with D2H GPU0"])
        # GPU copies while the device DMA streams both ways through two other NUMA-0 staging buffers
        qi, qo = be.queue(), be.queue()
        dev = be.alloc((M, H), torch.bfloat16)
        be.trace()
        with Feeder(be, qi, qo, dev, hosts["c"], hosts["d"], M, 1):
            time.sleep(0.05)
            for n in (1, 8):
                one(n, "", [(0, "d2h", g0, "a"), (2, "h2d", g2, "b")],
                    ["D2H GPU0 | with H2D GPU2 + device DMA both", "H2D GPU2 | with D2H GPU0 + device DMA both"])
        be.sync(timeout=300)
        tr = be.trace()
        emit(log, "D", "device copy_in during the above", 1, med_ms(dur(tr, "copy_in")), M * H * 2)
        emit(log, "D", "device copy_out during the above", 1, med_ms(dur(tr, "copy_out")), M * H * 2)
        for h in hosts.values():
            cm.unregister(h.addr)
        cm.unregister(be.table.addr)
    finally:
        be.close()


def section_f(log, reps):
    """Device DMA engine per chunk (copy_in + copy_out, GPU-staged slots) while GPU0 D2H and GPU2 H2D stream
    continuously through other NUMA-0 staging buffers: the pipeline's condition for the device copies."""
    be = CpuBackend(dma_threads=4, **BE_KW)
    try:
        q = be.queue()
        s = be.signal()
        v = 0
        dev = be.alloc((M, H), torch.bfloat16)
        bg = [be.host_alloc((M, H), torch.bfloat16) for _ in range(2)]
        for h in bg:
            cm.register(h.addr, h.nbytes)
        g0 = torch.randn(M, H, device="cuda:0").to(torch.bfloat16)
        g2 = torch.empty(M, H, dtype=torch.bfloat16, device="cuda:2")
        s0, s2 = torch.cuda.Stream(0), torch.cuda.Stream(2)
        for n in NS:
            cr = M // n
            nb = cr * H * 2
            slots = [be.host_alloc((cr, H), torch.bfloat16) for _ in range(4)]
            for h in slots:
                cm.register(h.addr, h.nbytes)
                cm.memcpy(s0, h.addr, g0.data_ptr(), nb)
            s0.synchronize()
            k = max(reps, 2 * n)
            # background: enough 40 MiB copies each way for ~3x the expected foreground time (0.5 ms per chunk)
            nbg = max(8, int(3 * k * 0.5e-3 / 1.9e-3) + 1)
            for _ in range(nbg):
                cm.memcpy(s0, bg[0].addr, g0.data_ptr(), M * H * 2)
                cm.memcpy(s2, g2.data_ptr(), bg[1].addr, M * H * 2)
            time.sleep(0.003)
            be.trace()
            for i in range(k):
                r = ((i % n) * cr, (i % n + 1) * cr)
                be.copy_in(q, dev, r, slots[i % 4], None, tag="in")
                v += 1
                be.copy_out(q, slots[(i + 2) % 4], None, dev, r, done=(s, v), tag="out")
                s.wait(v)
            covered = (not s0.query()) and (not s2.query())
            tr = be.trace()
            s0.synchronize()
            s2.synchronize()
            pair = [b["t_end"] - a["t_start"] for a, b in zip(
                [x for x in tr if x["kind"] == "copy_in"], [x for x in tr if x["kind"] == "copy_out"])]
            emit(log, "F", "copy_in+copy_out dma_threads=4 staged | GPU0 D2H + GPU2 H2D", n, med_ms(pair), 2 * nb,
                 f"GPU traffic covered the whole window: {covered}")
            for h in slots:
                cm.unregister(h.addr)
        for h in bg:
            cm.unregister(h.addr)
    finally:
        be.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--sections", default="ABCDEF")
    ap.add_argument("--omp_spincount", type=int, default=None)
    ap.add_argument("--dma_cores", default="", help="explicit DMA cores, e.g. 28,29,30,31 (staging side, NUMA 0)")
    args = ap.parse_args()
    if args.omp_spincount is not None:
        BE_KW["omp_spincount"] = args.omp_spincount
    if args.dma_cores:
        BE_KW["dma_cores"] = [int(c) for c in args.dma_cores.split(",")]
    # NUMA 0; with --dma_cores on NUMA 0 leave 24-31 to the DMA threads
    os.sched_setaffinity(0, list(range(0, 24 if args.dma_cores else 32)))
    os.makedirs(args.out, exist_ok=True)
    log = open(os.path.join(args.out, "characterize_log.txt"), "w")
    log.write("# i2_characterize_v2\n")
    log.write(f"# start {time.strftime('%Y-%m-%d %H:%M:%S')} CUDA_MODULE_LOADING={os.environ.get('CUDA_MODULE_LOADING')} "
              f"backend {BE_KW}\n")
    if "A" in args.sections:
        section_a(log, args.reps)
    if "B" in args.sections or "C" in args.sections or "E" in args.sections:
        section_b_c(log, args.reps)
    if "D" in args.sections:
        section_d(log, args.reps)
    if "F" in args.sections:
        section_f(log, args.reps)
    with open(os.path.join(args.out, "characterize.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["section", "case", "n_chunks", "ms", "bytes", "GBps", "extra"])
        w.writerows(ROWS)
    log.write(f"# end {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    log.close()


if __name__ == "__main__":
    main()
