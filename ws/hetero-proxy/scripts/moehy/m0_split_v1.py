################################################################################
# hetero-proxy M0 (2026-10-29): can 8 GPUs pulling cold MoE experts over all 4 PCIe uplinks add throughput to a
# CPU that computes the other cold experts (KTransformers-style hybrid), or does host DRAM contention eat the gain?
# reports/20261029_lit_moe_hybrid.md section 6 estimated ~1.9x from P1 numbers - but P1 timed ONE socket; KTransformers
# uses both (NUMA tensor parallel), so the CPU side is ~2x faster and the expected gain smaller. This is the cheap
# falsification test before building anything on SGLang / kt-kernel.
#
# Actors (all on host DRAM, NUMA-local, pools >= 1.5 GB so L3 never serves them):
#   CPU   two worker processes, one per socket (cores 4-31 / 36-63, 28 threads each); each computes ITS half of the
#         intermediate dim of every expert (KTransformers NUMA TP), bf16 weights, fp32 math (hxb_moe_expert, the P1
#         kernel), n tokens; experts taken round-robin from the socket's own pool.
#   GPU   this process (cores 0-1); expert weights cut into 8 slices along the intermediate dim (= TP8 shards);
#         slices 0-3 live in node-0 memory, 4-7 in node-1 memory (pinned); cuMemcpyAsync host -> device only
#         (GPU-side SwiGLU on a slice is ~10 us, not included). Sets:
#           g8  GPU g pulls slice g (NUMA-local)                       one expert = 8 copies of 1/8
#           g4  GPUs 0,2,4,6 (one per PCIe switch) pull 2 slices each  one expert = 4 copies of 1/4
#           g1  GPU0 pulls all 8 (4 from node 1 over the socket link)  single uplink, like a one-GPU system
#           pair_same / pair_diff  GPUs 0+1 (same switch) / 0+2 (different switches) pull the node-0 half: GB/s only
# Conditions per round, order shuffled: per shape {g1, g4, g8, pair_same, pair_diff} (GPU only) and per (shape, n)
# {cpu, cpu+g1, cpu+g4, cpu+g8}. Each condition runs W seconds; rates use only work that started >= 50 ms after the
# start signal. CPU rate = min over the two sockets (an expert needs both halves); GPU rate = min over GPUs of
# slices/s (an expert needs all its slices). Paired per round: gain = (cpu_rate_both + gpu_rate_both) / cpu_rate_alone.
# Shapes: DeepSeek-V3 expert (H 7168, I 2048: 88 MB bf16) and Qwen3-235B-A22B (H 4096, I 1536: 37.7 MB). n = 1, 4, 16.
#
# Predictions (registered 2026-10-29 before any run):
#   M1 GPU only: g8 aggregate 80-95 GB/s (per GPU 9-12; K1 anchor 10.1-12.2); g4 within 15% of g8 (the 4 uplinks are
#      the limit); g1 18-22 GB/s (PCIe ~22 anchor). pair_same 20-25 GB/s total, pair_diff 38-46.
#   M2 CPU only, dsv3 n=1: 125-150 GB/s per socket (P1 Mixtral n=1 139 GB/s on node 1; read anchor 172-176).
#   M3 cpu+g8: CPU per-socket rate drops 10-25%; GPU aggregate drops <= 10%; per-socket DRAM read (CPU + DMA from that
#      node) <= 1.1 x read anchor (~195 GB/s); more would mean the accounting is wrong.
#   M4 gain, dsv3 n=1 and n=4: g8 1.15-1.30x; g1 1.03-1.08x.
#   M5 n=16: g8 gain > 1.5x, but overstated (this kernel is compute-bound at n=16, P1 Mixtral 9.1 vs 2.5 ms at n=1;
#      kt-kernel would be faster) -> reported, not used for the decision.
#   Decision: build the scheduler only if median gain(g8, dsv3, n=4) >= 1.20 and gain(g8) - gain(g1) >= 0.10.
# Usage (repo root, exclusive guard, tmux):
#   python3 common/measure/exclusive_guard_v2.py --log <dir>/guard.log -- env CUDA_MODULE_LOADING=EAGER \
#     pixi run --manifest-path pixi.toml python ws/hetero-proxy/scripts/moehy/m0_split_v1.py --out <dir> [--rounds 30]
################################################################################
import argparse
import json
import multiprocessing as mp
import os
import random
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "kvtier"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "..", "common", "measure"))

SHAPES = {"dsv3": (7168, 2048, 48), "qwen235": (4096, 1536, 96)}  # H, I, pool experts (pool >= 1.5 GB per node)
NS = (1, 4, 16)
CPU_CORES = {0: list(range(4, 32)), 1: list(range(36, 64))}
CTRL_CORES = [0, 1]
GPU_SETS = {  # name -> [(gpu, node, first slice in node pool, n slices)]
    "g8": [(g, g // 4, g % 4, 1) for g in range(8)],
    "g4": [(0, 0, 0, 2), (2, 0, 2, 2), (4, 1, 0, 2), (6, 1, 2, 2)],
    "g1": [(0, 0, 0, 4), (0, 1, 0, 4)],
    "pair_same": [(0, 0, 0, 2), (1, 0, 2, 2)],
    "pair_diff": [(0, 0, 0, 2), (2, 0, 2, 2)],
}
DEPTH = 4
WARM = 0.05
GB = 1e9


def numa_pages(addr, nbytes, samples=256):
    """Node of `samples` pages spread over [addr, addr + nbytes) (move_pages query; /proc/self/numa_maps merges
    adjacent allocations into one mapping, so the smoke run's per-mapping counts mixed pools)."""
    import ctypes
    lib = ctypes.CDLL("libnuma.so.1")
    lib.numa_move_pages.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p,
                                    ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    pages = (ctypes.c_void_p * samples)(*[(addr + (nbytes - 1) * i // (samples - 1)) & ~4095 for i in range(samples)])
    status = (ctypes.c_int * samples)()
    assert lib.numa_move_pages(0, samples, pages, None, status, 0) == 0
    out = {}
    for v in status:
        out[f"N{v}" if v >= 0 else f"err{v}"] = out.get(f"N{v}" if v >= 0 else f"err{v}", 0) + 1
    return out


def cpu_worker(node, ctl, q):
    """Socket `node`: compute this socket's half of experts while ctl says so; send (seq, records) per phase."""
    os.sched_setaffinity(0, CPU_CORES[node])
    import torch
    from hxb import native
    nt = len(CPU_CORES[node])
    torch.set_num_threads(nt)
    L = native.lib()
    pools, info = {}, {}
    for name, (H, I, P) in SHAPES.items():
        Ih = I // 2
        w = torch.empty((P, 3, Ih, H), dtype=torch.bfloat16).uniform_(-0.05, 0.05)  # first touch on this node
        pools[name] = w
        info[name] = numa_pages(w.data_ptr(), w.numel() * 2)
    xs = {name: torch.randn(max(NS), SHAPES[name][0]) for name in SHAPES}
    outs = {name: torch.empty(max(NS), SHAPES[name][0]) for name in SHAPES}
    acts = {name: torch.empty(max(NS), SHAPES[name][1] // 2) for name in SHAPES}
    q.put(("ready", node, info))
    last = 0
    while True:
        seq = ctl[0]
        if ctl[4]:
            return
        if seq == last:
            time.sleep(0.0002)
            continue
        last = seq
        if not ctl[1]:
            continue
        name = list(SHAPES)[ctl[2]]
        n = ctl[3]
        H, I, P = SHAPES[name]
        Ih = I // 2
        w, x, o, a = pools[name], xs[name], outs[name], acts[name]
        recs, e = [], random.randrange(P)
        while ctl[0] == seq and ctl[5] == 0:
            base = w[e].data_ptr()
            t0 = time.perf_counter()
            L.hxb_moe_expert(x.data_ptr(), base, base + Ih * H * 2, base + 2 * Ih * H * 2, o.data_ptr(), a.data_ptr(),
                             n, H, Ih, nt)
            recs.append((t0, time.perf_counter()))
            e = (e + 1) % P
        q.put(("phase", node, seq, recs))


def cpu_mhz(cores):
    v = []
    for c in cores:
        try:
            v.append(int(open(f"/sys/devices/system/cpu/cpu{c}/cpufreq/scaling_cur_freq").read()) / 1000)
        except OSError:
            pass
    return statistics.mean(v) if v else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--window", type=float, default=0.5)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    ctx = mp.get_context("spawn")
    ctl = ctx.Array("i", [0, 0, 0, 1, 0, 0])  # seq, cpu_active, shape idx, n, exit, stop
    q = ctx.Queue()
    workers = [ctx.Process(target=cpu_worker, args=(node, ctl, q), daemon=True) for node in (0, 1)]
    for p in workers:
        p.start()

    os.sched_setaffinity(0, CTRL_CORES)
    import torch
    from hxb import cudamem as cm
    from clock_logger import ClockLogger

    log = open(os.path.join(args.out, "m0_log.txt"), "w")

    def say(s):
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()

    # GPU-side pools: node k holds slices 4k..4k+3 of every expert, layout [P, 4, 3, I/8, H] bf16, pinned
    hp, numa = {}, {}
    nt0 = torch.get_num_threads()
    torch.set_num_threads(1)
    for name, (H, I, P) in SHAPES.items():
        for node in (0, 1):
            os.sched_setaffinity(0, CPU_CORES[node][:2])
            t = torch.empty((P, 4, 3, I // 8, H), dtype=torch.bfloat16).fill_(1.0)  # first touch on `node`
            os.sched_setaffinity(0, CTRL_CORES)
            cm.register(t.data_ptr(), t.numel() * 2)
            hp[(name, node)] = t
            numa[f"gpu_pool_{name}_node{node}"] = numa_pages(t.data_ptr(), t.numel() * 2)
    torch.set_num_threads(nt0)
    for _ in workers:
        tag, node, info = q.get(timeout=600)
        for name, pages in info.items():
            numa[f"cpu_pool_{name}_node{node}"] = pages
    say(f"[m0] NUMA pages: {json.dumps(numa)}")

    streams = {g: torch.cuda.Stream(device=g) for g in range(8)}
    need = {}  # slices per expert a GPU pulls, max over sets
    for st in GPU_SETS.values():
        per = {}
        for (g, _, _, ns) in st:
            per[g] = per.get(g, 0) + ns
        for g, v in per.items():
            need[g] = max(need.get(g, 0), v)
    devbuf = {}
    for name, (H, I, P) in SHAPES.items():
        slice_b = 3 * (I // 8) * H * 2
        for g, v in need.items():
            devbuf[(name, g)] = torch.empty(DEPTH * v * slice_b, dtype=torch.uint8, device=g)

    def run_gpu(name, setname, t_go, window):
        """Pull experts with GPU set `setname` until window ends; per-GPU (bytes, seconds) from CUDA events."""
        H, I, P = SHAPES[name]
        slice_b = 3 * (I // 8) * H * 2
        per_gpu = {}
        for (g, node, s0, ns) in GPU_SETS[setname]:
            per_gpu.setdefault(g, []).append((node, s0, ns))
        evs = {g: [] for g in per_gpu}
        host_t = []
        e0 = random.randrange(P)
        i = 0
        while time.perf_counter() - t_go < window:
            e = (e0 + i) % P
            host_t.append(time.perf_counter())
            for g, parts in per_gpu.items():
                if i >= DEPTH:
                    evs[g][i - DEPTH].synchronize()
                s = streams[g]
                with torch.cuda.device(g):
                    off = 0
                    gbytes = sum(ns for (_, _, ns) in parts) * slice_b
                    dst = devbuf[(name, g)].data_ptr() + (i % DEPTH) * gbytes
                    for (node, s0, ns) in parts:
                        src = hp[(name, node)][e, s0].data_ptr()
                        cm.memcpy(s, dst + off, src, ns * slice_b)
                        off += ns * slice_b
                    ev = torch.cuda.Event(enable_timing=True)
                    ev.record(s)
                    evs[g].append(ev)
            i += 1
        for g in per_gpu:
            streams[g].synchronize()
        k = next((j for j, t in enumerate(host_t) if t >= t_go + WARM), None)
        res = {}
        for g, parts in per_gpu.items():
            gbytes = sum(ns for (_, _, ns) in parts) * slice_b
            if k is None or k + 1 >= i:
                res[g] = None
                continue
            sec = evs[g][k].elapsed_time(evs[g][i - 1]) / 1e3
            res[g] = dict(bytes=gbytes * (i - 1 - k), sec=sec, slices_per_expert=sum(ns for (_, _, ns) in parts))
        return res, i

    def cpu_rates(recs_by_node, t_go, t_stop, name):
        H, I, P = SHAPES[name]
        half_b = 3 * (I // 2) * H * 2
        out = {}
        for node, recs in recs_by_node.items():
            r = [(a, b) for a, b in recs if a >= t_go + WARM and b <= t_stop]
            if len(r) < 2:
                out[node] = None
                continue
            sec = r[-1][1] - r[0][0]
            out[node] = dict(count=len(r), sec=sec, gbps=len(r) * half_b / sec / GB, eps=len(r) / sec)
        return out

    conds = []
    for si, name in enumerate(SHAPES):
        for st in ("g1", "g4", "g8", "pair_same", "pair_diff"):
            conds.append((name, si, None, None, st))
        for n in NS:
            conds.append((name, si, n, "cpu", None))
            for st in ("g1", "g4", "g8"):
                conds.append((name, si, n, "cpu", st))

    raw, seq = [], 0
    with ClockLogger(0, period_s=0.5) as clk:
        for rnd in range(-1, args.rounds):  # round -1 = warm-up, discarded
            order = conds[:]
            random.shuffle(order)
            for (name, si, n, cpu, st) in order:
                seq += 1
                ctl[1], ctl[2], ctl[3], ctl[5] = (1 if cpu else 0), si, (n or 1), 0
                t_go = time.perf_counter()
                ctl[0] = seq
                mhz = None
                if st:
                    gres, iters = run_gpu(name, st, t_go, args.window)
                else:
                    time.sleep(args.window / 2)
                    mhz = (cpu_mhz(CPU_CORES[0][::7]), cpu_mhz(CPU_CORES[1][::7]))
                    time.sleep(args.window / 2)
                    gres, iters = None, 0
                t_stop = time.perf_counter()
                ctl[5] = 1
                recs = {}
                if cpu:
                    for _ in workers:
                        tag, node, sq, r = q.get(timeout=60)
                        assert sq == seq, (sq, seq)
                        recs[node] = r
                rec = dict(round=rnd, shape=name, n=n, cpu=bool(cpu), gpu=st, t_go=t_go, t_stop=t_stop, cpu_mhz=mhz,
                           sm_clock_min=clk.min_clock_between(t_go, t_stop),
                           cpu_res=cpu_rates(recs, t_go, t_stop, name) if cpu else None,
                           gpu_res=gres, gpu_iters=iters)
                if rnd >= 0:
                    raw.append(rec)
            say(f"[m0] round {rnd} done ({len(raw)} records)")
        sm = [c for _, c in clk.samples]
    ctl[4] = 1
    for p in workers:
        p.join(timeout=10)

    # summary: medians over rounds; gains paired within a round
    def gpu_metrics(gres):
        if not gres or any(v is None for v in gres.values()):
            return None
        gbps = {g: v["bytes"] / v["sec"] / GB for g, v in gres.items()}
        sps = {g: v["bytes"] / v["sec"] for g, v in gres.items()}  # bytes/s; expert rate below
        return gbps, sps

    summ, by = {}, {}
    for r in raw:
        by.setdefault((r["round"], r["shape"], r["n"], r["cpu"], r["gpu"]), r)
    for name, (H, I, P) in SHAPES.items():
        for st in ("g1", "g4", "g8", "pair_same", "pair_diff"):
            agg, per = [], {}
            for rnd in range(args.rounds):
                m = gpu_metrics(by[(rnd, name, None, False, st)]["gpu_res"])
                if m:
                    agg.append(sum(m[0].values()))
                    for g, v in m[0].items():
                        per.setdefault(g, []).append(v)
            summ[f"{name}|gpu_only|{st}"] = dict(agg_gbps=statistics.median(agg),
                                                 per_gpu_gbps={g: statistics.median(v) for g, v in per.items()},
                                                 rounds=len(agg))
        for n in NS:
            ca, cb, gains, ideal, sock_alone, sock_both, gpu_alone_eps, gpu_both_eps = [], {}, {}, {}, [], {}, {}, {}
            for rnd in range(args.rounds):
                a = by[(rnd, name, n, True, None)]["cpu_res"]
                if not a or any(v is None for v in a.values()):
                    continue
                ra = min(v["eps"] for v in a.values())  # experts/s (both halves needed)
                ca.append(ra)
                sock_alone.append({nd: v["gbps"] for nd, v in a.items()})
                for st in ("g1", "g4", "g8"):
                    b = by[(rnd, name, n, True, st)]
                    go = by[(rnd, name, None, False, st)]
                    mb, mo = gpu_metrics(b["gpu_res"]), gpu_metrics(go["gpu_res"])
                    if not b["cpu_res"] or any(v is None for v in b["cpu_res"].values()) or not mb or not mo:
                        continue
                    rb = min(v["eps"] for v in b["cpu_res"].values())
                    slice_b = 3 * (I // 8) * H * 2
                    spe = {g: v["slices_per_expert"] for g, v in b["gpu_res"].items()}
                    geb = min(mb[1][g] / (spe[g] * slice_b) for g in mb[1])  # GPU experts/s
                    gea = min(mo[1][g] / (spe[g] * slice_b) for g in mo[1])
                    gains.setdefault(st, []).append((rb + geb) / ra)
                    ideal.setdefault(st, []).append((ra + gea) / ra)
                    cb.setdefault(st, []).append(rb / ra)
                    gpu_both_eps.setdefault(st, []).append(geb / gea)
                    dram = {}
                    for nd, v in b["cpu_res"].items():
                        dma = sum(mb[0][g] for (g, node, _, _) in GPU_SETS[st] if node == nd and st != "g1")
                        if st == "g1":
                            dma = mb[0][0] / 2  # half of GPU0's bytes come from each node
                        dram[nd] = v["gbps"] + dma
                    sock_both.setdefault(st, []).append(dram)
            row = dict(cpu_alone_eps=statistics.median(ca), cpu_alone_gbps_per_socket={
                nd: statistics.median([s[nd] for s in sock_alone]) for nd in (0, 1)}, rounds=len(ca))
            for st in gains:
                row[st] = dict(gain=statistics.median(gains[st]), gain_ideal=statistics.median(ideal[st]),
                               cpu_rate_vs_alone=statistics.median(cb[st]),
                               gpu_rate_vs_alone=statistics.median(gpu_both_eps[st]),
                               dram_read_gbps_per_socket={nd: statistics.median([d[nd] for d in sock_both[st]])
                                                          for nd in (0, 1)},
                               gain_p10=sorted(gains[st])[len(gains[st]) // 10],
                               gain_p90=sorted(gains[st])[len(gains[st]) * 9 // 10], rounds=len(gains[st]))
            summ[f"{name}|n{n}"] = row
    mhz = [r["cpu_mhz"] for r in raw if r["cpu_mhz"]]
    out = dict(args=vars(args), numa=numa, shapes=SHAPES, gpu_sets=GPU_SETS, cpu_cores=CPU_CORES,
               sm_clock_modal=statistics.mode(sm) if sm else None,
               cpu_mhz_median=[statistics.median([m[0] for m in mhz if m[0]]), statistics.median(
                   [m[1] for m in mhz if m[1]])] if mhz else None,
               summary=summ, raw=raw)
    with open(os.path.join(args.out, "m0_split_v1.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)
    for k, v in summ.items():
        say(f"{k}: {json.dumps(v)}")
    say("[m0] done")


if __name__ == "__main__":
    main()
