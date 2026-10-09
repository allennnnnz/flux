################################################################################
# hetero-proxy I3 (2026-10-09): does Flux-style chunk overlap work through the hxb interface, and does it deliver
# what the stage model predicts?  One pipeline-stage boundary pass (hxb/pipeline.py):
#     src GPU A = X @ W1 (row chunks)  ->  device B = lowrank_gelu(A)  ->  dst GPU Z = B @ W3 (chunk-signaled GEMM)
#     M=4096, H=5120 bf16 (40 MiB each way); n = 1 (level 0, op-level) or 2..32 chunks (level 1).
# Devices: "emu" = CpuBackend with timing-only compute (sleep flops / rate; no memory traffic, so the device has no
# CPU-specific contention) -> isolates the Bridge's overlap mechanism; "cpu" = real fp32 compute on NUMA node 1.
# Configs (name, device, R, emulated device compute per pass, src, dst):
#     emu4 / emu4_same   emu, compute 4 ms,  GPU0 -> GPU2   /  GPU0 -> GPU0
#     emu0               emu, compute ~0,    GPU0 -> GPU2   (link-bound)
#     cpu64 / cpu256     cpu, R = 64 / 256,  GPU0 -> GPU2
#     cpu64_same         cpu, R = 64,        GPU0 -> GPU0
# Measurement: all n of one config built first, then `rounds` rounds, n in random order each round; each pass is
# pre-enqueued behind a host gate, timed on the dst GPU from the gate to the consumer's end (CUDA events), cross-
# checked by the host (gate -> pass_done counter). SM clock logged for src and dst; a round is dropped when the
# HIGHEST sample inside its window is below 95% of the modal value (the lowest sample would flag the GPUs' idle
# phases of a device-bound pass as down-clocking; dry run 2026-10-09). If no round survives, all are kept and the
# summary says so. Device per-op times from the device trace, GPU copy times from CUDA events, every pass.
#
# PRE-REGISTERED PREDICTION (committed before the run; printed as "pred_*" columns before any pass):
#   per-chunk stage times at n chunks, from results/i2_characterize_v2/characterize.csv (same day) and the in-run
#   GEMM measurements (producer chunk p, consumer GEMM-only / n = g):
#     d  D2H   "D2H GPU0 | with H2D GPU2"  (src != dst) or "... (same GPU)" (src == dst)
#     h  H2D   "H2D GPU2 | with D2H GPU0"  (src != dst) or "H2D GPU0 | with D2H GPU0 (same GPU)"
#     e  device DMA engine per chunk = copy_in + copy_out of GPU-staged data through the one engine (v2 section A)
#     c  device compute: emu = flops / rate; cpu: optimistic = "B alone", pessimistic = "C copies both ways"
#        (n <= 2 use the n=1 contention ratio, n >= 4 the n=8 ratio)
#   level 0 (n=1): T0 = p + d_alone + e + c_alone + h_alone + g          (one step at a time)
#   level 1:       T1 = (p + d + e + c + h + g) + (n - 1) * max(p, d, e, c, h, g)   (linear pipeline)
#   Claims: P1 emu configs: measured within 15% of T (src != dst; same-GPU uses the contended H2D, an upper bound).
#           P2 cpu configs: measured between the optimistic and pessimistic T.
#           P3 at the best n, emu4 is >= 2x faster than its level 0; emu4_same is slower than emu4 for n >= 2.
#           P4 the best n is the one the model picks, or its neighbour.
# Usage: CUDA_MODULE_LOADING=EAGER python i3_overlap_v1.py --out <dir> [--rounds 200]   (inside exclusive_guard_v2)
################################################################################
import argparse
import csv
import json
import os
import random
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from common.measure.clock_logger import ClockLogger  # noqa: E402
from hxb.bridge import Bridge  # noqa: E402
from hxb.cpu_backend import CpuBackend  # noqa: E402
from hxb.gpu_consumer import chunk_gemm  # noqa: E402
from hxb.pipeline import StagePipeline, Weights  # noqa: E402

M, H = 4096, 5120
NS = (1, 2, 4, 8, 16, 32)
HERE = os.path.dirname(os.path.abspath(__file__))
CHAR = os.path.join(HERE, "..", "results", "i2_characterize_v2", "characterize.csv")
EMU_R = 256
EMU_FLOPS = 4 * M * H * EMU_R
CONFIGS = [  # name, device, R, emulated compute ms per pass (None = real), src, dst
    ("emu4", "emu", EMU_R, 4.0, 0, 2),
    ("emu4_same", "emu", EMU_R, 4.0, 0, 0),
    ("emu0", "emu", EMU_R, 0.0, 0, 2),
    ("cpu64", "cpu", 64, None, 0, 2),
    ("cpu256", "cpu", 256, None, 0, 2),
    ("cpu64_same", "cpu", 64, None, 0, 0),
]


def max_clock(clk, t0, t1):
    """Highest SM-clock sample inside [t0, t1]; else the last sample before t0."""
    v = [c for t, c in clk.samples if t0 <= t <= t1]
    if v:
        return max(v)
    prev = [c for t, c in clk.samples if t <= t0]
    return prev[-1] if prev else None


def load_char():
    t = {}
    with open(CHAR) as f:
        for r in csv.DictReader(f):
            t[(r["section"], r["case"].strip(), int(r["n_chunks"]))] = float(r["ms"])
    return t


def predict(ch, name, R, emu_ms, src, dst, n, p, g):
    """Pre-registered stage model; returns (optimistic, pessimistic) pass time in ms."""
    same = src == dst
    if same:
        d = ch[("D", "D2H GPU0 | with H2D GPU0 (same GPU)", n)]
        h = ch[("D", "H2D GPU0 | with D2H GPU0 (same GPU)", n)]
        h_alone = ch[("D", "H2D GPU0 alone", n)]
    else:
        d = ch[("D", "D2H GPU0 | with H2D GPU2", n)]
        h = ch[("D", "H2D GPU2 | with D2H GPU0", n)]
        h_alone = ch[("D", "H2D GPU2 alone", n)]
    d_alone = ch[("D", "D2H GPU0 alone", n)]
    e = ch[("A", "copy_in+copy_out dma_threads=4 staged (engine per chunk)", n)]
    if emu_ms is not None:
        c_opt = c_pes = emu_ms / n
    else:
        c_opt = ch[("B", f"lowrank_gelu R={R} alone", n)]
        ref = 1 if n <= 2 else 8
        ratio = ch[("C", f"lowrank_gelu R={R} | copies both ways", ref)] / ch[("B", f"lowrank_gelu R={R} alone", ref)]
        c_pes = c_opt * ratio
    out = []
    for c, c_lvl0 in ((c_opt, c_opt), (c_pes, c_opt)):
        if n == 1:
            out.append(p + d_alone + e + c_lvl0 + h_alone + g)
        else:
            st = (p, d, e, c, h, g)
            out.append(sum(st) + (n - 1) * max(st))
    return out[0], out[1], dict(p=p, d=d, h=h, e=e, c_opt=c_opt, c_pes=c_pes, g=g)


def gemm_times(pl, reps=30):
    """Producer chunk (src) and consumer GEMM-only full (dst), CUDA events, median ms."""
    cr = pl.cr
    res = {}
    for key, dev, fn in (("p", pl.src, lambda: torch.mm(pl.X[:cr], pl.w.W1, out=pl.A[:cr])),
                         ("g_full", pl.dst, lambda: chunk_gemm(pl.B, pl.w.W3, pl.Z))):
        with torch.cuda.device(dev):
            fn()
            ts = []
            for _ in range(reps):
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record()
                fn()
                b.record()
                ts.append((a, b))
            torch.cuda.synchronize(dev)
            res[key] = statistics.median(a.elapsed_time(b) for a, b in ts)
    return res


def device_backend(kind, emu_ms):
    if kind == "emu":
        rate = EMU_FLOPS / (emu_ms * 1e-3) / 1e12 if emu_ms else 1e9
        return CpuBackend(emulate_tflops=rate, name=f"emu{emu_ms}")
    return CpuBackend()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--configs", default=",".join(c[0] for c in CONFIGS))
    args = ap.parse_args()
    os.sched_setaffinity(0, list(range(0, 32)))
    os.makedirs(args.out, exist_ok=True)
    ch = load_char()
    log = open(os.path.join(args.out, "i3_log.txt"), "w")

    def say(s):
        print(s, flush=True)
        log.write(s + "\n")
    say(f"# start {time.strftime('%Y-%m-%d %H:%M:%S')} rounds={args.rounds} CUDA_MODULE_LOADING="
        f"{os.environ.get('CUDA_MODULE_LOADING')} CUDA_DEVICE_MAX_CONNECTIONS={os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS')}")
    rounds_f = open(os.path.join(args.out, "rounds.csv"), "w", newline="")
    rw = csv.writer(rounds_f)
    rw.writerow(["config", "n", "round", "gpu_ms", "host_ms", "enqueue_ms", "prod_ms", "dev_in_sum_ms",
                 "dev_op_sum_ms", "dev_out_sum_ms", "dev_op_med_ms", "dev_first_in_ms", "dev_last_out_ms",
                 "d2h_sum_ms", "h2d_sum_ms", "submit_ms", "dev_in_med_ms", "dev_out_med_ms", "clk_src_max",
                 "clk_dst_max"])
    summary = []
    rng = random.Random(0)
    wanted = args.configs.split(",")
    for kind in ("emu", "cpu"):
        groups = [c for c in CONFIGS if c[1] == kind and c[0] in wanted]
        by_emu = {}
        for c in groups:
            by_emu.setdefault(c[3], []).append(c)
        for emu_ms, cfgs in by_emu.items():
            be = device_backend(kind, emu_ms)
            br = Bridge(be)
            qs = (be.queue(), be.queue(), be.queue())
            try:
                for name, _, R, emu, src, dst in cfgs:
                    w = Weights(br, qs[0], src, dst, H, R)
                    pls = {n: StagePipeline(br, qs, w, src, dst, M, n, timing=True) for n in NS}
                    comp = {n: gemm_times(pls[n]) for n in NS}
                    preds = {}
                    for n in NS:
                        po, pp, parts = predict(ch, name, R, emu, src, dst, n, comp[n]["p"], comp[n]["g_full"] / n)
                        preds[n] = (po, pp, parts)
                        say(f"[pred] {name:<11} n={n:<3} T_opt {po:7.3f} ms  T_pes {pp:7.3f} ms  stages " +
                            " ".join(f"{k}={v:.3f}" for k, v in parts.items()))
                    # correctness (real compute only; emulated compute does not write its output)
                    for n in NS:
                        pls[n].enqueue()
                        pls[n].run()
                    if kind == "cpu":
                        ref = pls[1].reference()
                        for n in NS:
                            err = ((pls[n].Z.float() - ref.float()).abs().max() / ref.float().abs().max()).item()
                            say(f"[check] {name} n={n} max rel err vs fp32 reference {err:.2e}"
                                f" {'OK' if err < 2e-2 else 'MISMATCH'}")
                    be.trace()
                    for n in NS:
                        pls[n].ob.events.clear()
                        pls[n].ib.events.clear()
                    recs = {n: [] for n in NS}
                    with ClockLogger(src, 0.25) as cs, ClockLogger(dst, 0.25) as cd:
                        for r in range(args.rounds):
                            order = list(NS)
                            rng.shuffle(order)
                            for n in order:
                                pl = pls[n]
                                pl.enqueue()
                                rec = pl.run()
                                tr = be.trace()
                                t0 = rec["t_gate"]
                                ins = [x for x in tr if x["kind"] == "copy_in"]
                                ops = [x for x in tr if x["kind"] == "launch"]
                                outs = [x for x in tr if x["kind"] == "copy_out"]
                                d2h = sum(a.elapsed_time(b) for _, a, b in pl.ob.events)
                                h2d = sum(a.elapsed_time(b) for _, a, b in pl.ib.events)
                                pl.ob.events.clear()
                                pl.ib.events.clear()
                                row = [name, n, r, rec["gpu_s"] * 1e3, rec["host_s"] * 1e3, rec["enqueue_s"] * 1e3,
                                       rec["prod_s"] * 1e3,
                                       sum(x["t_end"] - x["t_start"] for x in ins) * 1e3,
                                       sum(x["t_end"] - x["t_start"] for x in ops) * 1e3,
                                       sum(x["t_end"] - x["t_start"] for x in outs) * 1e3,
                                       statistics.median(x["t_end"] - x["t_start"] for x in ops) * 1e3,
                                       (min(x["t_start"] for x in ins) - t0) * 1e3,
                                       (max(x["t_end"] for x in outs) - t0) * 1e3, d2h, h2d,
                                       rec["submit_s"] * 1e3,
                                       statistics.median(x["t_end"] - x["t_start"] for x in ins) * 1e3,
                                       statistics.median(x["t_end"] - x["t_start"] for x in outs) * 1e3,
                                       (t0, t0 + rec["host_s"])]
                                recs[n].append(row)
                        for n in NS:
                            for row in recs[n]:
                                w0, w1 = row[-1]
                                row[-1:] = [max_clock(cs, w0, w1), max_clock(cd, w0, w1)]
                                rw.writerow([f"{x:.4f}" if isinstance(x, float) else x for x in row])
                    rounds_f.flush()
                    clocks = [row[-2] for n in NS for row in recs[n] if row[-2]] + \
                             [row[-1] for n in NS for row in recs[n] if row[-1]]
                    modal = statistics.mode(clocks) if clocks else None
                    t_lvl0 = None
                    for n in NS:
                        kept = [row for row in recs[n] if modal is None or
                                ((row[-2] or modal) >= 0.95 * modal and (row[-1] or modal) >= 0.95 * modal)]
                        all_kept = not kept
                        if all_kept:
                            kept = recs[n]
                        gpu = statistics.median(row[3] for row in kept)
                        host = statistics.median(row[4] for row in kept)
                        if n == 1:
                            t_lvl0 = gpu
                        po, pp, parts = preds[n]
                        s = dict(config=name, n=n, kept=len(kept), rounds=len(recs[n]), modal_clock=modal,
                                 clock_filter_fell_back=all_kept,
                                 dev_in_med_ms=statistics.median(row[16] for row in kept),
                                 dev_out_med_ms=statistics.median(row[17] for row in kept),
                                 gpu_ms=gpu, host_ms=host, speedup_vs_level0=t_lvl0 / gpu,
                                 enqueue_ms=statistics.median(row[5] for row in kept),
                                 dev_op_sum_ms=statistics.median(row[8] for row in kept),
                                 dev_in_sum_ms=statistics.median(row[7] for row in kept),
                                 dev_out_sum_ms=statistics.median(row[9] for row in kept),
                                 d2h_sum_ms=statistics.median(row[13] for row in kept),
                                 h2d_sum_ms=statistics.median(row[14] for row in kept),
                                 pred_opt_ms=po, pred_pes_ms=pp, **{f"stage_{k}": v for k, v in parts.items()})
                        summary.append(s)
                        say(f"[meas] {name:<11} n={n:<3} gpu {gpu:7.3f} ms (host {host:7.3f}) x{t_lvl0 / gpu:4.2f} vs "
                            f"level 0 | pred {po:7.3f}..{pp:7.3f} | dev op sum {s['dev_op_sum_ms']:6.2f} in "
                            f"{s['dev_in_sum_ms']:5.2f} out {s['dev_out_sum_ms']:5.2f} | d2h {s['d2h_sum_ms']:5.2f} "
                            f"h2d {s['h2d_sum_ms']:5.2f} | enqueue {s['enqueue_ms']:5.2f} | kept {len(kept)}/{len(recs[n])}")
                    del pls
                    torch.cuda.empty_cache()
            finally:
                br.close()
                be.close()
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    say(f"# end {time.strftime('%Y-%m-%d %H:%M:%S')}")
    rounds_f.close()
    log.close()


if __name__ == "__main__":
    main()
