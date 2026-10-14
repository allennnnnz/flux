################################################################################
# hetero-proxy I5 (2026-10-09): NPU-like device characteristics through the CPU backend's knobs, as an OUT-OF-SAMPLE
# test of the I3 stage model (the model was built from I2 v2 numbers and checked on emu4 / emu0 in I3; none of the
# configurations below was run before this script was committed).
# Same pass as I3 (GPU0 producer -> device -> GPU2 chunk-signaled consumer, M=4096, H=5120 bf16), timing-only device
# compute (CpuBackend emulate_tflops: sleep flops / rate, no CPU memory traffic). Configs:
#     base4      compute 4 ms / pass, dynamic (3 queues), device link unthrottled, no op overhead   (= I3 emu4)
#     fast1      compute 1 ms                                                 (an A100-class chip: s ~ 1/4 .. 1)
#     slow16     compute 16 ms                                                (a weak chip)
#     static4    compute 4 ms, ONE queue (static device), sends one chunk ahead (lookahead 1)
#     link8      compute 4 ms, every device copy throttled to 8 GB/s (link_gbps=8: a Gen3 x8 / Gen4 x4 chip)
#     ovh4       compute 4 ms, +100 us fixed cost per device op (op_overhead_us=100: heavy launch path)
# n = 1, 2, 4, 8, 16; 100 interleaved rounds per config; gate after flush; dst events; guard.
#
# PRE-REGISTERED MODEL (same form as I3; stage times per chunk at n chunks, I2 v2 characterisation):
#   p producer chunk, g consumer GEMM-only / n (in-run), d "D2H GPU0 | with H2D GPU2", h "H2D GPU2 | with D2H GPU0",
#   e engine per chunk "copy_in+copy_out dma_threads=4 staged (engine per chunk)", c = compute ms / n.
#   knobs:  link8   e = max(e, 2 x chunk_bytes / 8 GB/s)
#           ovh4    e += 2 x 0.1 ms, c += 0.1 ms
#           static4 the device is one stage: dev = e + c (copies and compute serialised on one queue)
#   level 0: T0 = p + d_alone + e + c + h_alone + g;  level 1: T1 = sum(stages) + (n - 1) x max(stages)
#   stages = (p, d, e, c, h, g), or (p, d, e + c, h, g) for static4
# Claims: Q1 for n <= 8 every config is within 15% of T (I3: the model held to 14% for n <= 8 when compute-bound;
#            emu0 / link-bound was off by up to 46%, so link8 is the risky one).
#         Q2 the best n is the model's pick or a neighbour (1-2-4-8-16 ladder).
#         Q3 static4 is slower than base4 at every n >= 2 and its best-n speedup over its level 0 is below base4's.
#         Q4 ovh4's best n is not larger than base4's (per-op overhead pushes the optimum to fewer chunks).
# Usage: CUDA_MODULE_LOADING=EAGER python i5_npu_sim_v1.py --out <dir> [--rounds 100]   (inside exclusive_guard_v2)
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
from hxb.pipeline import StagePipeline, Weights  # noqa: E402
from i3_overlap_v1 import gemm_times, load_char, max_clock  # noqa: E402

M, H, R = 4096, 5120, 256
NS = (1, 2, 4, 8, 16)
SRC, DST = 0, 2
FLOPS = 4 * M * H * R
CONFIGS = {  # name: (compute ms per pass, scheduling, link_gbps, op_overhead_us)
    "base4": (4.0, "dynamic", None, 0.0),
    "fast1": (1.0, "dynamic", None, 0.0),
    "slow16": (16.0, "dynamic", None, 0.0),
    "static4": (4.0, "static", None, 0.0),
    "link8": (4.0, "dynamic", 8.0, 0.0),
    "ovh4": (4.0, "dynamic", None, 100.0),
}


def predict(ch, name, n, p, g):
    comp_ms, sched, link, ovh = CONFIGS[name]
    nb = (M // n) * H * 2
    d = ch[("D", "D2H GPU0 | with H2D GPU2", n)]
    h = ch[("D", "H2D GPU2 | with D2H GPU0", n)]
    d0, h0 = ch[("D", "D2H GPU0 alone", n)], ch[("D", "H2D GPU2 alone", n)]
    e = ch[("A", "copy_in+copy_out dma_threads=4 staged (engine per chunk)", n)]
    c = comp_ms / n
    if link:
        e = max(e, 2 * nb / (link * 1e9) * 1e3)
    if ovh:
        e += 2 * ovh / 1e3
        c += ovh / 1e3
    if n == 1:
        return p + d0 + e + c + h0 + g, dict(p=p, d=d0, e=e, c=c, h=h0, g=g)
    st = (p, d, e + c, h, g) if sched == "static" else (p, d, e, c, h, g)
    return sum(st) + (n - 1) * max(st), dict(p=p, d=d, e=e, c=c, h=h, g=g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=100)
    ap.add_argument("--configs", default=",".join(CONFIGS))
    args = ap.parse_args()
    os.sched_setaffinity(0, list(range(0, 32)))
    os.makedirs(args.out, exist_ok=True)
    ch = load_char()
    log = open(os.path.join(args.out, "i5_log.txt"), "w")

    def say(s):
        print(s, flush=True)
        log.write(s + "\n")
    say(f"# start {time.strftime('%Y-%m-%d %H:%M:%S')} rounds={args.rounds}")
    rw_f = open(os.path.join(args.out, "rounds.csv"), "w", newline="")
    rw = csv.writer(rw_f)
    rw.writerow(["config", "n", "round", "gpu_ms", "host_ms", "enqueue_ms", "submit_ms", "dev_in_sum_ms",
                 "dev_op_sum_ms", "dev_out_sum_ms", "clk_src_max", "clk_dst_max"])
    summary = []
    rng = random.Random(0)
    for name in args.configs.split(","):
        comp_ms, sched, link, ovh = CONFIGS[name]
        be = CpuBackend(emulate_tflops=FLOPS / (comp_ms * 1e-3) / 1e12, scheduling=sched, link_gbps=link,
                        op_overhead_us=ovh, name=name)
        br = Bridge(be)
        try:
            qs = (be.queue(), be.queue(), be.queue())  # a static device returns its one queue three times
            w = Weights(br, qs[0], SRC, DST, H, R)
            pls = {n: StagePipeline(br, qs, w, SRC, DST, M, n, lookahead=1 if sched == "static" else 0)
                   for n in NS}
            preds = {}
            for n in NS:
                gt = gemm_times(pls[n])
                preds[n] = predict(ch, name, n, gt["p"], gt["g_full"] / n)
                say(f"[pred] {name:<8} n={n:<3} T {preds[n][0]:7.3f} ms  stages " +
                    " ".join(f"{k}={v:.3f}" for k, v in preds[n][1].items()))
            for n in NS:
                pls[n].enqueue()
                pls[n].run()
            be.trace()
            recs = {n: [] for n in NS}
            with ClockLogger(SRC, 0.25) as cs, ClockLogger(DST, 0.25) as cd:
                for r in range(args.rounds):
                    order = list(NS)
                    rng.shuffle(order)
                    for n in order:
                        rec = pls[n].enqueue() and pls[n].run()
                        tr = be.trace()
                        sums = {k: sum(x["t_end"] - x["t_start"] for x in tr if x["kind"] == k) * 1e3
                                for k in ("copy_in", "launch", "copy_out")}
                        recs[n].append([name, n, r, rec["gpu_s"] * 1e3, rec["host_s"] * 1e3, rec["enqueue_s"] * 1e3,
                                        rec["submit_s"] * 1e3, sums["copy_in"], sums["launch"], sums["copy_out"],
                                        (rec["t_gate"], rec["t_gate"] + rec["host_s"])])
                for n in NS:
                    for row in recs[n]:
                        w0, w1 = row[-1]
                        row[-1:] = [max_clock(cs, w0, w1), max_clock(cd, w0, w1)]
                        rw.writerow([f"{x:.4f}" if isinstance(x, float) else x for x in row])
            rw_f.flush()
            lvl0 = None
            for n in NS:
                gpu = statistics.median(x[3] for x in recs[n])
                lvl0 = gpu if n == 1 else lvl0
                T = preds[n][0]
                s = dict(config=name, n=n, rounds=len(recs[n]), gpu_ms=gpu, pred_ms=T, err=gpu / T - 1,
                         speedup_vs_level0=lvl0 / gpu, host_ms=statistics.median(x[4] for x in recs[n]),
                         enqueue_ms=statistics.median(x[5] for x in recs[n]),
                         dev_in_sum_ms=statistics.median(x[7] for x in recs[n]),
                         dev_op_sum_ms=statistics.median(x[8] for x in recs[n]),
                         dev_out_sum_ms=statistics.median(x[9] for x in recs[n]),
                         **{f"stage_{k}": v for k, v in preds[n][1].items()})
                summary.append(s)
                say(f"[meas] {name:<8} n={n:<3} gpu {gpu:7.3f} ms  pred {T:7.3f}  err {100 * s['err']:+6.1f}%  "
                    f"x{s['speedup_vs_level0']:4.2f} vs level 0 | dev in {s['dev_in_sum_ms']:5.2f} op "
                    f"{s['dev_op_sum_ms']:5.2f} out {s['dev_out_sum_ms']:5.2f} | enqueue {s['enqueue_ms']:5.2f}")
            del pls
            torch.cuda.empty_cache()
        finally:
            br.close()
            be.close()
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    say(f"# end {time.strftime('%Y-%m-%d %H:%M:%S')}")
    rw_f.close()
    log.close()


if __name__ == "__main__":
    main()
