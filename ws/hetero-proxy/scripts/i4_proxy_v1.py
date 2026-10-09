################################################################################
# hetero-proxy I4 (2026-10-09): the same hxb interface and Bridge with a second, very different backend - an A100
# (GPU2) playing a PCIe-only chip (hxb/proxy_backend.py: queues are CUDA streams, waits / done are stream memory ops
# on host counters, no host thread, peer access never enabled). Same pass as I3:
#     GPU0 A = X @ W1 (row chunks) -> proxy GPU2 B = lowrank_gelu(A) -> GPU4 Z = B @ W3 (chunk-signaled GEMM)
# dst = GPU4: a different PCIe switch and NUMA node from GPU0 and GPU2 (GPU1 / GPU3 share a switch with them).
# Configs (name, compute, copy queues):
#     pemu4_sep / pemu4_shared   timing-only compute 4 ms per pass (calibrated spin)  separate / shared (lookahead 1)
#     preal_sep / preal_shared   real bf16 lowrank_gelu R=256 on GPU2 (~0.1 ms)         separate / shared
# "shared" = copy_in and copy_out on one queue (directions serialised), sends submitted one chunk ahead.
# Measurement as I3 (gate after flush, dst events, 200 interleaved rounds, guard); NVLink data counters of GPUs 0, 2, 4
# read before and after the whole run (`nvidia-smi nvlink -gt d`, raw text saved).
#
# PRE-REGISTERED (committed before the run):
#   P1 NVLink: the Tx + Rx delta over all links of GPUs 0, 2 and 4 is < 1 MiB each (everything went over PCIe).
#   P2 an A100 receiving while it sends drops its H2D to ~6 GB/s (PHASE0 2.4; I2 v2 D): pemu4_shared is faster
#      than pemu4_sep for every n >= 4.
#   P3 preal output matches the fp32 reference (max rel err < 2e-2) for every n.
#   P4 the Bridge mechanism does not depend on the backend: pemu4_shared's best time is within 25% of the CPU
#      backend's emu4 best (5.878 ms, results/i3_overlap). [推論 estimate for pemu4_shared n=8: per-chunk stages
#      p 0.16, d 0.22, in+out 0.47, c 0.50, h ~0.28 (GPU4 from NUMA-0 staging, Phase 0 18.6 GB/s), g 0.11 ->
#      1.74 + 7 x 0.50 = 5.2 ms]
# Usage: CUDA_MODULE_LOADING=EAGER python i4_proxy_v1.py --out <dir> [--rounds 200]   (inside exclusive_guard_v2)
################################################################################
import argparse
import csv
import json
import os
import random
import re
import statistics
import subprocess
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from common.measure.clock_logger import ClockLogger  # noqa: E402
from hxb.bridge import Bridge  # noqa: E402
from hxb.pipeline import StagePipeline, Weights  # noqa: E402
from hxb.proxy_backend import ProxyBackend  # noqa: E402
from i3_overlap_v1 import max_clock  # noqa: E402

M, H, R = 4096, 5120, 256
NS = (1, 2, 4, 8, 16, 32)
SRC, DEV, DST = 0, 2, 4
EMU_MS = 4.0
EMU_MS_PER_GFLOP = EMU_MS / (4 * M * H * R / 1e9)
CONFIGS = [("pemu4_sep", True, False), ("pemu4_shared", True, True),
           ("preal_sep", False, False), ("preal_shared", False, True)]


def nvlink_counters(gpus):
    raw, tot = {}, {}
    for g in gpus:
        out = subprocess.run(["nvidia-smi", "nvlink", "-gt", "d", "-i", str(g)], capture_output=True, text=True,
                             timeout=30).stdout
        raw[g] = out
        tot[g] = sum(int(v) for v in re.findall(r"Data [TR]x: (\d+) KiB", out))
    return raw, tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--configs", default=",".join(c[0] for c in CONFIGS))
    args = ap.parse_args()
    os.sched_setaffinity(0, list(range(0, 32)))
    os.makedirs(args.out, exist_ok=True)
    log = open(os.path.join(args.out, "i4_log.txt"), "w")

    def say(s):
        print(s, flush=True)
        log.write(s + "\n")
    say(f"# start {time.strftime('%Y-%m-%d %H:%M:%S')} rounds={args.rounds} src={SRC} proxy={DEV} dst={DST}")
    raw0, nv0 = nvlink_counters((SRC, DEV, DST))
    with open(os.path.join(args.out, "nvlink_before.txt"), "w") as f:
        f.write("\n".join(raw0[g] for g in raw0))
    rw_f = open(os.path.join(args.out, "rounds.csv"), "w", newline="")
    rw = csv.writer(rw_f)
    rw.writerow(["config", "n", "round", "gpu_ms", "host_ms", "enqueue_ms", "submit_ms", "dev_in_sum_ms",
                 "dev_op_sum_ms", "dev_out_sum_ms", "d2h_sum_ms", "h2d_sum_ms", "clk_src_max", "clk_dst_max"])
    summary = []
    rng = random.Random(0)
    wanted = args.configs.split(",")
    for emu in (True, False):
        cfgs = [c for c in CONFIGS if c[1] == emu and c[0] in wanted]
        if not cfgs:
            continue
        be = ProxyBackend(DEV, emulate_ms_per_gflop=EMU_MS_PER_GFLOP if emu else None)
        br = Bridge(be)
        try:
            q_in, q_op, q_out = be.queue(), be.queue(), be.queue()
            w = Weights(br, q_in, SRC, DST, H, R)
            for name, _, shared in cfgs:
                qs = (q_in, q_op, q_in if shared else q_out)
                pls = {n: StagePipeline(br, qs, w, SRC, DST, M, n, timing=True, lookahead=1 if shared else 0)
                       for n in NS}
                for n in NS:
                    pls[n].enqueue()
                    pls[n].run()
                if not emu:
                    ref = pls[1].reference()
                    for n in NS:
                        err = ((pls[n].Z.float() - ref.float()).abs().max() / ref.float().abs().max()).item()
                        say(f"[check] {name} n={n} max rel err vs fp32 reference {err:.2e} "
                            f"{'OK' if err < 2e-2 else 'MISMATCH'}")
                be.trace()
                for n in NS:
                    pls[n].ob.events.clear()
                    pls[n].ib.events.clear()
                recs = {n: [] for n in NS}
                with ClockLogger(SRC, 0.25) as cs, ClockLogger(DST, 0.25) as cd:
                    for r in range(args.rounds):
                        order = list(NS)
                        rng.shuffle(order)
                        for n in order:
                            pl = pls[n]
                            pl.enqueue()
                            rec = pl.run()
                            tr = be.trace()
                            d2h = sum(a.elapsed_time(b) for _, a, b in pl.ob.events)
                            h2d = sum(a.elapsed_time(b) for _, a, b in pl.ib.events)
                            pl.ob.events.clear()
                            pl.ib.events.clear()
                            sums = {k: sum(x["t_end"] - x["t_start"] for x in tr if x["kind"] == k) * 1e3
                                    for k in ("copy_in", "launch", "copy_out")}
                            recs[n].append([name, n, r, rec["gpu_s"] * 1e3, rec["host_s"] * 1e3,
                                            rec["enqueue_s"] * 1e3, rec["submit_s"] * 1e3, sums["copy_in"],
                                            sums["launch"], sums["copy_out"], d2h, h2d,
                                            (rec["t_gate"], rec["t_gate"] + rec["host_s"])])
                    for n in NS:
                        for row in recs[n]:
                            w0, w1 = row[-1]
                            row[-1:] = [max_clock(cs, w0, w1), max_clock(cd, w0, w1)]
                            rw.writerow([f"{x:.4f}" if isinstance(x, float) else x for x in row])
                rw_f.flush()
                lvl0 = None
                for n in NS:
                    rows = recs[n]
                    gpu = statistics.median(x[3] for x in rows)
                    lvl0 = gpu if n == 1 else lvl0
                    s = dict(config=name, n=n, rounds=len(rows), gpu_ms=gpu,
                             host_ms=statistics.median(x[4] for x in rows), speedup_vs_level0=lvl0 / gpu,
                             enqueue_ms=statistics.median(x[5] for x in rows),
                             dev_in_sum_ms=statistics.median(x[7] for x in rows),
                             dev_op_sum_ms=statistics.median(x[8] for x in rows),
                             dev_out_sum_ms=statistics.median(x[9] for x in rows),
                             d2h_sum_ms=statistics.median(x[10] for x in rows),
                             h2d_sum_ms=statistics.median(x[11] for x in rows))
                    summary.append(s)
                    say(f"[meas] {name:<13} n={n:<3} gpu {gpu:7.3f} ms (host {s['host_ms']:7.3f}) x{lvl0 / gpu:4.2f} | "
                        f"dev in {s['dev_in_sum_ms']:5.2f} op {s['dev_op_sum_ms']:5.2f} out {s['dev_out_sum_ms']:5.2f}"
                        f" | d2h {s['d2h_sum_ms']:5.2f} h2d {s['h2d_sum_ms']:5.2f} | enqueue {s['enqueue_ms']:5.2f}")
                del pls
                torch.cuda.empty_cache()
        finally:
            br.close()
            be.close()
    raw1, nv1 = nvlink_counters((SRC, DEV, DST))
    with open(os.path.join(args.out, "nvlink_after.txt"), "w") as f:
        f.write("\n".join(raw1[g] for g in raw1))
    for g in nv0:
        d = nv1[g] - nv0[g]
        say(f"[nvlink] GPU{g}: Tx+Rx delta over the run {d} KiB {'OK (< 1 MiB)' if d < 1024 else 'TRAFFIC'}")
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(dict(summary=summary, nvlink_delta_kib={g: nv1[g] - nv0[g] for g in nv0}), f, indent=1)
    say(f"# end {time.strftime('%Y-%m-%d %H:%M:%S')}")
    rw_f.close()
    log.close()


if __name__ == "__main__":
    main()
