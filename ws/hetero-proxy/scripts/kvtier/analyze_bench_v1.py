################################################################################
# hetero-proxy KV tier (2026-10-16): analysis of kvtier_v1.py --mode bench.
# Inputs: <dir>/bench_kvtier_v1.json (rank-max ms per case and round, rank-0 wall windows) and <dir>/clocks.csv
# (nvidia-smi -lms 200: timestamp, gpu, SM MHz, W, C; css-host-158 local time = UTC).
# Clock filter (CLAUDE.md 5.1.6): a round's clock = the lowest SM clock of any GPU among the samples inside its
# window (or the last one before it); rounds below 0.95 x the modal clock are dropped. Reports both all rounds and
# kept rounds; the pre-registered predictions K1-K7 are checked against the kept-round medians.
# Usage: python3 analyze_bench_v1.py <dir>
################################################################################
import bisect
import datetime
import json
import os
import statistics
import sys

d = sys.argv[1]
b = json.load(open(os.path.join(d, "bench_kvtier_v1.json")))
out = []


def say(s=""):
    print(s)
    out.append(s)


samples = {}
for line in open(os.path.join(d, "clocks.csv")):
    p = [x.strip() for x in line.split(",")]
    if len(p) < 3 or "MHz" not in p[2]:
        continue
    t = datetime.datetime.strptime(p[0], "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=datetime.timezone.utc).timestamp()
    samples.setdefault(t, {})[int(p[1])] = int(p[2].split()[0])
ts = sorted(samples)


def clock(w0, w1):
    i = max(0, bisect.bisect_right(ts, w0) - 1)
    j = bisect.bisect_right(ts, w1)
    vals = [min(samples[t].values()) for t in ts[i:max(j, i + 1)]]
    return min(vals) if vals else None


rm, win = b["rank_max_ms"], b["windows_rank0"]
clk = {k: [clock(*w) for w in win[k]] for k in rm}
allc = [c for v in clk.values() for c in v if c]
modal = statistics.mode(allc)
keep = {k: [ms for ms, c in zip(rm[k], clk[k]) if c is None or c >= 0.95 * modal] for k in rm}
say(f"SM clock: modal {modal} MHz over {len(allc)} rounds; min {min(allc)}, max {max(allc)}; "
    f"rounds dropped (< {0.95 * modal:.0f} MHz): {sum(len(rm[k]) - len(keep[k]) for k in rm)} of "
    f"{sum(len(v) for v in rm.values())}")
med = {k: statistics.median(v) for k, v in keep.items() if v}
med_all = {k: statistics.median(v) for k, v in rm.items()}
args = b["args"]
Ps = [int(p) for p in args["P"].split(",")]
kinds = ["ideal", "recompute", "serial", "layer"] + [f"split{float(x):g}" for x in args["split"].split(",") if x]
NB = lambda P: P * 32 * 1024  # noqa: E731  KV bytes per GPU

say("")
say(f"TTFT of one request, history P + new turn T={args['T']} tokens, rank-max median over kept rounds (ms); "
    "x = speedup vs recompute")
say(f"{'P':>6} " + " ".join(f"{k:>18}" for k in kinds))
for P in Ps:
    rec = med[f"recompute|P{P}"]
    cells = []
    for k in kinds:
        v = med[f"{k}|P{P}"]
        cells.append(f"{v:9.2f}" + (f" (x{rec / v:5.1f})" if k != "recompute" else "         "))
    say(f"{P:>6} " + " ".join(f"{c:>18}" for c in cells))
say("")
say("pure copies, per GPU with all 8 GPUs at once (rank-max median; per-rank medians from all rounds)")
for P in Ps:
    for k in ("load", "save"):
        key = f"{k}|P{P}"
        pr = [statistics.median(r[key]) for r in b["per_rank_ms"]]
        say(f"  {k:<4} P={P:<6} {NB(P) / 1e6:7.1f} MB: {med[key]:7.2f} ms = {NB(P) / med[key] / 1e6:5.1f} GB/s rank-max;"
            f" per rank {' '.join(f'{NB(P) / v / 1e6:4.1f}' for v in pr)} GB/s")
say("")
say("all rounds vs kept rounds (median ms), largest relative change:")
ch = sorted(((abs(med_all[k] / med[k] - 1), k) for k in med), reverse=True)[:5]
for c, k in ch:
    say(f"  {k:<22} all {med_all[k]:9.2f}  kept {med[k]:9.2f}  ({100 * c:.1f}%)")

say("")
say("pre-registered predictions (kvtier_v1.py header) vs kept-round medians:")
lo = min(NB(P) / med[f"load|P{P}"] / 1e6 for P in Ps if P >= 4096)
hi = max(NB(P) / med[f"load|P{P}"] / 1e6 for P in Ps if P >= 4096)
say(f"  K1 load 9-12.5 GB/s/GPU (P >= 4096): {lo:.1f}-{hi:.1f} -> {'holds' if lo >= 9 and hi <= 12.5 else 'FAILS'}")
r8, r32 = med.get("recompute|P8192"), med.get("recompute|P32000")
if r8 and r32:
    say(f"  K2 recompute P=8192 450-650 ms: {r8:.0f}; P=32000 1800-3000 ms: {r32:.0f} -> "
        f"{'holds' if 450 <= r8 <= 650 and 1800 <= r32 <= 3000 else 'FAILS'}")
idl = [med[f"ideal|P{P}"] for P in Ps]
say(f"  K3 ideal 10-30 ms, last <= 2x first: {min(idl):.1f}-{max(idl):.1f}, ratio {idl[-1] / idl[0]:.2f} -> "
    f"{'holds' if min(idl) >= 10 and max(idl) <= 30 and idl[-1] <= 2 * idl[0] else 'FAILS'}")
k4 = [(P, med[f"layer|P{P}"] - max(med[f"load|P{P}"], med[f"ideal|P{P}"])) for P in Ps]
say("  K4 layer - max(load, ideal) <= 3 ms: " + ", ".join(f"P{P} {v:+.1f}" for P, v in k4) +
    f" -> {'holds' if all(v <= 3 for _, v in k4) else 'FAILS'}")
k5 = [(P, med[f"recompute|P{P}"] / med[f"layer|P{P}"]) for P in Ps]
ok5 = all(s >= (10 if P >= 4096 else 3) for P, s in k5 if P >= 1024)
say("  K5 layer speedup >= 10x (P >= 4096), >= 3x (P = 1024): " + ", ".join(f"P{P} x{s:.1f}" for P, s in k5) +
    f" -> {'holds' if ok5 else 'FAILS'}")
k6 = [(P, med[f"serial|P{P}"] - med[f"layer|P{P}"]) for P in Ps]
say("  K6 serial - layer 5-25 ms: " + ", ".join(f"P{P} {v:.1f}" for P, v in k6) +
    f" -> {'holds' if all(5 <= v <= 25 for _, v in k6) else 'FAILS'}")
k7 = [(P, min(med[f"{k}|P{P}"] for k in kinds if k.startswith("split")) / med[f"layer|P{P}"] - 1) for P in Ps]
say("  K7 best split vs layer > -10%: " + ", ".join(f"P{P} {100 * v:+.1f}%" for P, v in k7) +
    f" -> {'holds' if all(v > -0.10 for _, v in k7) else 'FAILS'}")
open(os.path.join(d, "analysis.txt"), "w").write("\n".join(out) + "\n")
