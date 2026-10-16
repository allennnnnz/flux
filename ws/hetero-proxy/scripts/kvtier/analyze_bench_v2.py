################################################################################
# hetero-proxy KV tier (2026-10-16): analysis of kvtier_v5.py --mode bench (pre-registered G1-G7).
# Inputs: <dir>/bench_kvtier_v5.json and <dir>/clocks.csv (nvidia-smi -lms 200; css-host-158 local time = UTC).
# Clock (CLAUDE.md 5.1.6): a round's clock = the lowest SM clock of any GPU among the samples inside its window (or the
# last one before it). Filter PER CASE: rounds below 0.95 x that case's own modal clock are dropped. (analyze_bench_v1
# used one modal clock for all cases; in the v2 smoke run every P = 32000 recompute round - 2 s of dense GEMM - ran
# below 0.95 x 1410 MHz under the power cap and was dropped, so the long recompute had no number. A long prefill runs
# at the capped clock in deployment too, so its own modal clock is the right reference.)
# Conservative speedup: recompute time scaled to 1410 MHz (x clock / 1410: GPU-bound work cannot get faster than
# that), divided by the case's time - a lower bound on the speedup if the recompute ran at the full clock.
# Usage: python3 analyze_bench_v2.py <dir>
################################################################################
import bisect
import datetime
import json
import os
import statistics
import sys

d = sys.argv[1]
b = json.load(open(os.path.join(d, "bench_kvtier_v5.json")))
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
med, cmod, dropped = {}, {}, 0
for k in rm:
    cs = [c for c in clk[k] if c]
    cmod[k] = statistics.mode(cs) if cs else None
    kept = [ms for ms, c in zip(rm[k], clk[k]) if c is None or cmod[k] is None or c >= 0.95 * cmod[k]]
    dropped += len(rm[k]) - len(kept)
    med[k] = statistics.median(kept)
med_all = {k: statistics.median(v) for k, v in rm.items()}
allc = [c for v in clk.values() for c in v if c]
say(f"SM clock over all {len(allc)} rounds: min {min(allc)}, modal {statistics.mode(allc)}, max {max(allc)} MHz; "
    f"rounds dropped by the per-case filter: {dropped} of {sum(len(v) for v in rm.values())}")
args = b["args"]
Ps = [int(p) for p in args["P"].split(",")]
xs = [f"split{float(x):g}" for x in args["split"].split(",") if x]
kinds = ["ideal", "ideal_eager", "recompute", "serial", "layer"] + xs
NB = lambda P: P * 32 * 1024  # noqa: E731  KV bytes per GPU

say("")
say(f"TTFT of one request: history P + new turn T={args['T']}, rank-max median (ms), kept rounds; (xN) = speedup vs "
    "recompute")
say(f"{'P':>6} " + " ".join(f"{k:>17}" for k in kinds))
for P in Ps:
    rec = med[f"recompute|P{P}"]
    say(f"{P:>6} " + " ".join(f"{med[f'{k}|P{P}']:8.2f}" + (f" (x{rec / med[f'{k}|P{P}']:5.1f})" if k != "recompute"
                                                          else "         ") for k in kinds))
say("")
say("modal SM clock per case (MHz):")
for P in Ps:
    say(f"  P={P:<6} " + "  ".join(f"{k} {cmod[f'{k}|P{P}']}" for k in kinds + ["load", "save"]))
say("")
say("conservative speedup of layer (recompute scaled to 1410 MHz):")
for P in Ps:
    rec = med[f"recompute|P{P}"] * (cmod[f"recompute|P{P}"] or 1410) / 1410
    say(f"  P={P:<6} recompute@1410 >= {rec:8.2f} ms  layer {med[f'layer|P{P}']:7.2f} ms  -> x{rec / med[f'layer|P{P}']:5.1f}")
say("")
say("pure copies, per GPU with all 8 GPUs at once (rank-max median; per-rank medians over all rounds):")
for P in Ps:
    for k in ("load", "save"):
        key = f"{k}|P{P}"
        pr = [statistics.median(r[key]) for r in b["per_rank_ms"]]
        say(f"  {k:<4} P={P:<6} {NB(P) / 1e6:7.1f} MB: {med[key]:7.2f} ms = {NB(P) / med[key] / 1e6:5.1f} GB/s rank-max;"
            f" per rank {' '.join(f'{NB(P) / v / 1e6:4.1f}' for v in pr)}")
say("")
say("all rounds vs kept rounds, largest relative changes:")
for c, k in sorted(((abs(med_all[k] / med[k] - 1), k) for k in med), reverse=True)[:5]:
    say(f"  {k:<22} all {med_all[k]:9.2f}  kept {med[k]:9.2f}  ({100 * c:.1f}%)")

smoke = {1024: 78.20, 2048: 132.74, 4096: 231.33, 8192: 445.57, 16384: 906.40, 32000: 1939.86}  # v2 smoke, stdout
say("")
say("pre-registered predictions (kvtier_v5.py / v3 header):")
idl = [med[f"ideal|P{P}"] for P in Ps]
say(f"  G1 ideal 8-20 ms: {min(idl):.1f}-{max(idl):.1f} -> {'holds' if min(idl) >= 8 and max(idl) <= 20 else 'FAILS'}")
g2 = [(P, med[f"recompute|P{P}"] / smoke[P] - 1) for P in Ps if P >= 8192]
d1 = smoke[1024] - med["recompute|P1024"]
say("  G2 recompute within 10% of eager (P >= 8192): " + ", ".join(f"P{P} {100 * v:+.1f}%" for P, v in g2) +
    f"; P=1024 faster by 15-45 ms: {d1:.1f} -> {'holds' if all(abs(v) <= 0.10 for _, v in g2) and 15 <= d1 <= 45 else 'FAILS'}")
g3 = [(P, med[f"layer|P{P}"] - max(med[f"load|P{P}"], med[f"ideal|P{P}"])) for P in Ps]
say("  G3 layer - max(load, ideal) <= 3 ms: " + ", ".join(f"P{P} {v:+.1f}" for P, v in g3) +
    f" -> {'holds' if all(v <= 3 for _, v in g3) else 'FAILS'}")
need = {1024: 2, 2048: 2, 4096: 8, 8192: 14, 16384: 14, 32000: 18}
g4 = [(P, med[f"recompute|P{P}"] / med[f"layer|P{P}"]) for P in Ps]
say("  G4 layer speedup (>= 2/2/8/14/14/18x): " + ", ".join(f"P{P} x{s:.1f}" for P, s in g4) +
    f" -> {'holds' if all(s >= need.get(P, 0) for P, s in g4) else 'FAILS'}")
g5 = [(P, (med[f"serial|P{P}"] - med[f"layer|P{P}"]) - min(med[f"load|P{P}"], med[f"ideal|P{P}"])) for P in Ps]
say("  G5 (serial - layer) - min(load, ideal) within 3 ms: " + ", ".join(f"P{P} {v:+.1f}" for P, v in g5) +
    f" -> {'holds' if all(abs(v) <= 3 for _, v in g5) else 'FAILS'}")
g6 = [(P, min(med[f"{k}|P{P}"] for k in xs) / med[f"layer|P{P}"] - 1) for P in Ps]
ok6 = all((v > -0.10) and (P >= 8192 or v > 0) for P, v in g6)
say("  G6 best split vs layer (>-10%; slower at P <= 4096): " + ", ".join(f"P{P} {100 * v:+.1f}%" for P, v in g6) +
    f" -> {'holds' if ok6 else 'FAILS'}")
ld = [NB(P) / med[f"load|P{P}"] / 1e6 for P in Ps]
sv = [NB(P) / med[f"save|P{P}"] / 1e6 for P in Ps]
say(f"  G7 load 9-12: {min(ld):.1f}-{max(ld):.1f}; save 9.5-12: {min(sv):.1f}-{max(sv):.1f} -> "
    f"{'holds' if min(ld) >= 9 and max(ld) <= 12 and min(sv) >= 9.5 and max(sv) <= 12 else 'FAILS'}")
open(os.path.join(d, "analysis.txt"), "w").write("\n".join(out) + "\n")
