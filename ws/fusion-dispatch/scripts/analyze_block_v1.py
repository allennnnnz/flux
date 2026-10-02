################################################################################
# fusion-dispatch F1.3 analysis: block-level results of validate_block_v1.py.
#   * per (phase, mode, M): median of each policy (rank-max, kept rounds)
#   * two-level dispatcher "auto": layout (tp_ar vs sp_dispatch) chosen per M from the
#     CALIBRATION run, evaluated on the EVALUATION run (different seed, fresh rounds)
#   * compared with vLLM default (tp_ar), always-Flux (sp_flux; absent in graph mode,
#     where AGKernel cannot be captured), and the eval-run oracle (best measured policy)
#   * table check (plan G1): does the per-op table predict block-level differences?
#     predicted = L x sum over the 4 ops of (t_path - t_nccl) from the op table;
#     measured  = median per-round (policy - sp_nccl) in the eval run.
# Usage: python3 analyze_block_v1.py <f1_dir>   (expects <f1_dir>/cal, <f1_dir>/eval, table_*.json)
################################################################################
import csv
import glob
import json
import os
import statistics
import sys
from collections import defaultdict

d = sys.argv[1]


def load(sub):
    t = defaultdict(lambda: defaultdict(dict))  # (phase,mode,M) -> policy -> round -> ms
    for f in glob.glob(os.path.join(d, sub, "raw_*.csv")):
        for r in csv.DictReader(open(f)):
            if r["kept"] == "1":
                t[(r["phase"], r["mode"], int(r["M"]))][r["policy"]][int(r["round"])] = float(r["rank_max_ms"])
    return t


def med(x):
    return statistics.median(x.values())


def q(v, p):
    s = sorted(v)
    return s[max(0, min(len(s) - 1, int(p * len(s))))]


cal, ev = load("cal"), load("eval")
L = 4
tables = {m: json.load(open(os.path.join(d, f"table_{m}.json"))) for m in ("steady", "gpu")}
ops = [("ag", "1280x8192"), ("rs", "8192x1024"), ("ag", "7168x8192"), ("rs", "8192x3584")]


def op_time(tbl, kind, key, M, path):
    ent = tbl[kind][key]
    i = ent["M"].index(M)
    return dict((p, t) for p, t in ent["rank"][i]).get(path)


def pred(mode, M, policy):
    tbl = tables["gpu" if mode == "graph" else "steady"]
    tot = 0.0
    for kind, key in ops:
        ent = tbl[kind][key]
        if M not in ent["M"]:
            return None
        i = ent["M"].index(M)
        times = dict((p, t) for p, t in ent["rank"][i])
        if policy == "sp_flux":
            p = "flux"
        elif policy == "sp_rsflux":
            p = "flux" if kind == "rs" else "nccl"
        elif policy == "sp_dispatch":
            p = ent["rank"][i][0][0]
            if mode == "graph" and kind == "ag" and p == "flux":
                p = ent["rank"][i][1][0]
        else:
            p = "nccl"
        tot += times[p] - times["nccl"]
    return L * tot


rows = []
print(f"{'phase':<8}{'mode':<6}{'M':>6} | {'tp_ar':>7} {'sp_nccl':>8} {'sp_flux':>8} {'sp_rsfl':>8} {'sp_disp':>8} | "
      f"{'auto':>7} (layout) | auto vs tp_ar | auto vs sp_flux | auto vs oracle")
for key in sorted(ev, key=lambda k: (k[0], k[1], k[2])):
    phase, mode, M = key
    e, c = ev[key], cal.get(key)
    m = {p: med(v) for p, v in e.items()}
    layout = "tp_ar" if c and med(c["tp_ar"]) < med(c["sp_dispatch"]) else "sp_dispatch"
    auto = m[layout]
    oracle = min(m.values())
    fl = m.get("sp_flux")

    def rel(a, b):
        return (b - a) / b * 100 if b else float("nan")
    rows.append(dict(phase=phase, mode=mode, M=M, **{p: round(v, 4) for p, v in m.items()}, auto=round(auto, 4),
                     auto_layout=layout, oracle=round(oracle, 4), oracle_policy=min(m, key=m.get),
                     auto_saving_vs_tp_ar_pct=round(rel(auto, m["tp_ar"]), 2),
                     auto_saving_vs_sp_flux_pct=round(rel(auto, fl), 2) if fl else "",
                     auto_regret_vs_oracle_pct=round((auto - oracle) / oracle * 100, 2)))
    print(f"{phase:<8}{mode:<6}{M:>6} | {m['tp_ar']:7.3f} {m['sp_nccl']:8.3f} {fl if fl else float('nan'):8.3f} "
          f"{m['sp_rsflux']:8.3f} {m['sp_dispatch']:8.3f} | {auto:7.3f} ({layout[:5]}) | "
          f"{rel(auto, m['tp_ar']):+6.1f}% | {rel(auto, fl) if fl else float('nan'):+6.1f}% | "
          f"{(auto - oracle) / oracle * 100:+5.1f}% [{min(m, key=m.get)}]")

print("\ntable check (eval run): predicted vs measured block difference to sp_nccl, ms")
chk = []
for key in sorted(ev):
    phase, mode, M = key
    e = ev[key]
    for pol in ("sp_flux", "sp_rsflux", "sp_dispatch"):
        if pol not in e:
            continue
        pr = pred(mode, M, pol)
        rounds = sorted(set(e[pol]) & set(e["sp_nccl"]))
        meas = statistics.median(e[pol][r] - e["sp_nccl"][r] for r in rounds)
        if pr is None:
            continue
        chk.append((phase, mode, M, pol, pr, meas))
        print(f"  {phase:<8}{mode:<6}{M:>6} {pol:<12} predicted {pr:+8.3f}  measured {meas:+8.3f}")
with open(os.path.join(d, "block_summary.csv"), "w", newline="") as f:
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("phase", "mode", "M"), k))
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    w.writerows(rows)
with open(os.path.join(d, "table_check.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["phase", "mode", "M", "policy", "predicted_ms", "measured_ms"])
    w.writerows([[a, b, c_, p, f"{x:.4f}", f"{y:.4f}"] for a, b, c_, p, x, y in chk])
# trace aggregates (uniform over measured M buckets)
print("\ntrace aggregates (sum of block time over M buckets; decode=graph, prefill=eager):")
for name, sel in [("decode (graph, M<=512)", lambda r: r["phase"] == "decode" and r["mode"] == "graph"),
                  ("decode (eager, M<=512)", lambda r: r["phase"] == "decode" and r["mode"] == "eager"),
                  ("prefill (eager, M>=1024)", lambda r: r["phase"] == "prefill")]:
    rs = [r for r in rows if sel(r)]
    if not rs:
        continue
    tot = {p: sum(r[p] for r in rs if p in r) for p in ("tp_ar", "sp_flux", "auto", "oracle")}
    n_fl = sum(1 for r in rs if "sp_flux" in r)
    s = f"  {name}: auto {tot['auto']:.3f} ms | vs vLLM default {(tot['tp_ar'] - tot['auto']) / tot['tp_ar'] * 100:+.1f}%"
    if n_fl == len(rs):
        s += f" | vs always-Flux {(tot['sp_flux'] - tot['auto']) / tot['sp_flux'] * 100:+.1f}%"
    s += f" | regret vs oracle {(tot['auto'] - tot['oracle']) / tot['oracle'] * 100:+.2f}%"
    print(s)
