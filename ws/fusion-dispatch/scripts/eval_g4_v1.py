################################################################################
# fusion-dispatch G4: score the pre-registered decisions (results/g4_predictions/decisions_g4.csv,
# committed before the oracle maps) against the oracle maps (results/g4_map). Nothing is refitted.
# Policies: on / off / thr512 / pred_g3 (G3 method) / pred_g4m (model only, G4 profiles) /
# g4 (model comm + overlap, measured GEMMs) / g4+probe (g4 plus the multi-GPU path probes).
# Also reports: path-time MAPE of g4, the measurement cost of each method (GEMM probes, path probes)
# against the cost of the full map, and a self-consistency check of the GEMM probes against the
# GEMM components measured inside the oracle maps (CLAUDE.md 5.1.3).
# Usage: python3 eval_g4_v1.py
################################################################################
import csv
import os
import re
import statistics
import subprocess
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(WS))
R = os.path.join(WS, "results")
POL = [("on", None), ("off", None), ("thr512", "pick_thr512"), ("pred_g3", "pick_pred_g3"),
       ("pred_g4m", "pick_pred_g4m"), ("g4", "pick_g4"), ("g4+probe", "pick_g4_probe"), ("g4+probe3", "pick_g4_probe3")]
ITEM = {"A": "A_fused", "B": "B_nccl_cublas", "C": "C_fluxag_cublas", "D": "D_fluxag_fluxgemm"}


def measured():
    meas, items = {}, {}
    for d in sorted(os.listdir(os.path.join(R, "g4_map"))):
        if not d.startswith("tp"):
            continue
        tp, path = int(d[2:]), os.path.join(R, "g4_map", d)
        subprocess.run([sys.executable, os.path.join(HERE, "analyze_v1.py"), path, "--out", os.path.join(path, "summary")],
                       check=True, capture_output=True)
        for r in csv.DictReader(open(os.path.join(path, "summary_items.csv"))):
            items.setdefault((tp, r["layer"], int(r["M"]), r["mode"]), {})[r["item"]] = float(r["median_ms"])
        for r in csv.DictReader(open(os.path.join(path, "summary_points.csv"))):
            k = (tp, r["layer"], int(r["M"]), r["mode"])
            arms = {x: items[k][ITEM[x]] for x in "ABCD" if ITEM[x] in items[k]}
            meas[k] = {"arms": arms, "verdict": r["verdict"]}
    return meas, items


def wall(path, start_pat, end_pat):
    t = open(path).read()
    a = re.search(start_pat + r" (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)", t)
    b = re.search(end_pat + r" (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)", t)
    if not (a and b):
        return None
    import datetime
    f = "%Y-%m-%d %H:%M:%S"
    return (datetime.datetime.strptime(b.group(1), f) - datetime.datetime.strptime(a.group(1), f)).total_seconds()


def main():
    meas, items = measured()
    dec = list(csv.DictReader(open(os.path.join(R, "g4_predictions", "decisions_g4.csv"))))
    pred = {(r["tp"], r["layer"], r["M"], r["mode"]): r for r in csv.DictReader(open(os.path.join(R, "g4_predictions", "predictions_g4.csv")))}
    out = open(os.path.join(R, "g4_map", "eval_g4_log.txt"), "w")

    def say(*x):
        s = " ".join(str(v) for v in x)
        print(s, flush=True)
        out.write(s + "\n")
    pts = []
    for r in dec:
        k = (int(r["tp"]), r["layer"], int(r["M"]), r["mode"])
        if k not in meas:
            continue
        m = meas[k]["arms"]
        ch = {name: ("A" if name == "on" else "B" if name == "off" else r[col]) for name, col in POL}
        pts.append({"k": k, "side": r["side"], "model": r["model"], "meas": m, "ch": ch, "verdict": meas[k]["verdict"],
                    "probed": bool(r["probe_medians"]), "probed3": r["probed3"] == "1",
                    "p": pred[(r["tp"], r["layer"], r["M"], r["mode"])]})
    say(f"G4: {len(pts)} oracle points matched to {len(dec)} pre-registered decisions")

    def reg(P, name):
        num = sum(p["meas"][p["ch"][name]] - min(p["meas"].values()) for p in P)
        den = sum(min(p["meas"].values()) for p in P)
        worst = max(P, key=lambda p: p["meas"][p["ch"][name]] / min(p["meas"].values()))
        nw = sum(p["meas"][p["ch"][name]] > 1.05 * min(p["meas"].values()) for p in P)
        return num / den, worst["meas"][worst["ch"][name]] / min(worst["meas"].values()) - 1, worst["k"], nw
    groups = [("all", lambda p: True), ("TP=8 Qwen2.5-32B", lambda p: p["k"][0] == 8),
              ("TP=4 Qwen2.5-32B", lambda p: p["k"][0] == 4 and p["model"] == "qwen2.5-32b"),
              ("TP=4 Llama-3-8B", lambda p: p["model"] == "llama3-8b"),
              ("gpu", lambda p: p["k"][3] == "gpu"), ("steady", lambda p: p["k"][3] == "steady")]
    summary = []
    for side in ("ag", "rs", "both"):
        for gname, gf in groups:
            P = [p for p in pts if (side == "both" or p["side"] == side) and gf(p)]
            if not P:
                continue
            say(f"\n[{side} {gname}] {len(P)} points  probed {sum(p['probed'] for p in P)} (eps rule), "
                f"{sum(p['probed3'] for p in P)} (3% variant)")
            say(f"    {'policy':<10}{'regret':>9}{'decode':>9}{'prefill':>9}{'worst':>9}  {'point':<26}{'>5%':>5}")
            for name, _ in POL:
                rg, wr, wk, nw = reg(P, name)
                dec_ = [p for p in P if p["k"][2] <= 512]
                pre_ = [p for p in P if p["k"][2] >= 1024]
                say(f"    {name:<10}{rg * 100:8.2f}%{reg(dec_, name)[0] * 100:8.2f}%{reg(pre_, name)[0] * 100:8.2f}%"
                    f"{wr * 100:8.1f}%  {str(wk):<26}{nw:>5}")
                summary.append([side, gname, name, f"{rg:.5f}", f"{reg(dec_, name)[0]:.5f}", f"{reg(pre_, name)[0]:.5f}",
                                f"{wr:.4f}", str(wk), nw, sum(p["probed"] for p in P), len(P)])
    # path-time accuracy of the g4 predictions
    say("\n[g4 path-time MAPE vs oracle]")
    for side in ("ag", "rs"):
        P = [p for p in pts if p["side"] == side]
        for x in ("A", "B", "C", "D"):
            e = [abs(float(p["p"][f"g4_{x}"]) / p["meas"][x] - 1) for p in P if x in p["meas"] and p["p"][f"g4_{x}"]]
            if e:
                say(f"    {side} {x}: MAPE {statistics.mean(e) * 100:.1f}%  median {statistics.median(e) * 100:.1f}%  "
                    f"p90 {sorted(e)[int(0.9 * len(e))] * 100:.1f}%  (n={len(e)})")
        near = [p for p in P if abs(p["meas"]["A"] / min(v for k, v in p["meas"].items() if k != "A") - 1) < 0.15]
        e = [abs(float(p["p"][f"g4_{x}"]) / p["meas"][x] - 1) for p in near for x in p["meas"] if p["p"][f"g4_{x}"]]
        if e:
            say(f"    {side} near crossover (|A/best_off - 1| < 15%, {len(near)} pts): MAPE {statistics.mean(e) * 100:.1f}%")
    # GEMM probe vs GEMM components inside the oracle runs (same item, different run)
    say("\n[self-consistency: GEMM probe (50 / 20 rounds) vs the same GEMM inside the oracle map (200 rounds)]")
    g = {}
    for f in os.listdir(os.path.join(R, "g4_gemm")):
        if f.startswith("gemm_") and f.endswith(".csv"):
            tp = int(f.split("_tp")[1].split(".")[0])
            model = f[5:].split("_tp")[0]
            for r in csv.DictReader(open(os.path.join(R, "g4_gemm", f))):
                g[(tp, model, r["layer"], int(r["M"]), r["mode"], r["item"])] = float(r["median_ms"])
    lname = {"Q32-QKV": "qkv", "Q32-GU": "gate_up", "Q32-O": "o", "Q32-down": "down",
             "L8-QKV": "qkv", "L8-GU": "gate_up", "L8-O": "o", "L8-down": "down"}
    dev = defaultdict(list)
    for p in pts:
        tp, layer, M, mode = p["k"]
        it = items[p["k"]]
        for probe_item, map_item in (("c_cublas", "c_cublas"), ("c_fluxgemm", "c_fluxgemm"), ("c_fluxgemm_only", "c_fluxgemm")):
            v = g.get((tp, p["model"], lname[layer], M, mode, probe_item))
            if v and map_item in it and not (p["side"] == "rs" and probe_item == "c_fluxgemm"):
                dev[(p["side"], probe_item, mode)].append(v / it[map_item] - 1)
    for k, v in sorted(dev.items()):
        say(f"    {k[0]} {k[1]:<16} {k[2]:<6} median dev {statistics.median(v) * 100:+.1f}%  "
            f"max |dev| {max(abs(x) for x in v) * 100:.1f}%  n={len(v)}")
    # measurement cost
    say("\n[measurement cost, wall clock incl. process start-up]")
    c_gemm = wall(os.path.join(R, "g4_gemm", "run_log.txt"), r"\[G4-gemm\] start", r"\[G4-gemm\] done")
    c_probe = wall(os.path.join(R, "g4_probes", "run_log.txt"), r"\[G4-probe\] start", r"\[G4-probe\] done")
    c_map = wall(os.path.join(R, "g4_map", "run_log.txt"), r"\[G4-map\] start", r"\[G4-map\] done")
    say(f"    GEMM probes {c_gemm}s   path probes {c_probe}s   full oracle map (what a table needs) {c_map}s")
    if c_gemm and c_map:
        say(f"    g4 cost / table cost = {c_gemm / c_map * 100:.1f}%   g4+probe = {(c_gemm + (c_probe or 0)) / c_map * 100:.1f}%")
    n_probe, n3 = sum(p["probed"] for p in pts), sum(p["probed3"] for p in pts)
    say(f"    path probes: {n_probe} of {len(pts)} table points ({n_probe / len(pts) * 100:.0f}%); 3% variant {n3} "
        f"({n3 / len(pts) * 100:.0f}%)")
    for side in ("ag", "rs"):
        for tp in (8, 4):
            for lay in sorted({p["k"][1] for p in pts if p["side"] == side and p["k"][0] == tp}):
                for md in ("gpu", "steady"):
                    L = sorted((p for p in pts if p["k"][0] == tp and p["k"][1] == lay and p["k"][3] == md), key=lambda p: p["k"][2])
                    say(f"  picks tp{tp} {lay:<9} {md:<6} M:       " + " ".join(f"{p['k'][2]:>5}" for p in L))
                    say(f"  {'':<26} oracle:  " + " ".join(f"{min(p['meas'], key=p['meas'].get):>5}" for p in L))
                    for name in ("pred_g3", "g4", "g4+probe", "g4+probe3"):
                        say(f"  {'':<26} {name:<8} " + " ".join(f"{p['ch'][name]:>5}" for p in L))
    with open(os.path.join(R, "g4_map", "eval_g4_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["side", "group", "policy", "regret", "regret_decode", "regret_prefill", "worst_rel", "worst_point",
                    "n_wrong_gt5pct", "probed", "n_points"])
        w.writerows(summary)
    say(f"\nwrote {os.path.relpath(os.path.join(R, 'g4_map'), REPO)}/eval_g4_log.txt, eval_g4_summary.csv")
    out.close()


if __name__ == "__main__":
    main()
