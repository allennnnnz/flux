################################################################################
# fusion-dispatch G3: compare the PRE-REGISTERED predictions (results/g3_predictions/predictions.csv,
# committed before measuring) with the measured G3 op maps (run_g3_map_v1.sh). Nothing is refitted.
# Per (TP, layer, M, mode) the oracle is the best measured path; regret as in policy_eval_v1.py.
# Policies: on / off / thr512 / pred / hyb (probe the top-2 when the predicted margin < eps) /
# hyb+pcie (also probe pick vs best non-Flux-GEMM path when the pick runs a PCIe-tuned config),
# hyb5+pcie (same with eps = 5%, pre-registered second variant); the probe sets come from the
# predictions file, the probe outcome from the measurement.
# Steps: analyze_v1.py on every results/g3_map/tp<TP>/ dir (summary_*), then join and score.
# Usage: python3 eval_g3_v1.py [--map ws/fusion-dispatch/results/g3_map] [--pred .../g3_predictions]
################################################################################
import argparse
import csv
import os
import statistics
import subprocess
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(WS))
ARMS = {"ag": ("A", "B", "C", "D"), "rs": ("A", "B")}
POLICIES = ["on", "off", "thr512", "pred", "hyb", "hyb+pcie", "hyb5+pcie"]


def load_measured(map_dir):
    meas = {}
    for d in sorted(os.listdir(map_dir)):
        if not d.startswith("tp"):
            continue
        tp, path = int(d[2:]), os.path.join(map_dir, d)
        subprocess.run([sys.executable, os.path.join(HERE, "analyze_v1.py"), path, "--out", os.path.join(path, "summary")],
                       check=True, capture_output=True)
        items = defaultdict(dict)
        for r in csv.DictReader(open(os.path.join(path, "summary_items.csv"))):
            items[(r["layer"], int(r["M"]), r["mode"])][r["item"]] = float(r["median_ms"])
        for r in csv.DictReader(open(os.path.join(path, "summary_points.csv"))):
            key = (tp, r["layer"], int(r["M"]), r["mode"])
            it = items[(r["layer"], int(r["M"]), r["mode"])]
            arms = {"A": float(r["t_fused"]), "B": float(r["t_B"])}
            for x, c in (("C", "t_C"), ("D", "t_D")):
                if r[c]:
                    arms[x] = float(r[c])
            meas[key] = {"arms": arms, "verdict": r["verdict"], "items": it, "clock": r["modal_sm_clock"]}
    return meas


def regret(points):
    num = sum(p["meas"][p["choice"]] - min(p["meas"].values()) for p in points)
    den = sum(min(p["meas"].values()) for p in points)
    worst = max(points, key=lambda p: p["meas"][p["choice"]] / min(p["meas"].values()))
    wrel = worst["meas"][worst["choice"]] / min(worst["meas"].values()) - 1
    nwrong = sum(p["meas"][p["choice"]] > 1.05 * min(p["meas"].values()) for p in points)
    return num / den, wrel, f"{worst['layer']}@tp{worst['tp']}@{worst['M']}", nwrong


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default=os.path.join(WS, "results", "g3_map"))
    ap.add_argument("--pred", default=os.path.join(WS, "results", "g3_predictions"))
    a = ap.parse_args()
    meas = load_measured(a.map)
    pts = []
    for r in csv.DictReader(open(os.path.join(a.pred, "predictions.csv"))):
        key = (int(r["tp"]), r["layer"], int(r["M"]), r["mode"])
        if key not in meas:
            continue
        side = r["side"]
        m = meas[key]
        arms = [x for x in ARMS[side] if x in m["arms"]]
        pred = {x: float(r[f"pred_{x}"]) for x in arms}
        rank = sorted(arms, key=lambda x: pred[x])
        probe_margin = {rank[0], rank[1]} if r["probe_margin"] == "1" else set()
        probe_all = set(r["probe_arms"].split("|")) if r["probe_arms"] else set()
        probe5 = set(r["probe_arms_eps5"].split("|")) if r["probe_arms_eps5"] else set()
        mm = {x: m["arms"][x] for x in arms}
        choice = {"on": "A", "off": "B", "thr512": r["thr512_pick"], "pred": r["pred_pick"],
                  "hyb": min(probe_margin, key=lambda x: mm[x]) if probe_margin else r["pred_pick"],
                  "hyb+pcie": min(probe_all, key=lambda x: mm[x]) if probe_all else r["pred_pick"],
                  "hyb5+pcie": min(probe5, key=lambda x: mm[x]) if probe5 else r["pred_pick"]}
        pts.append({"tp": key[0], "layer": key[1], "M": key[2], "mode": key[3], "side": side, "meas": mm, "pred": pred,
                    "choices": choice, "verdict": m["verdict"], "items": m["items"], "pcie": r["probe_pcie"] == "1",
                    "probe_margin": bool(probe_margin), "probe_all": bool(probe_all), "probe5": bool(probe5), "row": r})
    out = open(os.path.join(a.map, "eval_g3_log.txt"), "w")

    def say(*x):
        s = " ".join(str(v) for v in x)
        print(s, flush=True)
        out.write(s + "\n")
    n_pred = sum(1 for _ in csv.DictReader(open(os.path.join(a.pred, "predictions.csv"))))
    say(f"G3: {len(pts)} measured points matched to {n_pred} pre-registered predictions")
    summary = []
    groups = [("all", lambda p: True)] + [(f"TP={tp}", (lambda tp: lambda p: p["tp"] == tp)(tp)) for tp in (8, 4, 2)] + \
             [(f"mode={md}", (lambda md: lambda p: p["mode"] == md)(md)) for md in ("gpu", "steady")]
    for side in ("ag", "rs"):
        for gname, gf in groups:
            P = [p for p in pts if p["side"] == side and gf(p)]
            if not P:
                continue
            err = {x: [abs(p["pred"][x] / p["meas"][x] - 1) for p in P if x in p["meas"]] for x in ARMS[side]}
            nt = [p for p in P if p["verdict"] in ("off", "fused")]
            acc = sum((p["choices"]["pred"] == "A") == (p["verdict"] == "fused") for p in nt)
            say(f"\n[{side} {gname}] {len(P)} points  arm MAPE " +
                "  ".join(f"{x} {statistics.mean(v) * 100:.1f}%" for x, v in err.items() if v) +
                f"  fused-vs-off accuracy (non-tie) {acc}/{len(nt)}")
            say(f"    {'policy':<9}{'regret':>9}{'decode':>9}{'prefill':>9}{'worst':>10}  {'point':<22}{'>5%':>5}{'probes':>9}")
            for pol in POLICIES:
                ch = [dict(p, choice=p["choices"][pol]) for p in P]
                rg, wrel, wpt, nw = regret(ch)
                dec = [c for c in ch if c["M"] <= 512]
                pre = [c for c in ch if c["M"] >= 1024]
                probes = {"hyb": sum(p["probe_margin"] for p in P), "hyb+pcie": sum(p["probe_all"] for p in P),
                          "hyb5+pcie": sum(p["probe5"] for p in P)}.get(pol, 0)
                say(f"    {pol:<9}{rg * 100:8.2f}%{regret(dec)[0] * 100:8.2f}%{regret(pre)[0] * 100:8.2f}%{wrel * 100:9.1f}%  "
                    f"{wpt:<22}{nw:>5}{probes:>5}/{len(P)}")
                summary.append([side, gname, pol, f"{rg:.5f}", f"{regret(dec)[0]:.5f}", f"{regret(pre)[0]:.5f}",
                                f"{wrel:.4f}", wpt, nw, probes, len(P)])
    # PCIe-config rule: was the flagged Flux pick really worse than the best path without a Flux GEMM?
    say("\n[PCIe-tuned config rule, pre-registered flags]")
    for p in pts:
        if p["pcie"]:
            pick = p["row"]["pred_pick"]
            safe = min((x for x in p["meas"] if x not in ({"A", "D"} if p["side"] == "ag" else {"A"})), key=lambda x: p["meas"][x])
            say(f"    {p['layer']:<8} tp{p['tp']} M={p['M']:<5} {p['mode']:<6} pick {pick} meas {p['meas'][pick]:.4f}  "
                f"best non-Flux-GEMM {safe} {p['meas'][safe]:.4f}  -> {'cliff confirmed' if p['meas'][pick] > p['meas'][safe] else 'Flux was fine'}")
    for side in ("ag", "rs"):
        for tp in (8, 4, 2):
            for lay in sorted({p["layer"] for p in pts if p["side"] == side and p["tp"] == tp}):
                for md in ("gpu", "steady"):
                    L = sorted((p for p in pts if p["layer"] == lay and p["tp"] == tp and p["mode"] == md), key=lambda p: p["M"])
                    if not L:
                        continue
                    say(f"  picks tp{tp} {lay:<8} {md:<6} M:      " + " ".join(f"{p['M']:>5}" for p in L))
                    say(f"  {'':<24} oracle: " + " ".join(f"{min(p['meas'], key=p['meas'].get):>5}" for p in L))
                    say(f"  {'':<24} pred:   " + " ".join(f"{p['choices']['pred']:>5}" for p in L))
                    say(f"  {'':<24} verdict:" + " ".join(f"{p['verdict'][:5]:>5}" for p in L))
    with open(os.path.join(a.map, "eval_g3_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["side", "group", "policy", "regret", "regret_decode", "regret_prefill", "worst_rel", "worst_point",
                    "n_wrong_gt5pct", "probes", "n_points"])
        w.writerows(summary)
    with open(os.path.join(a.map, "eval_g3_points.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tp", "layer", "side", "M", "mode", "verdict", "oracle"] + [f"meas_{x}" for x in "ABCD"] +
                   [f"pred_{x}" for x in "ABCD"] + [f"choice_{p}" for p in POLICIES])
        for p in pts:
            w.writerow([p["tp"], p["layer"], p["side"], p["M"], p["mode"], p["verdict"], min(p["meas"], key=p["meas"].get)] +
                       [f"{p['meas'][x]:.4f}" if x in p["meas"] else "" for x in "ABCD"] +
                       [f"{p['pred'][x]:.4f}" if x in p["pred"] else "" for x in "ABCD"] + [p["choices"][q] for q in POLICIES])
    say(f"\nwrote {os.path.relpath(a.map, REPO)}/eval_g3_log.txt, eval_g3_summary.csv, eval_g3_points.csv")
    out.close()


if __name__ == "__main__":
    main()
