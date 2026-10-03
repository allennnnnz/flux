################################################################################
# fusion-dispatch G4-block: score the pre-registered layout + table decisions at BLOCK level.
# Input: results/g4_block_oracle/raw_<model>_tp<TP>_<phase>_<mode>_L4.csv (validate_block_v4.py, all
# policies, 200 rounds), results/g4_block_tables/layout_g4.csv (pre-registered layout per M).
# Deployable policies: tp_ar_vllm (vLLM 0.8.5 default), sp_nccl, sp_flux (eager only), sp_rsflux,
# sp_g3 (SP with the G3-method table), sp_g4 (SP with the G4 table).
#   auto_g4        pre-registered layout choice (block model + layout probes): tp_ar_vllm or sp_g4
#   auto_g4_model  the same without the layout probes (block model alone)
#   oracle         best measured deployable policy at that M
# Block regret = sum(t_chosen - t_oracle) / sum(t_oracle) over the M of a group (policy_eval_v1 style);
# also the saving of each policy against the vLLM default.
# Usage: python3 eval_g4_block_v1.py
################################################################################
import csv
import glob
import os
import statistics
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
R = os.path.join(WS, "results")
DEPLOY = ["tp_ar_vllm", "sp_nccl", "sp_flux", "sp_rsflux", "sp_g3", "sp_g4"]


def main():
    med = defaultdict(dict)  # (model, tp, phase, M) -> policy -> ms
    kept = {}
    for f in glob.glob(os.path.join(R, "g4_block_oracle", "raw_*.csv")):
        base = os.path.basename(f)[4:-4]
        model, tp = base.split("_tp")[0], int(base.split("_tp")[1].split("_")[0])
        v = defaultdict(list)
        for r in csv.DictReader(open(f)):
            if r["kept"] == "1":
                v[(r["phase"], int(r["M"]), r["policy"])].append(float(r["rank_max_ms"]))
        for (phase, M, pol), x in v.items():
            med[(model, tp, phase, M)][pol] = statistics.median(x)
            kept[(model, tp, phase, M)] = len(x)
    lay = {(r["model"], int(r["tp"]), r["phase"], int(r["M"])): r for r in
           csv.DictReader(open(os.path.join(R, "g4_block_tables", "layout_g4.csv")))}
    out = open(os.path.join(R, "g4_block_oracle", "eval_g4_block_log.txt"), "w")

    def say(*x):
        s = " ".join(str(v) for v in x)
        print(s, flush=True)
        out.write(s + "\n")
    rows = []
    for k in sorted(med):
        m = med[k]
        dep = {p: t for p, t in m.items() if p in DEPLOY}
        L = lay.get(k)
        if not L or "tp_ar_vllm" not in dep or "sp_g4" not in dep:
            continue
        ch = {p: p for p in dep}
        ch["auto_g4"] = L["choice_final"]
        ch["auto_g4_model"] = L["choice_model"]
        orc = min(dep, key=dep.get)
        rows.append({"k": k, "t": dep, "ch": ch, "orc": orc, "note": L["note"], "kept": kept[k]})
        say(f"{k[0]:<12} tp{k[1]} {k[2]:<7} M={k[3]:<5} kept {kept[k]:<4} " +
            "  ".join(f"{p}={t:.3f}" for p, t in sorted(dep.items())) +
            f"  | oracle {orc}  auto_g4 {ch['auto_g4']} (model {ch['auto_g4_model']}{'; ' + L['note'] if L['note'] else ''})")
    pols = DEPLOY + ["auto_g4_model", "auto_g4"]

    def score(P, pol):
        P = [p for p in P if p["ch"].get(pol) in p["t"]]
        if not P:
            return None
        num = sum(p["t"][p["ch"][pol]] - p["t"][p["orc"]] for p in P)
        den = sum(p["t"][p["orc"]] for p in P)
        base = sum(p["t"]["tp_ar_vllm"] for p in P)
        mine = sum(p["t"][p["ch"][pol]] for p in P)
        worst = max(P, key=lambda p: p["t"][p["ch"][pol]] / p["t"][p["orc"]])
        return num / den, 1 - mine / base, worst["t"][worst["ch"][pol]] / worst["t"][worst["orc"]] - 1, worst["k"], len(P)
    groups = [("all", lambda p: True), ("decode (graph)", lambda p: p["k"][2] == "decode"),
              ("prefill (eager)", lambda p: p["k"][2] == "prefill")]
    for model, tp in sorted({(p["k"][0], p["k"][1]) for p in rows}):
        groups.append((f"{model} tp{tp}", (lambda mo, t: lambda p: p["k"][0] == mo and p["k"][1] == t)(model, tp)))
    summary = []
    for gname, gf in groups:
        P = [p for p in rows if gf(p)]
        say(f"\n[{gname}] {len(P)} (config, phase, M) points")
        say(f"    {'policy':<14}{'regret':>9}{'saving vs vLLM':>16}{'worst':>9}  point")
        for pol in pols:
            s = score(P, pol)
            if s is None:
                continue
            say(f"    {pol:<14}{s[0] * 100:8.2f}%{s[1] * 100:15.1f}%{s[2] * 100:8.1f}%  {s[3]}  (n={s[4]})")
            summary.append([gname, pol, f"{s[0]:.5f}", f"{s[1]:.5f}", f"{s[2]:.4f}", str(s[3]), s[4]])
    with open(os.path.join(R, "g4_block_oracle", "eval_g4_block_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "policy", "block_regret", "saving_vs_vllm_default", "worst_rel", "worst_point", "n"])
        w.writerows(summary)
    out.close()


if __name__ == "__main__":
    main()
