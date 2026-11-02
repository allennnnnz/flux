# fusion-dispatch E4a3: pool the three per-config evaluations (21 fresh points) and apply the pre-registered E1-E3
# (header of run_e4a3_v1.sh). Pooled regret = sum(chosen time) / sum(best time) - 1 over all points, as in the E4a2
# report. Reads the oracle medians through e4a_layout_v2.medians (same files the per-config eval used).
# Usage: E4A_OUT_DIR=<abs results dir> python3 e4a3_pool_v1.py   (writes pooled_new.txt there)
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e4a_layout_v2 as e  # noqa: E402

CONFIGS = [("tp8x2n3", "qwen2.5-32b", 8), ("tp4x2n3", "qwen2.5-32b", 4), ("tp4x2n3", "llama3-8b", 4)]
POLS = ["decider", "decider_model", "R1", "R1p", "R2", "R3"]
pts = []
for tag, model, tp in CONFIGS:
    pred = {(r["phase"], int(r["M"])): r for r in csv.DictReader(open(os.path.join(e.OUT, f"predictions_{tag}_{model}_tp{tp}_new.csv")))}
    fin = {(r["phase"], int(r["M"])): r["decider"] for r in csv.DictReader(open(os.path.join(e.OUT, f"decisions_{tag}_{model}_tp{tp}_new.csv")))}
    med = e.medians(os.path.join(e.OUT, f"oracle_{tag}_{model}_tp{tp}_*", "raw_*.csv"))
    for k, m in med.items():
        p = pred[k]
        ch = {"decider": fin[k], "decider_model": p["decider_model"], "R1": p["R1"], "R1p": p["R1p"], "R2": p["R2"], "R3": p["R3"]}
        pts.append(dict(cfg=f"{model} TP{tp}", k=k, t=m, best=min(e.DEPLOY, key=lambda x: m[x]), ch=ch))
L = []
L.append(f"E4a3 pooled over {len(pts)} fresh points (3 configs x 7), changed cross-node env; oracle 200 rounds, both nodes CLEAN")
L.append(f"{'policy':<15}{'regret':>8}{'saving vs vLLM':>16}{'worst point':>13}  where{'':<26}right")
res = {}
for g, gf in (("all", lambda p: True), ("decode", lambda p: p["k"][0] == "decode"), ("prefill", lambda p: p["k"][0] == "prefill")):
    P = [p for p in pts if gf(p)]
    o, v = sum(p["t"][p["best"]] for p in P), sum(p["t"]["tp_ar_vllm"] for p in P)
    L.append(f"[{g}] {len(P)} points")
    for pol in POLS:
        t = sum(p["t"][p["ch"][pol]] for p in P)
        w = max(P, key=lambda p: p["t"][p["ch"][pol]] / p["t"][p["best"]])
        wr = w["t"][w["ch"][pol]] / w["t"][w["best"]] - 1
        right = sum(p["ch"][pol] == p["best"] for p in P)
        res[(g, pol)] = (t / o - 1, wr, right, len(P))
        L.append(f"  {pol:<13}{(t / o - 1) * 100:7.2f}%{(1 - t / v) * 100:14.1f}%{wr * 100:12.1f}%  {w['cfg']} {w['k'][0]} M={w['k'][1]:<6}{right:>6}/{len(P)}")
d = res[("all", "decider")]
e1 = d[0] <= 0.02 and d[2] >= 17
e2 = any(res[("all", r)][0] >= 0.05 or res[("all", r)][1] >= 0.20 for r in ("R1", "R1p", "R2", "R3"))
e3 = all(d[0] <= res[("all", pol)][0] for pol in ("R1", "R1p", "R2", "R3"))
L.append("")
L.append(f"E1 decider regret <= 2% and right >= 17/21: {d[0] * 100:.2f}%, {d[2]}/{d[3]} -> {'HOLDS' if e1 else 'FAILS'}")
L.append("E2 some simple rule regret >= 5% or worst >= 20%: " + ", ".join(f"{r} {res[('all', r)][0] * 100:.2f}% / worst {res[('all', r)][1] * 100:.1f}%" for r in ("R1", "R1p", "R2", "R3")) + f" -> {'HOLDS' if e2 else 'FAILS'}")
L.append(f"E3 decider lowest pooled regret: {'HOLDS' if e3 else 'FAILS'}")
L.append(f"Success (E1 and E2): {'YES' if e1 and e2 else 'NO'}")
out = "\n".join(L)
print(out)
open(os.path.join(e.OUT, "pooled_new.txt"), "w").write(out + "\n")
