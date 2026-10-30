"""Analyze the V3 matrix (results/topic/v3/<case>[_LL128]/v3_rows.json) against the pre-registered rules
(reports/20261030_t_validation_plan.md).

copy_time = serial - none (serial runs every copy without overlap; act serial = D2H + H2D on the compute stream).
Lower bound LB (nothing slowed, copies perfectly hidden where possible):
  gimm: max(none, fwd_none + copy_time)   (gradients exist only after the forward pass)
  gdpu: max(none, copy_time)
  act : max(none, copy_time)               (loose; the act copies fit under compute in every case here)
loss(v) = (v - LB) / v ; recovered by window = (naive - window) / (naive - LB).
LL128 runs are separate processes: reported against their own 'none' and against the default run's 'none'.
"""
import json
import os
import sys

R = sys.argv[1] if len(sys.argv) > 1 else "ws/hetero-proxy/results/topic/v3"


def load(case):
    p = os.path.join(R, case, "v3_rows.json")
    return json.load(open(p)) if os.path.exists(p) else None


cases = sorted(c for c in os.listdir(R) if not c.endswith("_LL128") and os.path.isdir(os.path.join(R, c)))
out = ["| case | none ms | copy ms | LB ms | naive loss | serial loss | paced50 loss | paced25 loss | window loss | window recovers | "
       "LL128 none vs default none | LL128 naive loss (own LB) | LL128 window loss |",
       "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
js = {}
for c in cases:
    d = load(c)
    if d is None:
        continue
    r = {k: v["step_ms"] for k, v in d["rows"].items()}
    sc = d["meta"]["scenario"]
    copy = r["serial"] - r["none"]
    lb = max(r["none"], d["rows"]["none"]["fwd_ms"] + copy) if sc == "gimm" else max(r["none"], copy)
    loss = {k: (v - lb) / v for k, v in r.items()}
    rec = (r["naive"] - r["window"]) / (r["naive"] - lb) if r["naive"] - lb > 1e-6 else float("nan")
    ll = load(c + "_LL128")
    llcols = ["–", "–", "–"]
    ent = {"none": r["none"], "copy": copy, "LB": lb, "loss": loss, "window_recovers": rec,
           "coll_ms": {k: v["coll_ms"] for k, v in d["rows"].items()}}
    if ll:
        l = {k: v["step_ms"] for k, v in ll["rows"].items()}
        lb_ll = max(l["none"], d["rows"]["none"]["fwd_ms"] + copy) if sc == "gimm" else max(l["none"], copy)
        llcols = [f"x{l['none'] / r['none']:.3f}", f"{(l['naive'] - lb_ll) / l['naive'] * 100:.1f}%",
                  f"{(l['window'] - lb_ll) / l['window'] * 100:.1f}%"]
        ent["LL128"] = {"none": l["none"], "naive": l["naive"], "window": l["window"], "LB_own": lb_ll}
    js[c] = ent
    out.append(f"| {c} | {r['none']:.1f} | {copy:.1f} | {lb:.1f} | {loss['naive'] * 100:.1f}% | {loss['serial'] * 100:.1f}% | "
               f"{loss['paced50'] * 100:.1f}% | {loss['paced25'] * 100:.1f}% | {loss['window'] * 100:.1f}% | "
               f"{rec * 100:.0f}% | " + " | ".join(llcols) + " |")
txt = "\n".join(out)
print(txt)
open(os.path.join(R, "v3_table.md"), "w").write(txt + "\n")
json.dump(js, open(os.path.join(R, "v3_table.json"), "w"), indent=1)
