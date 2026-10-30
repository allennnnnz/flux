"""Analyze FS step 1 (results/topic/fs1_v1/fs1_raw.json) against the pre-registered rule
(reports/20261030_fs_validation_plan.md).

ideal = max(ag, min(gemm_only, cublas)); default gap = fused_default / ideal; best gap = fused_best / ideal;
config recovery = (fused_default - fused_best) / (fused_default - ideal).
fused_best = TopK=1 time from AGKernel.profiling() (all 64 compiled bf16 sm80 AG configs, same timing method).
"""
import json
import re
import statistics
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "ws/hetero-proxy/results/topic/fs1_v1/fs1_raw.json"
recs = json.load(open(path))
out = ["| layer | M | AG | GEMM (min) | ideal | fused default | fused best | default/ideal | best/ideal | config recovers | best config | clock meas/prof |",
       "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
small, large = [], []
for r in recs:
    m = re.search(r"TopK=1 \(([\d.]+) ms\): (.*)", r.get("prof_top5", ""))
    best = float(m.group(1)) if m else float("nan")
    hp = m.group(2) if m else ""
    cfg = ",".join(x for x in [
        (re.search(r"streamk_mode=Streamk(\w+)", hp) or [None, "?"])[1],
        (re.search(r"tile_shape=\(([\d,]+)\)", hp) or [None, "?"])[1].replace(",", "x"),
        (re.search(r"RasterAlong(\w)", hp) or [None, "?"])[1]])
    g = min(r["gemm_only"], r["cublas"])
    ideal = max(r["ag"], g)
    d, b = r["fused_default"] / ideal, best / ideal
    rec = (r["fused_default"] - best) / (r["fused_default"] - ideal) if r["fused_default"] > ideal else float("nan")
    (small if r["M"] <= 2048 else large).append((d, b, rec))
    out.append(f"| {r['layer']} | {r['M']} | {r['ag']:.3f} | {g:.3f} | {ideal:.3f} | {r['fused_default']:.3f} | {best:.3f} | "
               f"{d:.2f}x | {b:.2f}x | {rec * 100:.0f}% | {cfg} | {r.get('clock_meas_min')}/{r.get('clock_prof_min')} |")
for tag, xs in (("M <= 2048", small), ("M >= 4096", large)):
    if xs:
        out.append(f"\n{tag}: n={len(xs)} default/ideal median {statistics.median(x[0] for x in xs):.2f}x; "
                   f"best/ideal median {statistics.median(x[1] for x in xs):.2f}x (max {max(x[1] for x in xs):.2f}x); "
                   f"config recovers median {statistics.median(x[2] for x in xs if x[2] == x[2]) * 100:.0f}%")
txt = "\n".join(out)
print(txt)
open(path.replace("fs1_raw.json", "fs1_table.md"), "w").write(txt + "\n")
