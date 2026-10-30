"""Analyze FS1b (results/topic/fs1b_v1/fs1b_raw.json) with FS1's best-config times (fs1_v1/fs1_raw.json).

ideal = max(ag, gemm_only)  (here the concurrent run uses gemm_only, so the ideal uses gemm_only, not cuBLAS)
concurrent / ideal ~ 1 -> AG and GEMM do not slow each other; the fused gap is the fused kernel's waiting (scheduling).
concurrent / fused ~ 1 -> the gap is resource contention.
share of the fused gap explained by contention = (concurrent - ideal) / (fused_default - ideal).
Pre-registered (plan addendum): continue only if median concurrent / ideal <= 1.15 at M <= 2048.
"""
import json
import re
import statistics
import sys

R = "ws/hetero-proxy/results/topic"
b = json.load(open(f"{R}/fs1b_v1/fs1b_raw.json"))
a = {(r["layer"], r["M"]): r for r in json.load(open(f"{R}/fs1_v1/fs1_raw.json"))}
out = ["| layer | M | AG | gemm_only | ideal | concurrent | fused default | fused best (FS1) | concurrent/ideal | fused/ideal | contention share of gap | clock mode |",
       "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
small, large = [], []
for r in b:
    ideal = max(r["ag"], r["gemm_only"])
    m = re.search(r"TopK=1 \(([\d.]+) ms\)", a.get((r["layer"], r["M"]), {}).get("prof_top5", ""))
    best = float(m.group(1)) if m else float("nan")
    c, f = r["concurrent"] / ideal, r["fused_default"] / ideal
    share = (r["concurrent"] - ideal) / (r["fused_default"] - ideal) if r["fused_default"] > ideal else float("nan")
    (small if r["M"] <= 2048 else large).append((c, f, share))
    out.append(f"| {r['layer']} | {r['M']} | {r['ag']:.3f} | {r['gemm_only']:.3f} | {ideal:.3f} | {r['concurrent']:.3f} | "
               f"{r['fused_default']:.3f} | {best:.3f} | {c:.2f}x | {f:.2f}x | {share * 100:.0f}% | {r.get('clock_meas_mode')} |")
for tag, xs in (("M <= 2048", small), ("M >= 4096", large)):
    if xs:
        out.append(f"\n{tag}: n={len(xs)} concurrent/ideal median {statistics.median(x[0] for x in xs):.2f}x "
                   f"(max {max(x[0] for x in xs):.2f}x); fused/ideal median {statistics.median(x[1] for x in xs):.2f}x; "
                   f"contention share of gap median {statistics.median(x[2] for x in xs if x[2] == x[2]) * 100:.0f}%")
txt = "\n".join(out)
print(txt)
open(f"{R}/fs1b_v1/fs1b_table.md", "w").write(txt + "\n")
