################################################################################
# fusion-dispatch: merge analyze_v1.py outputs from several runs; LATER prefixes override earlier
# ones for the same (layer, M, mode). v2 vs v1: any number of prefixes (V7 adds a third run).
# Also reports every point whose verdict changed versus the first prefix that contained it.
# Usage: python3 merge_points_v2.py <out.csv> <prefix1> <prefix2> [...]
################################################################################
import csv
import sys

out, prefixes = sys.argv[1], sys.argv[2:]
rows, first = {}, {}
for pre in prefixes:
    kept = {}
    for r in csv.DictReader(open(pre + "_items.csv")):
        k = (r["layer"], r["M"], r["mode"])
        kept[k] = min(kept.get(k, 10**9), int(r["n_kept"]))
    for r in csv.DictReader(open(pre + "_points.csv")):
        k = (r["layer"], r["M"], r["mode"])
        r["source"], r["min_kept"] = pre, kept[k]
        first.setdefault(k, r)
        rows[k] = r
hdr = list(next(iter(rows.values())).keys())
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=hdr)
    w.writeheader()
    for k in sorted(rows, key=lambda k: (k[0], k[2], int(k[1]))):
        w.writerow(rows[k])
low = sorted((k, r["min_kept"]) for k, r in rows.items() if int(r["min_kept"]) < 200)
changed = [(k, first[k]["verdict"], rows[k]["verdict"]) for k in rows if first[k]["verdict"] != rows[k]["verdict"]]
print(f"{len(rows)} points; still min_kept<200: {len(low)}; verdict changed vs first source: {len(changed)}")
for k, v in low:
    print("  low", k, v, rows[k]["verdict"])
for k, a, b in sorted(changed):
    print("  changed", k, a, "->", b, "delta", first[k]["delta_med(A-best)"], "->", rows[k]["delta_med(A-best)"])
