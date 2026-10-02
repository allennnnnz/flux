################################################################################
# Merge E1 points with a rerun: rerun (layer, M, mode) rows REPLACE the originals.
# Inputs are analyze_v1.py outputs (<prefix>_points.csv, <prefix>_items.csv).
# Adds columns: source (which run), min_kept (min kept rounds over items).
# Usage: python3 merge_points_v1.py <base_prefix> <rerun_prefix> <out.csv>
################################################################################
import csv
import sys

base, rerun, out = sys.argv[1:4]


def load(prefix, tag):
    kept = {}
    for r in csv.DictReader(open(prefix + "_items.csv")):
        k = (r["layer"], r["M"], r["mode"])
        kept[k] = min(kept.get(k, 10**9), int(r["n_kept"]))
    rows = {}
    for r in csv.DictReader(open(prefix + "_points.csv")):
        k = (r["layer"], r["M"], r["mode"])
        r["source"], r["min_kept"] = tag, kept[k]
        rows[k] = r
    return rows


rows = load(base, base.split("/")[-2])
rep = load(rerun, rerun.split("/")[-2])
rows.update(rep)
hdr = list(next(iter(rows.values())).keys())
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=hdr)
    w.writeheader()
    for k in sorted(rows, key=lambda k: (k[0], k[2], int(k[1]))):
        w.writerow(rows[k])
low = [(k, r["min_kept"]) for k, r in rows.items() if int(r["min_kept"]) < 200]
print(f"{len(rows)} points, {len(rep)} replaced from rerun; points with min_kept<200: {len(low)}")
for k, v in sorted(low):
    print(" ", k, v, rows[k]["verdict"], rows[k]["delta_med(A-best)"])
