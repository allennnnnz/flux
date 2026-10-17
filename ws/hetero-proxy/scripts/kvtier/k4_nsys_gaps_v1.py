################################################################################
# hetero-proxy K4.1 mechanism (2026-10-17): in an nsys trace of k4_diag_v1 (eager, 8 ranks, rounds 3; the run whose
# wall-clock numbers are withdrawn - nsys overhead), are NCCL all-reduce kernels on GPU0 slower while a background
# 4 MiB PCIe copy overlaps them, or are the gaps between consecutive kernels longer? Kernel / copy times are CUPTI's.
# Usage: python3 k4_nsys_gaps_v1.py <diag.sqlite>   (nsys export --type sqlite of results/kvtier/k4/diag_v1_nsys)
################################################################################
import bisect
import sqlite3
import statistics as st
import sys

cur = sqlite3.connect(sys.argv[1]).cursor()
names = dict(cur.execute("select id, value from StringIds"))
ker = cur.execute("select start, end, shortName from CUPTI_ACTIVITY_KIND_KERNEL where deviceId=0").fetchall()
cp = sorted(cur.execute("select start, end, copyKind from CUPTI_ACTIVITY_KIND_MEMCPY where deviceId=0 and bytes=4194304"))
starts = [c[0] for c in cp]


def overlap(s, e):
    j, kinds = bisect.bisect_right(starts, e) - 1, set()
    while j >= 0 and cp[j][1] >= s and len(kinds) < 2:
        kinds.add(cp[j][2])
        j -= 1
    return kinds


nccl = sorted((s, e) for s, e, n in ker if "AllReduce" in names.get(n, ""))
print(f"GPU0: {len(ker)} kernels, {len(nccl)} NCCL all-reduce kernels, {len(cp)} background 4 MiB copies (kind 1 = H2D, 2 = D2H)")
grp = {"no copy": [], "H2D copy": [], "D2H copy": []}
for s, e in nccl:
    k = overlap(s, e)
    grp["no copy" if not k else ("H2D copy" if 1 in k else "D2H copy")].append((e - s) / 1e3)
for k, v in grp.items():
    small = [x for x in v if x < 200]
    print(f"  all-reduce kernel duration, overlapping {k:<8}: n={len(v):5d}  median {st.median(small):6.1f} us (kernels < 200 us)")
gaps = {"no copy": [], "copy in flight": []}
for (s0, e0), (s1, _) in zip(nccl, nccl[1:]):
    g = (s1 - e0) / 1e3
    if 0 < g < 500:
        gaps["copy in flight" if overlap(e0, s1) else "no copy"].append(g)
for k, v in gaps.items():
    print(f"  gap between consecutive all-reduce kernels, {k:<14}: n={len(v):5d}  median {st.median(v):6.1f} us")
