################################################################################
# hetero-proxy E2E (2026-10-14): where does a CUDA-graph decode step of qwen_tp_v3 (config A) spend GPU0's time?
# Reads the nsys sqlite export (--cuda-graph-trace=node): per graph launch on GPU0 -> kernel count, kernel-busy time,
# GPU span, gap to the next step; top kernels by time. Node tracing inflates kernel times (~2x here), so use the
# shares and counts, not the absolute ms. Usage: python3 analyze_nsys_decode_v1.py <decode.sqlite> [n_steps_per_case]
################################################################################
import collections
import sqlite3
import statistics
import sys

c = sqlite3.connect(sys.argv[1])
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 16
names = dict(c.execute("select id, value from StringIds").fetchall())
gl_ids = tuple(k for k, v in names.items() if v.startswith("cudaGraphLaunch"))
launches = c.execute(f"select start, end, correlationId from CUPTI_ACTIVITY_KIND_RUNTIME where nameId in {gl_ids} "
                     "order by start").fetchall()
bycorr = collections.defaultdict(list)
for s, e, corr, nm in c.execute("select start, end, correlationId, shortName from CUPTI_ACTIVITY_KIND_KERNEL "
                                "where deviceId=0 and graphId is not null"):
    bycorr[corr].append((s, e, nm))
rows = [(s, e, min(k[0] for k in bycorr[corr]), max(k[1] for k in bycorr[corr]),
         sum(k[1] - k[0] for k in bycorr[corr]), bycorr[corr]) for s, e, corr in launches if bycorr.get(corr)]
print(f"graph launches on GPU0: {len(rows)}; analysing the last 2 x {steps} (the timed round)")
for label, part in (("case1", rows[-2 * steps:-steps]), ("case2", rows[-steps:])):
    span = [(r[3] - r[2]) / 1e3 for r in part]
    busy = [r[4] / 1e3 for r in part]
    gaps = [(part[i + 1][2] - part[i][3]) / 1e3 for i in range(len(part) - 1)]
    print(f"{label}: kernels/step {len(part[0][5])}, GPU span {statistics.median(span):.1f} us, kernel busy "
          f"{statistics.median(busy):.1f} us ({100 * statistics.median(busy) / statistics.median(span):.0f}%), "
          f"gap between steps {statistics.median(gaps):.1f} us")
    agg, cnt = collections.Counter(), collections.Counter()
    for r in part:
        for s, e, nm in r[5]:
            agg[names.get(nm, nm)] += (e - s) / 1e3
            cnt[names.get(nm, nm)] += 1
    tot = sum(agg.values())
    for nm, t in agg.most_common(10):
        print(f"   {100 * t / tot:5.1f}%  {cnt[nm] // len(part):5d} launches/step  {nm[:80]}")
