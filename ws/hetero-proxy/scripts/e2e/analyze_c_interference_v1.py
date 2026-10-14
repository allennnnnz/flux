################################################################################
# hetero-proxy E2E C (2026-10-14): from the nsys export of c_interference_v1 (GPU0), every NCCL bf16 all-reduce kernel
# longer than 0.25 ms (the 80 MiB ones): its duration vs the fraction of it overlapped by PCIe copies (HtoD / DtoH)
# on the same GPU. If NCCL merely waited for the copies, overlapped kernels would not be longer; if they are, the
# PCIe traffic slows the NVLink all-reduce itself. Usage: python3 analyze_c_interference_v1.py interf.sqlite
################################################################################
import sqlite3
import statistics
import sys

c = sqlite3.connect(sys.argv[1])
names = dict(c.execute("select id, value from StringIds").fetchall())
ks = [(s, e) for s, e, n in c.execute("select start, end, shortName from CUPTI_ACTIVITY_KIND_KERNEL where deviceId=0")
      if "AllReduce_Sum_bf16" in names.get(n, "") and e - s > 250_000]
cp = [(s, e) for s, e, k in c.execute("select start, end, copyKind from CUPTI_ACTIVITY_KIND_MEMCPY where deviceId=0")
      if k in (1, 2) and e - s > 50_000]  # HtoD / DtoH, the 4 MiB background copies
buckets = {"0%": [], "1-50%": [], "50-100%": []}
for s, e in ks:
    cov = 0
    for a, b in cp:
        lo, hi = max(a, s), min(b, e)
        if hi > lo:
            cov += hi - lo
    frac = min(1.0, cov / (e - s))  # copies on two streams can overlap each other: clipped
    key = "0%" if frac == 0 else ("1-50%" if frac <= 0.5 else "50-100%")
    buckets[key].append((e - s) / 1e6)
for k, v in buckets.items():
    if v:
        print(f"[nccl-vs-pcie] PCIe copies overlapping the kernel {k:>8}: n={len(v):3d}  median kernel {statistics.median(v):.3f} ms"
              f"  (min {min(v):.3f}, max {max(v):.3f})")
