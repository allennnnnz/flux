################################################################################
# Parse an nsys sqlite export of nsys_probe_v1.py (design 5.5).
# GPU work is attributed to NVTX ranges via CUDA runtime/driver API
# correlationId (API call CPU time inside the range, same thread).
# Per iteration and device:
#   n_kernels, n_memcpy, n_memset       launches the item produced
#   span        first GPU op start -> last GPU op end
#   comm_end    end of the last shard-sized peer copy (classified by bytes, not kind)
#   gemm        the longest kernel (the GEMM) start/end/duration
#   tail        gemm_end - comm_end: GEMM time left after the last shard arrived
# If tail(A_fused) ~= gemm duration of c_fluxgemm, the GEMM did no useful work
# while communication was in flight (no overlap).
# Usage: python3 nsys_parse_v1.py <file.sqlite> --shard_bytes <M/8*K*2> [--device 0]
################################################################################
import argparse
import sqlite3
import statistics
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("sqlite")
ap.add_argument("--shard_bytes", type=int, required=True)
ap.add_argument("--device", type=int, default=-1, help="-1 = all devices")
a = ap.parse_args()
db = sqlite3.connect(a.sqlite)
cur = db.cursor()
tables = {r[0] for r in cur.execute("select name from sqlite_master where type='table'")}
strings = dict(cur.execute("select id, value from StringIds"))

nvtx = cur.execute("select start, end, coalesce(text, ''), textId, globalTid from NVTX_EVENTS "
                   "where end is not null").fetchall()
ranges = []
for s, e, text, tid_text, gtid in nvtx:
    t = text or strings.get(tid_text, "")
    if "#" in t:
        ranges.append((s, e, t, gtid))

api = []
for tbl in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER"):
    if tbl in tables:
        api += cur.execute(f"select start, end, correlationId, globalTid from {tbl}").fetchall()
api_by_tid = defaultdict(list)
for s, e, corr, gtid in api:
    api_by_tid[gtid].append((s, (gtid >> 24, corr)))
for v in api_by_tid.values():
    v.sort()

gpu = {}  # correlationId -> list of (kind, start, end, device, name/bytes)
# correlationId is unique only within a process: key GPU work by (pid, correlationId)
for s, e, dev, corr, nm, gp in cur.execute(
        "select start, end, deviceId, correlationId, shortName, globalPid from CUPTI_ACTIVITY_KIND_KERNEL"):
    gpu.setdefault((gp >> 24, corr), []).append(("kernel", s, e, dev, strings.get(nm, str(nm))))
if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables:
    for s, e, dev, corr, b, gp in cur.execute(
            "select start, end, deviceId, correlationId, bytes, globalPid from CUPTI_ACTIVITY_KIND_MEMCPY"):
        gpu.setdefault((gp >> 24, corr), []).append(("memcpy", s, e, dev, b))
if "CUPTI_ACTIVITY_KIND_MEMSET" in tables:
    for s, e, dev, corr, gp in cur.execute(
            "select start, end, deviceId, correlationId, globalPid from CUPTI_ACTIVITY_KIND_MEMSET"):
        gpu.setdefault((gp >> 24, corr), []).append(("memset", s, e, dev, 0))

import bisect  # noqa: E402

res = defaultdict(list)  # item -> list of dicts
gemm_names = defaultdict(int)
for s, e, text, gtid in ranges:
    item = text.split("#")[0]
    calls = api_by_tid.get(gtid, [])
    i0 = bisect.bisect_left(calls, (s,))
    i1 = bisect.bisect_right(calls, (e, (1 << 62, 1 << 62)))
    ops = [op for _, corr in calls[i0:i1] for op in gpu.get(corr, [])]
    by_dev = defaultdict(list)
    for op in ops:
        by_dev[op[3]].append(op)
    for dev, dops in by_dev.items():
        if a.device >= 0 and dev != a.device:
            continue
        kern = [o for o in dops if o[0] == "kernel"]
        cps = [o for o in dops if o[0] == "memcpy" and o[4] == a.shard_bytes]
        d = {"n_kernels": len(kern), "n_memcpy": sum(1 for o in dops if o[0] == "memcpy"),
             "n_memset": sum(1 for o in dops if o[0] == "memset"),
             "n_shard_copies": len(cps),
             "span_us": (max(o[2] for o in dops) - min(o[1] for o in dops)) / 1e3}
        if kern:
            g = max(kern, key=lambda o: o[2] - o[1])
            gemm_names[g[4][:60]] += 1
            d["gemm_us"] = (g[2] - g[1]) / 1e3
            if cps:
                comm_end = max(o[2] for o in cps)
                comm_start = min(o[1] for o in cps)
                d["comm_us"] = (comm_end - comm_start) / 1e3
                d["tail_us"] = (g[2] - comm_end) / 1e3
                d["gemm_start_minus_comm_start_us"] = (g[1] - comm_start) / 1e3
        res[item].append(d)

print(f"file {a.sqlite}  shard_bytes {a.shard_bytes}  device {'all' if a.device < 0 else a.device}")
print("longest-kernel names:", dict(gemm_names))
keys = ["n_kernels", "n_memcpy", "n_memset", "n_shard_copies", "span_us", "gemm_us", "comm_us",
        "tail_us", "gemm_start_minus_comm_start_us"]
print(f"{'item':<20} {'n':>4} " + " ".join(f"{k[:14]:>15}" for k in keys))
for item, lst in sorted(res.items()):
    vals = []
    for k in keys:
        v = [d[k] for d in lst if k in d]
        vals.append(f"{statistics.median(v):15.1f}" if v else f"{'-':>15}")
    print(f"{item:<20} {len(lst):>4} " + " ".join(vals))
