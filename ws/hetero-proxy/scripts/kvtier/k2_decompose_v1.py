################################################################################
# hetero-proxy KV tier K2 (2026-10-16): where does a turn's TTFT go? (k2_analyze_v1.py reports totals only.)
# For every run in <dir>: TTFT = queue wait (arrival -> the iteration that runs its prefill) + prefill GPU time.
# Also: running batch size distribution (is the decode batch capped by the slots?), the reason a waiting turn
# could not start (approximated by the iterations where the waiting queue was non-empty and no prefill ran), and
# the decode-step GPU ms with vs without a KV save issued in the same iteration, all iterations (k2_analyze_v1
# only used decode-only iterations, which are rare under heavy load).
# Usage: python3 k2_decompose_v1.py <dir>  -> prints and writes <dir>/decompose.txt
################################################################################
import glob
import json
import os
import statistics
import sys

d = sys.argv[1]
out = []


def say(s=""):
    print(s)
    out.append(s)


def pct(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p * len(v)))] if v else float("nan")


for f in sorted(glob.glob(os.path.join(d, "serve_*.json"))):
    r = json.load(open(f))
    a = r["args"]
    tl = [t for t in r["turns"] if t["t_arrive"] >= a["warmup"] and "t_done" in t]
    il = [i for i in r["iters"] if i["t0"] >= a["warmup"]]
    wait = [1e3 * (t["t_start"] - t["t_arrive"]) for t in tl]
    pre = [t["pre_ms"] for t in tl]
    say(f"{os.path.basename(f)} (slots {a['slots'] if a['policy'] != 'ideal' else a['seats']}, max batch {a['max_b']})")
    say(f"  queue wait  median {statistics.median(wait):8.1f}  p90 {pct(wait, .9):8.1f}  p99 {pct(wait, .99):8.1f} ms")
    say(f"  prefill GPU median {statistics.median(pre):8.1f}  p90 {pct(pre, .9):8.1f}  p99 {pct(pre, .99):8.1f} ms")
    B = [i["B"] for i in il if i["B"]]
    say(f"  running batch: median {statistics.median(B):.0f}, p90 {pct(B, .9)}, max {max(B)}; "
        f"iterations at batch >= 32: {100 * sum(b >= 32 for b in B) / len(B):.1f}%")
    nopre = sum(1 for i in il if i["kind"] is None)
    say(f"  iterations without a prefill: {100 * nopre / len(il):.1f}%")
    byB = {}
    for i in il:
        if i["dec_ms"] is not None:
            byB.setdefault(i["B"], {"s": [], "q": []})["s" if i["save_bytes"] else "q"].append(i["dec_ms"])
    rows = [(b, statistics.median(v["q"]), statistics.median(v["s"]), len(v["q"]), len(v["s"]))
            for b, v in sorted(byB.items()) if len(v["q"]) >= 30 and len(v["s"]) >= 30]
    if rows:
        rel = [s / q - 1 for _, q, s, _, _ in rows]
        say(f"  decode GPU ms with a save issued vs without, same batch ({len(rows)} batch sizes): "
            f"median {100 * statistics.median(rel):+.1f}%, range {100 * min(rel):+.1f}% .. {100 * max(rel):+.1f}%")
open(os.path.join(d, "decompose.txt"), "w").write("\n".join(out) + "\n")
