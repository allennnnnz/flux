################################################################################
# hetero-proxy KV tier K2 (2026-10-16): analysis of k2_serve_v1.py runs (serve_<policy>_<tag>.json in one directory).
# Per policy (runs of the same policy pooled; each run also listed):
#   turns completed after warm-up, generated tokens / s
#   TTFT (arrival -> first token) median / p90 / p99, all turns and by kind (hit / restore / recompute / new)
#   user-visible time per output token: wall duration of the iterations (one decode step + at most one prefill)
#   decode-step GPU ms by batch size, iterations WITH PCIe traffic (save or restore bytes > 0) vs WITHOUT at the
#   same batch size -> the interference of the KV traffic with the others' decode
#   GPU time shares: recompute prefill / restore+ext prefill / hit prefill / new prefill / decode
# Usage: python3 k2_analyze_v1.py <dir>
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


runs = {}
for f in sorted(glob.glob(os.path.join(d, "serve_*.json"))):
    r = json.load(open(f))
    runs.setdefault(r["args"]["policy"], []).append((os.path.basename(f), r))

summary = {}
for pol in ("ideal", "tier", "recompute"):
    if pol not in runs:
        continue
    say(f"=== {pol}: {len(runs[pol])} run(s)")
    T, I, gen_rate = [], [], []
    for name, r in runs[pol]:
        a = r["args"]
        tl = [t for t in r["turns"] if t["t_arrive"] >= a["warmup"] and "t_done" in t]
        il = [i for i in r["iters"] if i["t0"] >= a["warmup"]]
        gen = sum(i["B"] for i in il) + sum(1 for i in il if i["kind"])
        gen_rate.append(gen / a["duration"])
        say(f"  {name}: seats {a['seats']} slots {a['slots']} think {a['think']} s, {len(tl)} turns, "
            f"{gen / a['duration']:.0f} tok/s, TTFT median {statistics.median([t['ttft_ms'] for t in tl]):.1f} ms")
        T += tl
        I += il
    tt = [t["ttft_ms"] for t in T]
    say(f"  TTFT all  n={len(tt):<5} median {statistics.median(tt):8.1f}  p90 {pct(tt, .9):8.1f}  p99 {pct(tt, .99):8.1f} ms")
    for k in ("hit", "restore", "recompute", "new"):
        v = [t["ttft_ms"] for t in T if t["kind"] == k]
        if v:
            P = [t["P"] if k in ("hit", "restore") else t["T"] for t in T if t["kind"] == k]
            say(f"  TTFT {k:<9} n={len(v):<5} median {statistics.median(v):8.1f}  p90 {pct(v, .9):8.1f}  "
                f"p99 {pct(v, .99):8.1f} ms   (tokens already in context: median {statistics.median(P):.0f})")
    wall = [1e3 * (i["t1"] - i["t0"]) for i in I if i["B"] > 0]
    say(f"  time per output token (iteration wall): median {statistics.median(wall):.2f}  p90 {pct(wall, .9):.2f}  "
        f"p99 {pct(wall, .99):.2f} ms")
    gpu = {"decode": sum(i["dec_ms"] or 0 for i in I)}
    for k in ("hit", "restore", "recompute", "new"):
        gpu[k] = sum(i["pre_ms"] or 0 for i in I if i["kind"] == k)
    tot = sum(gpu.values())
    say("  GPU time shares: " + ", ".join(f"{k} {100 * v / tot:.1f}%" for k, v in gpu.items()) +
        f"  (busy {tot / 1e3:.0f} s of {sum(r['args']['duration'] for _, r in runs[pol]):.0f} s)")
    byB = {}
    for i in I:
        if i["dec_ms"] is not None and i["kind"] is None:  # decode-only iterations
            byB.setdefault(i["B"], {"pcie": [], "quiet": []})["pcie" if i["save_bytes"] else "quiet"].append(i["dec_ms"])
    rows = []
    for B in sorted(byB):
        q, p = byB[B]["quiet"], byB[B]["pcie"]
        if len(q) >= 20 and len(p) >= 20:
            rows.append(f"B={B}: quiet {statistics.median(q):.2f} (n={len(q)}) / with saves {statistics.median(p):.2f} "
                        f"(n={len(p)}) -> {100 * (statistics.median(p) / statistics.median(q) - 1):+.1f}%")
    if rows:
        say("  decode step GPU ms, decode-only iterations, with vs without a KV save in flight:")
        for x in rows:
            say("    " + x)
    summary[pol] = dict(ttft_median=statistics.median(tt), ttft_p90=pct(tt, .9), ttft_p99=pct(tt, .99),
                        tok_s=statistics.mean(gen_rate), tpot_median=statistics.median(wall), gpu_share=gpu)
say("")
if "recompute" in summary and "tier" in summary:
    a, b = summary["recompute"], summary["tier"]
    say(f"tier vs recompute: TTFT median x{a['ttft_median'] / b['ttft_median']:.2f}, p90 x{a['ttft_p90'] / b['ttft_p90']:.2f}, "
        f"p99 x{a['ttft_p99'] / b['ttft_p99']:.2f}; throughput {100 * (b['tok_s'] / a['tok_s'] - 1):+.1f}%; "
        f"time per token {100 * (b['tpot_median'] / a['tpot_median'] - 1):+.1f}%")
    g = a["gpu_share"]
    say(f"recompute share of GPU time in the recompute policy: {100 * g['recompute'] / sum(g.values()):.1f}% "
        "(pre-registered rule: < 10% -> switch to the MoE route)")
open(os.path.join(d, "analysis.txt"), "w").write("\n".join(out) + "\n")
