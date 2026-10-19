################################################################################
# hetero-proxy (2026-10-19): the tables of the advisor report (KV tier K1 / K2, interference K4.1, role screening P1),
# recomputed from the committed raw outputs with the same rules as the analysis scripts, written as CSV to
# results/report_prof_20261019/ (the report page reads only these files).
#   k1_ttft.csv      results/kvtier/bench_v5 (bench_kvtier_v5.json + clocks.csv; per-case clock filter of analyze_bench_v2)
#   k1_copies.csv    same run, pure load / save cases
#   k2_policies.csv  results/kvtier/k2/loadL|loadH serve_*.json (pooled runs; rules of k2_analyze_v1 / k2_decompose_v1)
#   k2_kinds.csv     same, TTFT by what happened to the returning turn
#   k4_interf.csv    results/kvtier/k4/interference_v2/k4_interference_v2.json
#   k4_place.csv     results/kvtier/k4/place_v1/**/k4_place_v1_B*.json
#   k4_gaps.csv      results/kvtier/k4/diag_v1_nsys/gaps.txt
#   p1_roles.csv     results/hetero_roles/p1/analysis.txt
#   checks.csv       results/kvtier/check_v5, k2/check_v1, results/e2e/check_v6
#   predictions.csv  pre-registered predictions with the measured values from the tables above
# Usage: python3 export_report_v1.py   (from anywhere; paths are relative to ws/hetero-proxy)
################################################################################
import bisect
import csv
import datetime
import glob
import json
import os
import re
import statistics as st

WS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
R = os.path.join(WS, "results")
OUT = os.path.join(R, "report_prof_20261019")
os.makedirs(OUT, exist_ok=True)


def write(name, rows):
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"{name}: {len(rows)} rows")


def pct(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p * len(v)))]


# ------------------------------------------------------------------ K1
d = os.path.join(R, "kvtier", "bench_v5")
b = json.load(open(os.path.join(d, "bench_kvtier_v5.json")))
samples = {}
for line in open(os.path.join(d, "clocks.csv")):
    p = [x.strip() for x in line.split(",")]
    if len(p) < 3 or "MHz" not in p[2]:
        continue
    t = datetime.datetime.strptime(p[0], "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=datetime.timezone.utc).timestamp()
    samples.setdefault(t, {})[int(p[1])] = int(p[2].split()[0])
ts = sorted(samples)


def clock(w0, w1):
    i = max(0, bisect.bisect_right(ts, w0) - 1)
    j = bisect.bisect_right(ts, w1)
    vals = [min(samples[t].values()) for t in ts[i:max(j, i + 1)]]
    return min(vals) if vals else None


med, cmod = {}, {}
for k, v in b["rank_max_ms"].items():
    cl = [clock(*w) for w in b["windows_rank0"][k]]
    cs = [c for c in cl if c]
    cmod[k] = st.mode(cs) if cs else None
    med[k] = st.median([ms for ms, c in zip(v, cl) if c is None or cmod[k] is None or c >= 0.95 * cmod[k]])
Ps = [int(p) for p in b["args"]["P"].split(",")]
names = {"ideal": "理想（KV 還在 GPU）", "ideal_eager": "理想（逐個發 kernel）", "recompute": "重算", "serial": "先搬完再算",
         "layer": "邊搬邊算", "split0.0625": "Cake 1/16", "split0.125": "Cake 1/8", "split0.25": "Cake 1/4"}
rows = []
for P in Ps:
    rec = med[f"recompute|P{P}"]
    rec1410 = rec * (cmod[f"recompute|P{P}"] or 1410) / 1410
    for k, zh in names.items():
        m = med[f"{k}|P{P}"]
        rows.append(dict(history_tokens=P, case=k, case_zh=zh, ttft_ms=round(m, 2),
                         speedup_vs_recompute=round(rec / m, 2) if k != "recompute" else "",
                         conservative_speedup=round(rec1410 / m, 2) if k == "layer" else "",
                         recompute_clock_mhz=cmod[f"recompute|P{P}"] if k == "recompute" else ""))
write("k1_ttft.csv", rows)
k1 = {(r["history_tokens"], r["case"]): r for r in rows}
rows = []
for P in Ps:
    nb = P * 32 * 1024
    for k, zh in (("load", "搬回 CPU→GPU"), ("save", "存出 GPU→CPU")):
        key = f"{k}|P{P}"
        pr = [st.median(r[key]) for r in b["per_rank_ms"]]
        rows.append(dict(history_tokens=P, direction=k, direction_zh=zh, mb_per_gpu=round(nb / 1e6, 1),
                         ms=round(med[key], 2), gbps_slowest_gpu=round(nb / med[key] / 1e6, 1),
                         gbps_gpu0_3=round(st.median([nb / v / 1e6 for v in pr[:4]]), 1),
                         gbps_gpu4_7=round(st.median([nb / v / 1e6 for v in pr[4:]]), 1)))
write("k1_copies.csv", rows)
k1c = {(r["history_tokens"], r["direction"]): r for r in rows}

# ------------------------------------------------------------------ K2
pol_zh = {"ideal": "理想（記憶體無限）", "tier": "CPU 外存", "recompute": "重算"}
kind_zh = {"hit": "命中（KV 還在）", "restore": "從 CPU 搬回", "recompute": "重算", "new": "新對話"}
rows, krows = [], []
for load, lz in (("L", "輕負載"), ("H", "重負載")):
    for pol in ("ideal", "tier", "recompute"):
        runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(R, "kvtier", "k2", f"load{load}",
                                                                            f"serve_{pol}_r*.json")))]
        a = runs[0]["args"]
        T, I, rate = [], [], []
        for r in runs:
            ra = r["args"]
            tl = [t for t in r["turns"] if t["t_arrive"] >= ra["warmup"] and "t_done" in t]
            il = [i for i in r["iters"] if i["t0"] >= ra["warmup"]]
            rate.append((sum(i["B"] for i in il) + sum(1 for i in il if i["kind"])) / ra["duration"])
            T += tl
            I += il
        tt = [t["ttft_ms"] for t in T]
        wall = [1e3 * (i["t1"] - i["t0"]) for i in I if i["B"] > 0]
        gpu = {"decode": sum(i["dec_ms"] or 0 for i in I)}
        for k in kind_zh:
            gpu[k] = sum(i["pre_ms"] or 0 for i in I if i["kind"] == k)
        tot = sum(gpu.values())
        wait = [1e3 * (t["t_start"] - t["t_arrive"]) for t in T]
        B = [i["B"] for i in I if i["B"]]
        slots = a["seats"] if pol == "ideal" else a["slots"]
        rows.append(dict(load=load, load_zh=lz, seats=a["seats"], slots=slots, policy=pol, policy_zh=pol_zh[pol],
                         runs=len(runs), turns=len(T), ttft_median_ms=round(st.median(tt), 1),
                         ttft_p90_ms=round(pct(tt, .9), 1), ttft_p99_ms=round(pct(tt, .99), 1),
                         token_interval_median_ms=round(st.median(wall), 2),
                         token_interval_p99_ms=round(pct(wall, .99), 1), tokens_per_s=round(st.mean(rate)),
                         gpu_decode_pct=round(100 * gpu["decode"] / tot, 1),
                         gpu_recompute_pct=round(100 * gpu["recompute"] / tot, 1),
                         gpu_restore_pct=round(100 * gpu["restore"] / tot, 1),
                         queue_wait_median_ms=round(st.median(wait), 1), queue_wait_p90_ms=round(pct(wait, .9), 1),
                         queue_wait_p99_ms=round(pct(wait, .99), 1),
                         prefill_gpu_median_ms=round(st.median([t["pre_ms"] for t in T]), 1),
                         batch_median=st.median(B),
                         batch_at_slot_limit_pct=round(100 * sum(x >= slots for x in B) / len(B), 1)
                         if pol != "ideal" else ""))
        for k, kz in kind_zh.items():
            v = [t["ttft_ms"] for t in T if t["kind"] == k]
            if v:
                ctx = [t["P"] if k in ("hit", "restore") else t["T"] for t in T if t["kind"] == k]
                krows.append(dict(load=load, load_zh=lz, policy=pol, policy_zh=pol_zh[pol], kind=k, kind_zh=kz,
                                  turns=len(v), ttft_median_ms=round(st.median(v), 1),
                                  ttft_p90_ms=round(pct(v, .9), 1), context_tokens_median=round(st.median(ctx))))
write("k2_policies.csv", rows)
write("k2_kinds.csv", krows)
k2 = {(r["load"], r["policy"]): r for r in rows}

# ------------------------------------------------------------------ K4.1
f = json.load(open(os.path.join(R, "kvtier", "k4", "interference_v2", "k4_interference_v2.json")))
op_zh = {"dec8": "decode 一步（8 人）", "dec16": "decode 一步（16 人）", "dec32": "decode 一步（32 人）",
         "new2048": "讀 2048 字（prefill）", "ext128": "新的一句 128 字"}
rows = []
for op in op_zh:
    base = f["summary"][f"{op}|none"]["median_ms"]
    for bg, bz in (("none", "沒有搬運"), ("h2d", "背景全速搬回"), ("d2h", "背景全速存出")):
        s = f["summary"][f"{op}|{bg}"]
        ms = s["median_ms"] if s["median_ms"] is not None else st.median(f["rank_max_ms"][f"{op}|{bg}"])
        rows.append(dict(op=op, op_zh=op_zh[op], background=bg, background_zh=bz, ms=round(ms, 2),
                         vs_none_pct=round(100 * (ms / base - 1), 1),
                         rounds_used="all" if s["median_ms"] is None else s["kept"]))
write("k4_interf.csv", rows)
k4i = {(r["op"], r["background"]): r for r in rows}
rows = []
mode_zh = {"none": "沒有搬運", "ticks": "只加計數器", "free": "全速搬", "gu": "矩陣乘法開始時放行", "ar": "all-reduce 開始時放行"}
for path in sorted(glob.glob(os.path.join(R, "kvtier", "k4", "place_v1", "**", "k4_place_v1_B*.json"), recursive=True)):
    j = json.load(open(path))
    a = j["args"]
    for k, s in j["summary"].items():
        mode, dirn = k.split("|")
        rows.append(dict(batch=a["B"], chunk_kb=a["chunk_kb"], mode=mode, mode_zh=mode_zh[mode],
                         direction=dirn if dirn != "-" else "", step_ms=round(s["ms"], 2),
                         vs_none_pct=round(100 * s["vs_none"], 1),
                         copy_gbps_per_gpu=round(s["vol_MB"] / s["bg_ms"], 2) if "bg_ms" in s else ""))
write("k4_place.csv", rows)
k4p = {(r["batch"], r["chunk_kb"], r["mode"], r["direction"]): r for r in rows}
rows = []
for line in open(os.path.join(R, "kvtier", "k4", "diag_v1_nsys", "gaps.txt")):
    m = re.search(r"kernel duration, overlapping (.+?)\s*: n=\s*(\d+)\s+median\s+([\d.]+) us", line)
    if m:
        rows.append(dict(metric="all-reduce kernel 本身", condition=m.group(1).strip(), samples=int(m.group(2)),
                         median_us=float(m.group(3))))
    m = re.search(r"gap between consecutive all-reduce kernels, (.+?)\s*: n=\s*(\d+)\s+median\s+([\d.]+) us", line)
    if m:
        rows.append(dict(metric="相鄰 kernel 之間的空檔", condition=m.group(1).strip(), samples=int(m.group(2)),
                         median_us=float(m.group(3))))
write("k4_gaps.csv", rows)

# ------------------------------------------------------------------ P1
txt = open(os.path.join(R, "hetero_roles", "p1", "analysis.txt")).read()
att = [float(x) for x in re.findall(r"device share next to 8 GPUs ([\d.]+)%", txt)]
full = float(re.search(r"full read bandwidth \((\d+) GB/s\): share ([\d.]+)%", txt).group(2))
moe = {(m.group(1), int(m.group(2))): (float(m.group(3)), float(m.group(4)), m.group(5), float(m.group(6)))
       for m in re.finditer(r"(\S+)\s+n=(\d+)\s+compute on device\s+([\d.]+) ms \| swap to GPU \+ compute\s+([\d.]+) ms "
                            r"-> (device|swap)\s+faster x\s*([\d.]+)", txt)}
dec = float(re.search(r"adds ~([\d.]+)% decode capacity", txt).group(1))
q1, q4, mx1 = moe[("qwen3-30b-a3b", 1)], moe[("qwen3-30b-a3b", 4)], moe[("mixtral-8x7b", 1)]
rows = [
    dict(role="T1", role_zh="CPU 幫忙算 attention", metric_zh="CPU 能分擔的比例（最好的情況）", value=max(att), unit="%",
         verdict="無用"),
    dict(role="T1", role_zh="CPU 幫忙算 attention", metric_zh="即使用滿 CPU 讀取頻寬", value=full, unit="%", verdict="無用"),
    dict(role="T2", role_zh="MoE 專家在 CPU 上算", metric_zh="Qwen3-30B-A3B，每專家 1 個 token：比搬權重快", value=q1[3],
         unit="×", verdict="有用（模型放不進 GPU 時）"),
    dict(role="T2", role_zh="MoE 專家在 CPU 上算", metric_zh="Qwen3-30B-A3B，每專家 4 個 token：比搬權重快", value=q4[3],
         unit="×", verdict="有用（模型放不進 GPU 時）"),
    dict(role="T2", role_zh="MoE 專家在 CPU 上算", metric_zh="Mixtral-8x7B，每專家 1 個 token：比搬權重快", value=mx1[3],
         unit="×", verdict="有用（模型放不進 GPU 時）"),
    dict(role="T4", role_zh="CPU 幫忙做 decode", metric_zh="多出的 decode 能力", value=dec, unit="%", verdict="無用"),
    dict(role="KV", role_zh="CPU 當 KV 外存", metric_zh="歷史 8192 字：搬回比重算快（K1 實測）",
         value=k1[(8192, "layer")]["speedup_vs_recompute"], unit="×", verdict="有用 → 選為主線"),
]
write("p1_roles.csv", rows)

# ------------------------------------------------------------------ correctness
rows = []
c = json.load(open(os.path.join(R, "kvtier", "check_v5", "check_kvtier_v5.json")))
for r in c["rows"]:
    for k in ("serial", "layer", "recompute"):
        v = r["results"][k]
        rows.append(dict(stage="K1", item=f"{r['name']}（歷史 {r['prefix_len']} 字）",
                         path={"serial": "先搬完再算", "layer": "邊搬邊算", "recompute": "重算"}[k],
                         compared_with="KV 一直留在 GPU",
                         result="逐位元相同" if v["logits_bit_identical_to_resident"] and v["tokens_identical_to_resident"]
                         else ("64 字相同、logits 有最後一位差異" if v["tokens_identical_to_resident"] else "在平手處分歧"),
                         first_diff_vs_hf=v["vs_hf"], hf_top2_gap_there=v["hf_gap_at_divergence"]))
c = json.load(open(os.path.join(R, "kvtier", "k2", "check_v1", "check_k2_v1.json")))
for r in c:
    v = r["results"]["restore"]
    rows.append(dict(stage="K2", item=f"{r['name']}（歷史 {r['P']} 字）", path="搬回 + 接續（服務用的程式）",
                     compared_with="KV 留在 GPU 接續", result="逐位元相同" if v["identical_to_ext"] else "不同",
                     first_diff_vs_hf=v["vs_hf"], hf_top2_gap_there=v["hf_gap_at_divergence"]))
e = json.load(open(os.path.join(R, "e2e", "check_v6", "check_v5.json")))
mg = {(x["name"], x["config"]): x for x in json.load(open(os.path.join(R, "e2e", "check_v6", "hf_margin_v6_vsA.json")))}
for r in e:
    A = r["results"]["A"]["tokens"]
    for cfg in ("Bfinal", "Bmid8"):
        t = r["results"][cfg]["tokens"]
        i = next((j for j, (x, y) in enumerate(zip(A, t)) if x != y), None)
        rows.append(dict(stage="E2E 重驗", item=r["name"], path=f"{cfg}（經 CPU）", compared_with="A（8 卡 NCCL）",
                         result="64 字相同" if i is None else f"第 {i} 字分歧",
                         first_diff_vs_hf=r["results"][cfg]["identical_prefix"],
                         hf_top2_gap_there=mg.get((r["name"], cfg), {}).get("gap", "")))
write("checks.csv", rows)

# ------------------------------------------------------------------ predictions (pre-registered before each run)
L = lambda pol: k2[("L", pol)]  # noqa: E731
H = lambda pol: k2[("H", pol)]  # noqa: E731
lay = [k1[(P, "layer")] for P in Ps]
idl = [k1[(P, "ideal")]["ttft_ms"] for P in Ps]
rows = [
    dict(stage="K1", id="G1", prediction="理想（只算新的一句）8–20 ms", measured=f"{min(idl):.1f}–{max(idl):.1f} ms",
         verdict="不成立"),
    dict(stage="K1", id="G3", prediction="邊搬邊算 ≤ max(純搬運, 理想) + 3 ms",
         measured=f"多 {min(k1[(P, 'layer')]['ttft_ms'] - max(k1c[(P, 'load')]['ms'], k1[(P, 'ideal')]['ttft_ms']) for P in Ps):.1f}"
                  f"–{max(k1[(P, 'layer')]['ttft_ms'] - max(k1c[(P, 'load')]['ms'], k1[(P, 'ideal')]['ttft_ms']) for P in Ps):.1f} ms",
         verdict="成立"),
    dict(stage="K1", id="G4", prediction="加速 ≥ 2 / 2 / 8 / 14 / 14 / 18 倍",
         measured=" / ".join(f"{r['speedup_vs_recompute']:.1f}" for r in lay) + " 倍", verdict="成立"),
    dict(stage="K1", id="G6", prediction="Cake 不會比邊搬邊算快超過 10%",
         measured="全部更慢", verdict="成立"),
    dict(stage="K1", id="G7", prediction="搬回每卡 9–12 GB/s",
         measured=f"{min(k1c[(P, 'load')]['gbps_slowest_gpu'] for P in Ps)}–{max(k1c[(P, 'load')]['gbps_slowest_gpu'] for P in Ps)} GB/s",
         verdict="不成立（1k 歷史時小拷貝較慢）"),
    dict(stage="K2 輕負載", id="L1", prediction="外存首字中位數在理想的 10% 內",
         measured=f"+{100 * (L('tier')['ttft_median_ms'] / L('ideal')['ttft_median_ms'] - 1):.1f}%", verdict="成立"),
    dict(stage="K2 輕負載", id="L2", prediction="重算首字中位數 ≥ 外存 1.5 倍",
         measured=f"{L('recompute')['ttft_median_ms'] / L('tier')['ttft_median_ms']:.2f} 倍", verdict="成立"),
    dict(stage="K2 輕負載", id="L3", prediction="三者吞吐量差在 ±3% 內",
         measured=f"{L('ideal')['tokens_per_s']} / {L('tier')['tokens_per_s']} / {L('recompute')['tokens_per_s']} 字/秒",
         verdict="成立"),
    dict(stage="K2 輕負載", id="L4", prediction="重算佔 GPU 時間 10–20%", measured=f"{L('recompute')['gpu_recompute_pct']}%",
         verdict="成立"),
    dict(stage="K2 重負載", id="H1", prediction="重算佔 GPU 時間 25–45%", measured=f"{H('recompute')['gpu_recompute_pct']}%",
         verdict="成立"),
    dict(stage="K2 重負載", id="H2", prediction="外存吞吐量比重算高 ≥ 10%",
         measured=f"+{100 * (H('tier')['tokens_per_s'] / H('recompute')['tokens_per_s'] - 1):.1f}%", verdict="不成立"),
    dict(stage="K2 重負載", id="H3", prediction="外存首字中位數比重算快 ≥ 2 倍",
         measured=f"{H('recompute')['ttft_median_ms'] / H('tier')['ttft_median_ms']:.0f} 倍", verdict="成立"),
    dict(stage="K4.1", id="I1", prediction="背景搬回時 decode 慢 0–10%",
         measured=f"+{k4i[('dec16', 'h2d')]['vs_none_pct']}%", verdict="不成立（干擾遠大於預期）"),
    dict(stage="K4.1", id="R1", prediction="計數器本身成本 ≤ 3%", measured=f"+{k4p[(16, 512, 'ticks', '')]['vs_none_pct']}%",
         verdict="成立"),
]
write("predictions.csv", rows)
