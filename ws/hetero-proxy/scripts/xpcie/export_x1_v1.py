################################################################################
# hetero-proxy X1 (2026-10-28): report tables for the X1 findings, from the committed raw outputs, written as CSV to
# results/report_x1_20261028/ (read by the advisor report page next to results/report_prof_20261019/).
#   x1_victims.csv   results/xpcie/x1_v1/x1_dispatch_v1.json      victim x background: graph time, per-item interval
#   x1_fence.csv     results/xpcie/x1b_v1 + x1c_v1                 NVLink ping-pong by fence scope, host read, local
#   x1_proto.csv     results/xpcie/x1_proto_{LL,LL128,Simple}      NCCL small all-reduce per protocol
# Usage: python3 export_x1_v1.py
################################################################################
import csv
import json
import os

WS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
R = os.path.join(WS, "results", "xpcie")
OUT = os.path.join(WS, "results", "report_x1_20261028")
os.makedirs(OUT, exist_ok=True)


def write(name, rows):
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"{name}: {len(rows)} rows")


VZ = {"tiny": "極小 kernel（只量啟動）", "gemm": "decode 大小的矩陣乘法", "ar": "NCCL 小 all-reduce", "mix": "模擬 decode 一步"}
TZ = {"none": "沒有搬運", "h2d_all": "全部 GPU 背景搬回", "d2h_all": "全部 GPU 背景存出", "h2d_self": "只有 GPU0 搬回",
      "h2d_others": "只有其他 GPU 搬回"}
s = json.load(open(os.path.join(R, "x1_v1", "x1_dispatch_v1.json")))["summary"]
rows = []
for v, vz in VZ.items():
    base = s[f"{v}|none"]
    for t, tz in TZ.items():
        x = s[f"{v}|{t}"]
        rows.append(dict(victim=v, victim_zh=vz, traffic=t, traffic_zh=tz, ms=round(x["ms"], 3),
                         vs_none_pct=round(100 * (x["ms"] / base["ms"] - 1), 1), item_us=round(x["gpu0_interval_us"], 2),
                         added_us=round(x["gpu0_interval_us"] - base["gpu0_interval_us"], 2)))
write("x1_victims.csv", rows)

MZ = {"none": "沒有搬運", "h2d_both": "兩張卡都背景搬回", "d2h_both": "兩張卡都背景存出", "d2h_gpu0": "只有 GPU0 存出",
      "d2h_gpu1": "只有 GPU1 存出"}
TESTS = [("x1c_v1/x1c_fence_v1.json", "pp_fsys", "NVLink 互傳小旗標：寫入後用系統層級屏障"),
         ("x1c_v1/x1c_fence_v1.json", "pp_fgpu", "NVLink 互傳小旗標：寫入後用 GPU 層級屏障"),
         ("x1c_v1/x1c_fence_v1.json", "pp_none", "NVLink 互傳小旗標：不加屏障"),
         ("x1b_v1/x1b_latency_v1.json", "hostread", "GPU 讀一次主機記憶體"),
         ("x1b_v1/x1b_latency_v1.json", "localspin", "GPU 讀一次自己的記憶體（對照）")]
rows = []
for path, t, tz in TESTS:
    sm = json.load(open(os.path.join(R, path)))["summary"]
    for m, mz in MZ.items():
        x = sm[f"{t}|{m}"]
        rows.append(dict(test=t, test_zh=tz, traffic=m, traffic_zh=mz, ns=round(x["ns"], 1), times_none=round(x["x_none"], 2)))
write("x1_fence.csv", rows)

rows = []
for p in ("LL", "LL128", "Simple"):
    sm = json.load(open(os.path.join(R, f"x1_proto_{p}", "x1_dispatch_v1.json")))["summary"]
    base = sm["ar|none"]
    for t in ("none", "h2d_all", "d2h_all"):
        x = sm[f"ar|{t}"]
        rows.append(dict(protocol=p, traffic=t, traffic_zh=TZ[t], ar_us=round(x["gpu0_interval_us"], 2),
                         vs_none_pct=round(100 * (x["ms"] / base["ms"] - 1), 1)))
write("x1_proto.csv", rows)
