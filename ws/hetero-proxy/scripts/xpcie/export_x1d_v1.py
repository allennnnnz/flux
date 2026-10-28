################################################################################
# hetero-proxy X1d (2026-10-28): report tables for X1d / D8 from the committed raw outputs, written as CSV to
# results/report_x1_20261028/ (next to the X1 tables).
#   x1d_scope.csv   results/xpcie/x1d_v1/x1d_scope_v1.json      victim x background on GPU0: ns per item, x none,
#                                                              share of cases the background outlasted the victim, GB/s
#   x1_nomix.csv    results/xpcie/x1_v1, x1_proto_{LL128,Simple} (NCCL_GRAPH_MIXING_SUPPORT=1, default)
#                   results/xpcie/x1_nomix, x1_nomix_{LL128,Simple} (=0): NCCL all-reduce and decode-step victims
# Usage: python3 export_x1d_v1.py
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


VZ = {"fs_sys": "GPU0 寫對端後用系統層級屏障（單獨）", "fs_gpu": "GPU0 寫對端後用 GPU 層級屏障（單獨）",
      "pp_sys": "GPU0↔GPU1 互傳旗標＋系統層級屏障", "hostread": "GPU0 讀一次主機記憶體",
      "llpp": "GPU0↔GPU1 互傳 16 KiB（LL 式，無屏障）", "tiny_local": "graph：極小 kernel 寫本卡",
      "tiny_peer": "graph：極小 kernel 寫對端（無屏障）", "tiny_peer_fsys": "graph：極小 kernel 寫對端＋系統層級屏障",
      "tiny_ev": "graph：event 等待＋極小 kernel＋event 記錄（NCCL 的包法）", "tiny_evrec": "graph：極小 kernel＋event 記錄"}
TZ = {"none": "沒有搬運", "d2h_g0": "GPU0 自己存出", "h2d_g0": "GPU0 自己搬回", "d2h_g0_64k": "GPU0 存出（64 KiB 小塊）",
      "d2h_g1": "只有 GPU1 存出（同一個 PCIe 交換器）", "d2h_g2": "只有 GPU2 存出（不同交換器）",
      "nvl_g0": "GPU0 經 NVLink 搬給 GPU2（不走 PCIe）"}
s = json.load(open(os.path.join(R, "x1d_v1", "x1d_scope_v1.json")))["summary"]
rows = []
for v, vz in VZ.items():
    for t, tz in TZ.items():
        x = s[f"{v}|{t}"]
        rows.append(dict(victim=v, victim_zh=vz, traffic=t, traffic_zh=tz, ns=round(x["ns"], 1), times_none=round(x["x_none"], 2),
                         outlast_frac=round(x.get("outlast_frac", 1.0), 2), bg_gbs=round(x["bg_gbs"], 1) if "bg_gbs" in x else ""))
write("x1d_scope.csv", rows)

RUNS = [("LL", 1, "x1_v1"), ("LL", 0, "x1_nomix"), ("LL128", 1, "x1_proto_LL128"), ("LL128", 0, "x1_nomix_LL128"),
        ("Simple", 1, "x1_proto_Simple"), ("Simple", 0, "x1_nomix_Simple")]
rows = []
for proto, mixing, d in RUNS:
    sm = json.load(open(os.path.join(R, d, "x1_dispatch_v1.json")))["summary"]
    for v in ("ar", "mix"):
        base = sm[f"{v}|none"]
        for t in ("none", "h2d_all", "d2h_all", "h2d_others"):
            x = sm[f"{v}|{t}"]
            rows.append(dict(protocol=proto, graph_mixing=mixing, victim=v, traffic=t, ms=round(x["ms"], 3),
                             item_us=round(x["gpu0_interval_us"], 2), vs_none_pct=round(100 * (x["ms"] / base["ms"] - 1), 1),
                             source=f"results/xpcie/{d}"))
write("x1_nomix.csv", rows)
