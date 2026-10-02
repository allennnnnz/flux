################################################################################
# fusion-dispatch F1.2: build a dispatch table (dispatcher_v1.py format) from
# measured maps (analyze_v1.py / merge_points_v1.py *_points.csv).
#   AG layers (dispatch_map_v2.py): flux=t_fused, nccl=t_B, fluxag=t_C, fluxag_fluxgemm=t_D
#   RS layers (dispatch_map_rs_v1.py): flux=t_fused, nccl=t_B
# Ranking = ascending median time; on a 'tie' verdict (design 5.6: p10..p90 of
# per-round fused - best_off contains 0) the fused path is moved behind the best
# non-fused path (prefer the simpler, capturable path when it makes no difference).
# Usage: python3 build_table_v1.py --points a.csv b.csv --mode gpu --out table.json
################################################################################
import argparse
import csv
import json
import subprocess
import time

AG_LAYERS = {"G-FC1", "G-QKV", "L-QKV", "L-GU", "P0-4096", "P0-8192"}
RS_LAYERS = {"G-FC2", "G-O", "L-down", "L-O"}
WORLD = 8
COLS = [("flux", "t_fused"), ("nccl", "t_B"), ("fluxag", "t_C"), ("fluxag_fluxgemm", "t_D")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", nargs="+", required=True)
    ap.add_argument("--mode", required=True, choices=["gpu", "steady"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    table = {"ag": {}, "rs": {}}
    for path in a.points:
        for r in csv.DictReader(open(path)):
            if r["mode"] != a.mode:
                continue
            L, N, K, M = r["layer"], int(r["N"]), int(r["K"]), int(r["M"])
            kind = "ag" if L in AG_LAYERS else "rs" if L in RS_LAYERS else None
            if kind is None:
                continue
            key = f"{N // WORLD}x{K}" if kind == "ag" else f"{N}x{K // WORLD}"
            ranked = sorted(((p, float(r[c])) for p, c in COLS if r.get(c)), key=lambda x: x[1])
            if r["verdict"] == "tie" and ranked[0][0] == "flux" and len(ranked) > 1:
                ranked = [ranked[1], ranked[0]] + ranked[2:]
            ent = table[kind].setdefault(key, {"layer": L, "M": [], "rank": [], "verdict": []})
            ent["M"].append(M)
            ent["rank"].append([[p, round(t, 4)] for p, t in ranked])
            ent["verdict"].append(r["verdict"])
    for kind in table:
        for ent in table[kind].values():
            order = sorted(range(len(ent["M"])), key=lambda i: ent["M"][i])
            for f in ("M", "rank", "verdict"):
                ent[f] = [ent[f][i] for i in order]

    def sh(cmd):
        try:
            return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:  # noqa: BLE001
            return None
    table["meta"] = {
        "built": time.strftime("%Y-%m-%d %H:%M:%S"), "mode": a.mode, "sources": a.points,
        "host": sh("hostname"), "gpu": sh("nvidia-smi --query-gpu=name,driver_version --format=csv,noheader -i 0"),
        "flux_git": sh("git rev-parse --short HEAD"), "world": WORLD, "dtype": "bf16",
        "protocol": "dispatch_map_v2 / dispatch_map_rs_v1 (interleaved, >=200 rounds, rank-max median)",
    }
    json.dump(table, open(a.out, "w"), indent=1)
    for kind in ("ag", "rs"):
        for key, ent in table[kind].items():
            firsts = " ".join(f"{m}:{rk[0][0]}" for m, rk in zip(ent["M"], ent["rank"]))
            print(f"{kind} {key} ({ent['layer']}): {firsts}")


if __name__ == "__main__":
    main()
