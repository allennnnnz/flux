################################################################################
# fusion-dispatch G3: PRE-REGISTERED predictions for situations the predictor has never seen
# (new models at TP=8, Llama-3-70B / Qwen2.5-72B at TP=4, Llama-3-8B at TP=2).
# Written and committed BEFORE the G3 op maps are measured (plan: "與 G2 的預測比較時不得回頭調參").
# Uses only the calibration-only hardware profiles common/cost_model/hw_profiles/
# css-host-158_tp<TP>_<mode>.json (TP=8 from G2; TP=4 / 2 from their own 4-minute calibrations) and
# the Flux registry metadata. Records, per point: predicted time of every path, the predictor's pick,
# the fixed-threshold pick, and which points the hybrid rule would probe (margin < eps, or the pick runs
# a Flux GEMM on a PCIe-tuned registry config), with eps = the profile's calibration p90 residual (the
# G2 rule) and, as a second pre-registered variant, eps = 5% (chosen from the G2 eps sweep on the old
# 320 points, i.e. without any G3 data).
# The measurement campaign (run_g3_map_v1.sh) uses the same CONFIGS / MS / MODES.
# Usage: python3 predict_g3_v1.py [--out ws/fusion-dispatch/results/g3_predictions]
################################################################################
import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "common", "cost_model"))
from predictor import AG_ARMS, FLUX_GEMM_ARMS, RS_ARMS, FluxConfigs, HardwareProfile, predict_ag, predict_rs  # noqa: E402

AG_FULL = {"L-QKV": (10240, 8192), "L-GU": (57344, 8192), "Q-GU": (59136, 8192), "L8-QKV": (6144, 4096),
           "L8-GU": (28672, 4096)}       # (N, K), each rank (N / TP, K)
RS_FULL = {"L-O": (8192, 8192), "L-down": (8192, 28672), "Q-down": (8192, 29568), "L8-O": (4096, 4096),
           "L8-down": (4096, 14336)}     # (N, K_full), each rank (N, K_full / TP)
CONFIGS = [  # (TP, side, layer)
    (8, "ag", "Q-GU"), (8, "rs", "Q-down"),
    (8, "ag", "L8-QKV"), (8, "ag", "L8-GU"), (8, "rs", "L8-O"), (8, "rs", "L8-down"),
    (4, "ag", "L-QKV"), (4, "ag", "L-GU"), (4, "rs", "L-O"), (4, "rs", "L-down"),
    (4, "ag", "Q-GU"), (4, "rs", "Q-down"),
    (2, "ag", "L8-QKV"), (2, "ag", "L8-GU"), (2, "rs", "L8-O"), (2, "rs", "L8-down"),
]
MS = [16, 64, 136, 256, 512, 1024, 2048, 3072, 4096, 8192]
MODES = ["gpu", "steady"]
EPS2 = 0.05


def dims(tp, side, layer):
    if side == "ag":
        N, K = AG_FULL[layer]
        return N // tp, K
    N, K = RS_FULL[layer]
    return N, K // tp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(HERE), "results", "g3_predictions"))
    ap.add_argument("--profiles", default=os.path.join(REPO, "common", "cost_model", "hw_profiles"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    profs, hashes, cfgs = {}, {}, {}
    for tp in sorted({c[0] for c in CONFIGS}):
        cfgs[tp] = FluxConfigs(tp)
        for mode in MODES:
            path = os.path.join(a.profiles, f"css-host-158_tp{tp}_{mode}.json")
            profs[(tp, mode)] = HardwareProfile.load(path)
            hashes[os.path.relpath(path, REPO)] = hashlib.sha256(open(path, "rb").read()).hexdigest()
    rows = []
    for tp, side, layer in CONFIGS:
        d0, d1 = dims(tp, side, layer)
        for mode in MODES:
            prof = profs[(tp, mode)]
            eps = prof.meta["calibration_residual_p90"]
            for M in MS:
                if side == "ag":
                    arms, comps, cfg = predict_ag(prof, cfgs[tp], M, d0, d1)
                    names = AG_ARMS
                else:
                    arms, comps, cfg = predict_rs(prof, cfgs[tp], M, d0, d1)
                    names = RS_ARMS
                rank = sorted(names, key=lambda x: arms[x])
                margin = (arms[rank[1]] - arms[rank[0]]) / arms[rank[0]]
                pcie = rank[0] in FLUX_GEMM_ARMS[side] and cfg.get("tuned_for") == "pcie"
                safe = next(x for x in rank if x not in FLUX_GEMM_ARMS[side])
                probe = sorted(({rank[0], rank[1]} if margin < eps else set()) | ({rank[0], safe} if pcie else set()))
                probe2 = sorted(({rank[0], rank[1]} if margin < EPS2 else set()) | ({rank[0], safe} if pcie else set()))
                rows.append([tp, side, layer, d0, d1, M, mode, cfg["source"], cfg.get("tuned_for", ""),
                             "-".join(str(v) for v in cfg["tile"]), cfg["sk"], cfg["raster"]] +
                            [f"{arms[x]:.5f}" if x in arms else "" for x in AG_ARMS] +
                            [rank[0], f"{margin:.4f}", f"{eps:.4f}", int(margin < eps), int(pcie), "|".join(probe),
                             "B" if M <= 512 else "A", int(margin < EPS2), "|".join(probe2)])
    with open(os.path.join(a.out, "predictions.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tp", "side", "layer", "dim0", "dim1", "M", "mode", "flux_cfg", "tuned_for", "tile", "sk", "raster",
                    "pred_A", "pred_B", "pred_C", "pred_D", "pred_pick", "pred_margin", "eps", "probe_margin",
                    "probe_pcie", "probe_arms", "thr512_pick", "probe_margin_eps5", "probe_arms_eps5"])
        w.writerows(rows)
    try:
        commit = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = ""
    meta = {"script": "ws/fusion-dispatch/scripts/predict_g3_v1.py", "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "git_head_before_commit": commit, "profiles_sha256": hashes, "configs": CONFIGS, "Ms": MS, "modes": MODES,
            "n_points": len(rows), "eps_rules": {"hyb": "profile calibration_residual_p90", "hyb5": EPS2},
            "rule": "predictions are frozen: G3 measurements are compared against this file, no parameter changes"}
    json.dump(meta, open(os.path.join(a.out, "predictions_meta.json"), "w"), indent=1)
    n_probe = sum(1 for r in rows if r[21])
    n_probe2 = sum(1 for r in rows if r[24])
    print(f"wrote {os.path.relpath(a.out, REPO)}/predictions.csv  ({len(rows)} points; probed: {n_probe} with the "
          f"G2 eps rule, {n_probe2} with eps = 5%)")
    for tp, side, layer in CONFIGS:
        sel = [r for r in rows if r[0] == tp and r[2] == layer and r[6] == "gpu"]
        print(f"  TP={tp} {layer:<8} gpu picks: " + " ".join(f"{r[5]}:{r[16]}" for r in sel))


if __name__ == "__main__":
    main()
