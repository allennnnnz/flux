################################################################################
# fusion-dispatch G4 (D-009): pre-registered decisions on a FRESH test set, made before its op maps
# are measured. Test set: Qwen2.5-32B (never used before) at TP=8 and TP=4, Llama-3-8B at TP=4;
# M = 24, 96, 200, 384, 768, 1536, 2560, 6144 (no M used in G1-G3); gpu and steady modes.
#
#   predict   per point, three methods (and the probe list of the G4 method):
#     pred_g3   model only, G2/G3 profiles, joint GemmRS model (the method that was evaluated in G3)
#     pred_g4m  model only, G4 profiles (GemmRS GEMM = gemm_only family, 2 fitted parameters)
#     g4        model for communication + overlap, MEASURED single-GPU GEMMs (probe_gemm_v1.py)
#     probe rule for g4: predicted margin < eps, eps = max(3%, p90 calibration residual with measured
#               GEMMs), or the pick runs a Flux GEMM on a PCIe-tuned registry config (then compare
#               with the best path without a Flux GEMM)
#   finalize  after the multi-GPU path probes (run_g4_probes_v1.py, 30 rounds each): g4+probe pick =
#             fastest PROBED candidate. Second pre-registered variant g4+probe3: probe only when the
#             margin < 3% (a subset of the same probes; added before any probe or oracle run, because
#             the declared rule flagged 38% of the points). Writes decisions_g4.csv. Both outputs
#             are committed before the oracle maps (run_g4_map_v1.sh) are measured.
# Usage: python3 predict_g4_v1.py predict | finalize
################################################################################
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(WS))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "common", "cost_model"))
from build_table_v2 import layers  # noqa: E402
from predictor import AG_ARMS, FLUX_GEMM_ARMS, RS_ARMS, FluxConfigs, HardwareProfile, predict_ag, predict_rs  # noqa: E402

CONFIGS = [(8, "qwen2.5-32b"), (4, "qwen2.5-32b"), (4, "llama3-8b")]
HARNESS = {("qwen2.5-32b", "qkv"): "Q32-QKV", ("qwen2.5-32b", "gate_up"): "Q32-GU", ("qwen2.5-32b", "o"): "Q32-O",
           ("qwen2.5-32b", "down"): "Q32-down", ("llama3-8b", "qkv"): "L8-QKV", ("llama3-8b", "gate_up"): "L8-GU",
           ("llama3-8b", "o"): "L8-O", ("llama3-8b", "down"): "L8-down"}
MS = [24, 96, 200, 384, 768, 1536, 2560, 6144]
MODES = ["gpu", "steady"]
ITEM = {"A": "A_fused", "B": "B_nccl_cublas", "C": "C_fluxag_cublas", "D": "D_fluxag_fluxgemm"}
OUT = os.path.join(WS, "results", "g4_predictions")
PROF = os.path.join(REPO, "common", "cost_model", "hw_profiles")


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def gemm_table():
    g = {}
    for tp, model in CONFIGS:
        path = os.path.join(WS, "results", "g4_gemm", f"gemm_{model}_tp{tp}.csv")
        for r in csv.DictReader(open(path)):
            g[(tp, model, r["layer"], int(r["M"]), r["mode"], r["item"])] = float(r["median_ms"])
    return g


def predict():
    os.makedirs(OUT, exist_ok=True)
    G = gemm_table()
    rows, probes, hashes = [], [], {}
    for tp, model in CONFIGS:
        cfgs = FluxConfigs(tp)
        for mode in MODES:
            p_old = os.path.join(PROF, f"css-host-158_tp{tp}_{mode}.json")
            p_new = os.path.join(PROF, f"css-host-158_tp{tp}_{mode}_g4.json")
            hashes[os.path.relpath(p_old, REPO)], hashes[os.path.relpath(p_new, REPO)] = sha(p_old), sha(p_new)
            old, new = HardwareProfile.load(p_old), HardwareProfile.load(p_new)
            eps = max(0.03, new.meta["calibration_residual_meas_p90"])
            for side, shapes in layers(model, tp).items():
                for lname, (d0, d1) in shapes.items():
                    for M in MS:
                        if side == "ag":
                            meas = {"cublas": G[(tp, model, lname, M, mode, "c_cublas")],
                                    "fluxgemm": G[(tp, model, lname, M, mode, "c_fluxgemm")]}
                            a3, _, cfg = predict_ag(old, cfgs, M, d0, d1)
                            a4m, _, _ = predict_ag(new, cfgs, M, d0, d1)
                            a4, _, _ = predict_ag(new, cfgs, M, d0, d1, meas=meas)
                            names = AG_ARMS
                        else:
                            meas = {"cublas": G[(tp, model, lname, M, mode, "c_cublas")],
                                    "fluxgemm_only": G[(tp, model, lname, M, mode, "c_fluxgemm_only")]}
                            a3, _, cfg = predict_rs(old, cfgs, M, d0, d1, legacy=True)
                            a4m, _, _ = predict_rs(new, cfgs, M, d0, d1)
                            a4, _, _ = predict_rs(new, cfgs, M, d0, d1, meas=meas)
                            names = RS_ARMS
                        rank = sorted(names, key=lambda x: a4[x])
                        margin = (a4[rank[1]] - a4[rank[0]]) / a4[rank[0]]
                        pcie = rank[0] in FLUX_GEMM_ARMS[side] and cfg.get("tuned_for") == "pcie"
                        cand = ({rank[0], rank[1]} if margin < eps else set())
                        if pcie:
                            cand |= {rank[0], next(x for x in rank if x not in FLUX_GEMM_ARMS[side])}
                        layer = HARNESS[(model, lname)]
                        rows.append([tp, model, side, layer, d0, d1, M, mode, cfg["source"], cfg.get("tuned_for", ""),
                                     min(a3, key=a3.get), min(a4m, key=a4m.get), rank[0], f"{margin:.4f}", f"{eps:.4f}",
                                     int(pcie), "|".join(sorted(cand)), "B" if M <= 512 else "A"] +
                                    [f"{a4[x]:.5f}" if x in a4 else "" for x in AG_ARMS] +
                                    [f"{a3[x]:.5f}" if x in a3 else "" for x in AG_ARMS])
                        if cand:
                            probes.append([tp, layer, side, M, mode, "|".join(sorted(cand))])
    hdr = ["tp", "model", "side", "layer", "dim0", "dim1", "M", "mode", "flux_cfg", "tuned_for", "pick_pred_g3",
           "pick_pred_g4m", "pick_g4", "g4_margin", "eps", "pcie_risk", "probe_candidates", "pick_thr512"] + \
          [f"g4_{x}" for x in AG_ARMS] + [f"g3_{x}" for x in AG_ARMS]
    with open(os.path.join(OUT, "predictions_g4.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(hdr)
        w.writerows(rows)
    with open(os.path.join(OUT, "probes_g4.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tp", "layer", "side", "M", "mode", "candidates"])
        w.writerows(probes)
    head = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    gemm_files = {os.path.relpath(os.path.join(WS, "results", "g4_gemm", f"gemm_{m}_tp{t}.csv"), REPO):
                  sha(os.path.join(WS, "results", "g4_gemm", f"gemm_{m}_tp{t}.csv")) for t, m in CONFIGS}
    json.dump({"script": "ws/fusion-dispatch/scripts/predict_g4_v1.py predict", "date": time.strftime("%Y-%m-%d %H:%M:%S"),
               "git_head_before_commit": head, "profiles_sha256": hashes, "gemm_probe_sha256": gemm_files,
               "configs": CONFIGS, "Ms": MS, "modes": MODES, "n_points": len(rows), "n_probes": len(probes),
               "rule": "frozen before the oracle maps are measured; no parameter changes afterwards"},
              open(os.path.join(OUT, "predictions_g4_meta.json"), "w"), indent=1)
    print(f"wrote predictions_g4.csv ({len(rows)} points) and probes_g4.csv ({len(probes)} probes, "
          f"{len(probes) / len(rows) * 100:.0f}%)")
    for tp, model in CONFIGS:
        for lay in sorted({r[3] for r in rows if r[0] == tp and r[1] == model}):
            sel = [r for r in rows if r[0] == tp and r[3] == lay and r[7] == "gpu"]
            print(f"  TP={tp} {lay:<9} gpu g4 picks: " + " ".join(f"{r[6]}:{r[12]}" for r in sel))


def finalize():
    probe_dir = os.path.join(WS, "results", "g4_probes")
    med = {}
    for d in sorted(os.listdir(probe_dir)):
        if not d.startswith("tp"):
            continue
        path = os.path.join(probe_dir, d)
        subprocess.run([sys.executable, os.path.join(HERE, "analyze_v1.py"), path, "--out", os.path.join(path, "summary")],
                       check=True, capture_output=True)
        for r in csv.DictReader(open(os.path.join(path, "summary_items.csv"))):
            med[(int(d[2:]), r["layer"], int(r["M"]), r["mode"], r["item"])] = float(r["median_ms"])
    out_rows, missing = [], 0
    rows = list(csv.DictReader(open(os.path.join(OUT, "predictions_g4.csv"))))
    for r in rows:
        tp, M = int(r["tp"]), int(r["M"])
        cand = r["probe_candidates"].split("|") if r["probe_candidates"] else []
        pick = r["pick_g4"]
        probed = ""
        if cand:
            got = {c: med.get((tp, r["layer"], M, r["mode"], ITEM[c])) for c in cand}
            if all(v is not None for v in got.values()):
                pick = min(got, key=got.get)
                probed = " ".join(f"{c}={v:.4f}" for c, v in got.items())
            else:
                missing += 1
                probed = "MISSING"
        pick3 = pick if (cand and float(r["g4_margin"]) < 0.03 and probed != "MISSING") else r["pick_g4"]
        out_rows.append([r["tp"], r["model"], r["side"], r["layer"], r["M"], r["mode"], r["pick_pred_g3"],
                         r["pick_pred_g4m"], r["pick_g4"], pick, pick3, int(bool(cand) and float(r["g4_margin"]) < 0.03),
                         r["pick_thr512"], probed])
    with open(os.path.join(OUT, "decisions_g4.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tp", "model", "side", "layer", "M", "mode", "pick_pred_g3", "pick_pred_g4m", "pick_g4",
                    "pick_g4_probe", "pick_g4_probe3", "probed3", "pick_thr512", "probe_medians"])
        w.writerows(out_rows)
    print(f"wrote decisions_g4.csv ({len(out_rows)} points; probes missing: {missing})")


if __name__ == "__main__":
    {"predict": predict, "finalize": finalize}[sys.argv[1]]()
