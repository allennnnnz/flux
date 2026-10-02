################################################################################
# fusion-dispatch G3 diagnostic (POST-HOC, for the G4 design only; not part of the pre-registered
# comparison): what if the decision used MEASURED standalone GEMM times (cuBLAS, Flux gemm_only:
# single-op, local, cheap to probe per deployment shape) and the MODEL only for communication and
# overlap? AG arms: A = schedule simulation(measured gemm_only, modelled Flux AG); B = modelled NCCL AG
# + measured cuBLAS; C = modelled Flux AG + measured cuBLAS; D = modelled Flux AG + measured gemm_only.
# RS arms: A = model (GemmRS cannot be split on sm80); B = measured cuBLAS + modelled NCCL RS.
# Usage: python3 g3_gemm_probe_diag_v1.py   (after eval_g3_v1.py has written the summaries)
################################################################################
import csv
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(WS))
sys.path.insert(0, os.path.join(REPO, "common", "cost_model"))
sys.path.insert(0, HERE)
from predict_g3_v1 import CONFIGS, dims  # noqa: E402
from predictor import FluxConfigs, HardwareProfile, predict_ag, predict_rs  # noqa: E402
from predictor.overlap import fused_ag_time  # noqa: E402

out = open(os.path.join(WS, "results", "g3_map", "gemm_probe_diag_log.txt"), "w")


def say(s):
    print(s)
    out.write(s + "\n")


say("POST-HOC diagnostic: measured standalone GEMMs + modelled comm / overlap (for G4 design)")
res = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0])  # group -> [num_pred, num_mix, den, n_wrong_mix, n]
for tp, side, layer in CONFIGS:
    it = defaultdict(dict)
    pts = {}
    d = os.path.join(WS, "results", "g3_map", f"tp{tp}")
    for r in csv.DictReader(open(os.path.join(d, "summary_items.csv"))):
        if r["layer"] == layer:
            it[(int(r["M"]), r["mode"])][r["item"]] = float(r["median_ms"])
    cfgs = FluxConfigs(tp)
    a, b = dims(tp, side, layer)
    for (M, mode), m in sorted(it.items()):
        prof = HardwareProfile.load(os.path.join(REPO, "common", "cost_model", "hw_profiles", f"css-host-158_tp{tp}_{mode}.json"))
        if side == "ag":
            arms, comps, cfg = predict_ag(prof, cfgs, M, a, b)
            meas = {"A": m["A_fused"], "B": m["B_nccl_cublas"], "C": m["C_fluxag_cublas"], "D": m["D_fluxag_fluxgemm"]}
            fa, cu, fg = comps["fluxag"], m["c_cublas"], m["c_fluxgemm"]
            mix = {"A": fused_ag_time(M, a, b, tp, cfg, fa, fg, prof.fused_ag), "B": comps["nccl"] + cu, "C": fa + cu, "D": fa + fg}
        else:
            arms, comps, cfg = predict_rs(prof, cfgs, M, a, b)
            meas = {"A": m["A_fused"], "B": m["B_nccl_cublas"]}
            mix = {"A": arms["A"], "B": comps["nccl_rs"] + m["c_cublas"]}
        orc = min(meas.values())
        pp, pm = min(arms, key=arms.get), min(mix, key=mix.get)
        for g in (f"{side} all", f"{side} TP={tp}", f"{side} {mode}"):
            r_ = res[g]
            r_[0] += meas[pp] - orc
            r_[1] += meas[pm] - orc
            r_[2] += orc
            r_[3] += int(meas[pm] > 1.05 * orc)
            r_[4] += 1
for g, (npred, nmix, den, nw, n) in sorted(res.items()):
    say(f"  {g:<14} n={n:<4} regret: model-only {npred / den * 100:5.2f}%   measured-GEMM + model comm/overlap {nmix / den * 100:5.2f}%  (>5% wrong: {int(nw)})")
out.close()
