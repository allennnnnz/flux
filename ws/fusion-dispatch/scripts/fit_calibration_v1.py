################################################################################
# fusion-dispatch G2: fit the predictor ONLY from the calibration microbenchmarks
# (calibrate_hw_v1.py; no decision-table point is used) and write the hardware profiles.
#
#   raw_<tag>.csv (kept rounds) -> per-item rank-max medians -> summary_calibration.csv
#   -> predictor.fit_profile per mode -> common/cost_model/hw_profiles/css-host-158_tp8_<mode>.json
# alpha_sync (Flux AllGather sync cost) = median over M <= 64 of c_flux_ag - W/(W-1) * ce_copy7 in
# gpu mode (same definition as G1 used on the F2 data; used for both profiles, see alpha_sync_gpu). The single-copy curve (ce_copy1) is stored in the
# profile meta for TP extrapolation (G3).
# Usage: python3 fit_calibration_v1.py <calibration dir> [--profiles_dir common/cost_model/hw_profiles]
################################################################################
import argparse
import csv
import glob
import json
import os
import statistics
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from predictor_data_v1 import REPO  # noqa: E402

sys.path.insert(0, os.path.join(REPO, "common", "cost_model"))
from predictor import FluxConfigs, fit_profile, predict_ag, predict_rs  # noqa: E402
from predictor.curves import Curve  # noqa: E402
from predictor.overlap import fused_ag_time, fused_rs_time  # noqa: E402

COMM_K = 6144
W = 8  # set from meta_comm.json in main()


def load_medians(cal_dir):
    v = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(cal_dir, "raw_*.csv"))):
        for r in csv.DictReader(open(path)):
            if r["kept"] == "1":
                v[(r["group"], int(r["dim0"]), int(r["dim1"]), int(r["M"]), r["mode"], r["item"])].append(
                    float(r["rank_max_ms"]))
    return {k: (statistics.median(x), len(x)) for k, x in v.items()}


def alpha_sync_gpu(med):
    """Flux AllGather sync cost from gpu mode only: back-to-back copies in steady mode pipeline across
    calls, so steady ce_copy7 is not a per-call time (seen in F2 as well) and gives alpha < 0."""
    v = []
    for (g, _, _, M, mode, it), (t, _) in med.items():
        if g == "comm" and mode == "gpu" and it == "c_flux_ag" and M <= 64:
            c7 = med.get((g, M // W, COMM_K, M, mode, "ce_copy7"))
            if c7:
                v.append(t - W / (W - 1) * c7[0])
    return statistics.median(v)


def fit_mode(med, mode, cfgs, meta_extra):
    get = lambda g, a, b, M, it: med.get((g, a, b, M, mode, it), (None, 0))[0]  # noqa: E731
    keys = [k for k in med if k[4] == mode]
    comm_ms = sorted({k[3] for k in keys if k[0] == "comm"})
    ag_shapes = sorted({(k[1], k[2]) for k in keys if k[0] == "ag"})
    rs_shapes = sorted({(k[1], k[2]) for k in keys if k[0] == "rs"})
    nccl = {"ag": [], "rs": [], "ar": []}
    flux_pts, copy1 = [], []
    for M in comm_ms:
        b = M * COMM_K * 2
        for prim, it in (("ag", "c_nccl_ag"), ("rs", "c_nccl_rs"), ("ar", "c_nccl_ar")):
            t = get("comm", M // W, COMM_K, M, it)
            if t:
                nccl[prim].append((b, t))
        fa, c7, c1 = (get("comm", M // W, COMM_K, M, x) for x in ("c_flux_ag", "ce_copy7", "ce_copy1"))
        if fa:
            flux_pts.append((b, fa))
        if c1:
            copy1.append((b / W, c1))
    cub, fag, fused_ag, fused_rs = [], [], [], []
    for n, K in ag_shapes:
        for M in sorted({k[3] for k in keys if k[0] == "ag" and (k[1], k[2]) == (n, K)}):
            cfg = cfgs.get("ag", M, n, K)
            c, g, fa, a = (get("ag", n, K, M, x) for x in ("c_cublas", "c_fluxgemm", "c_flux_ag", "A_fused"))
            if c:
                cub.append((M, n, K, None, c))
            if g:
                fag.append((M, n, K, cfg["tile"], g))
            if fa:
                flux_pts.append((M * K * 2, fa))
            if g and fa and a:
                fused_ag.append((M, n, K, cfg, fa, g, a))
    for N, k in rs_shapes:
        for M in sorted({kk[3] for kk in keys if kk[0] == "rs" and (kk[1], kk[2]) == (N, k)}):
            cfg = cfgs.get("rs", M, N, k)
            c, a = get("rs", N, k, M, "c_cublas"), get("rs", N, k, M, "A_gemmrs")
            if c:
                cub.append((M, N, k, None, c))
            if a:
                fused_rs.append((M, N, k, cfg, a))
    alpha = alpha_sync_gpu(med)
    meta = {"host": "css-host-158", "tp": W, "mode": mode,
            "method": "G2: fitted ONLY from calibration microbenchmarks (calibrate_hw_v1.py); no decision-table point",
            "alpha_sync_ms": alpha,
            "alpha_sync_method": "gpu mode: median_{M<=64}(c_flux_ag - W/(W-1) * ce_copy7) (steady copies pipeline)",
            "single_copy_curve": Curve.fit(copy1).to_dict() if copy1 else None,
            "n_samples": {"nccl": {p: len(v) for p, v in nccl.items()}, "flux_ag": len(flux_pts), "cublas": len(cub),
                          "flux_ag_gemm": len(fag), "fused_ag": len(fused_ag), "fused_rs": len(fused_rs)}}
    meta.update(meta_extra)
    prof = fit_profile(W, nccl, flux_pts, alpha, cub, fag, fused_ag, fused_rs, meta=meta)
    # in-sample residuals on the calibration set: used for the probe threshold eps (G2 eval)
    res = []
    for M, n, K, _, t in cub:
        res.append(("cublas", abs(prof.gemm["cublas"].time(M, n, K) / t - 1)))
    for M, n, K, tile, t in fag:
        res.append(("flux_ag_gemm", abs(prof.gemm["flux_ag"].time(M, n, K, tile) / t - 1)))
    for M, n, K, cfg, fa, g, a in fused_ag:
        arms, _, _ = predict_ag(prof, cfgs, M, n, K)
        res.append(("fused_ag", abs(arms["A"] / a - 1)))
        res.append(("flux_ag", abs(prof.comm.flux_ag(M * K * 2, W) / fa - 1)))
    for M, N, k, cfg, a in fused_rs:
        arms, _, _ = predict_rs(prof, cfgs, M, N, k)
        res.append(("gemm_rs", abs(arms["A"] / a - 1)))
    for prim, pts in nccl.items():
        for b, t in pts:
            res.append((f"nccl_{prim}", abs(prof.comm.nccl(prim, b) / t - 1)))
    prof.meta["calibration_residual_p90"] = sorted(r for _, r in res)[int(0.9 * len(res))]
    prof.meta["calibration_residual_by_kind"] = {
        kind: round(statistics.mean(r for k_, r in res if k_ == kind), 4) for kind in sorted({k_ for k_, _ in res})}
    return prof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cal_dir")
    ap.add_argument("--profiles_dir", default=os.path.join(REPO, "common", "cost_model", "hw_profiles"))
    a = ap.parse_args()
    global W
    W = json.load(open(os.path.join(a.cal_dir, "meta_comm.json")))["world"]
    med = load_medians(a.cal_dir)
    with open(os.path.join(a.cal_dir, "summary_calibration.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "dim0", "dim1", "M", "mode", "item", "median_ms", "n_kept"])
        for k in sorted(med):
            w.writerow(list(k) + [f"{med[k][0]:.5f}", med[k][1]])
    guards = {}
    for g in sorted(glob.glob(os.path.join(a.cal_dir, "guard_*.log"))):
        last = open(g).read().strip().splitlines()[-1]
        guards[os.path.basename(g)] = "CLEAN" if " CLEAN " in last else last
    cfgs = FluxConfigs(W)
    os.makedirs(a.profiles_dir, exist_ok=True)
    rel = os.path.relpath(a.cal_dir, REPO)
    for mode in ("gpu", "steady"):
        prof = fit_mode(med, mode, cfgs, {"source": f"{rel}/raw_*.csv", "guard": guards})
        out = os.path.join(a.profiles_dir, f"css-host-158_tp{W}_{mode}.json")
        prof.save(out)
        g = prof.gemm
        print(f"[{mode}] wrote {os.path.relpath(out, REPO)}  alpha_sync={prof.meta['alpha_sync_ms']:.4f} ms  "
              f"samples={prof.meta['n_samples']}")
        print(f"    fused_ag kappa={prof.fused_ag['kappa']:.3f} d_tail={prof.fused_ag['d_tail']:.4f}  "
              f"fused_rs alpha={prof.fused_rs['alpha_rs']:.4f} beta_scat={prof.fused_rs['beta_scat'] * 1e3 / 1e9:.0f} GB/s")
        print("    gemm " + "; ".join(f"{k}: t0={v.t0 * 1e3:.1f}us bw={v.bw / 1e12:.2f}TB/s eta={v.eta:.3f} p={v.p:.2f} "
                                    f"tile={v.tm}x{v.tn}" for k, v in g.items()))
        print(f"    calibration residual (in-sample, mean |rel err| by kind): {prof.meta['calibration_residual_by_kind']}"
              f"  p90={prof.meta['calibration_residual_p90'] * 100:.1f}%")
    print(f"guards: {guards}")


if __name__ == "__main__":
    main()
