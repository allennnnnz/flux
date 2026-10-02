################################################################################
# fusion-dispatch G1: fit the analytical predictor (common/cost_model/predictor) on existing
# op-level measurements and test whether its decisions generalise. No GPU.
# Plan: reports/20261002_plan_general_dispatcher.md (G1; regret definition = policy_eval_v1.py).
#
# Fit uses component columns only (NCCL / Flux AG / cuBLAS / Flux gemm_only standalone times) plus
# the fused-arm times of the TRAINING rows (two fused-kernel constants for AG, the GemmRS family for
# RS). Test rows are never used for fitting.
# Axes (each per mode for A and B):
#   A    unseen M:      train M in power-of-2 grid, test held-out M (24,72,136,264,520,1032,3072,6144)
#   B    GPT-3 -> Llama: train G-* layers, test L-*;   B'  Llama -> GPT-3
#   D    gpu -> steady:  train gpu-mode rows, test steady;  D'  steady -> gpu
#   IN   in-sample reference (train = test = all rows of a mode)
# Policies on the test rows (op level, per side): oracle (best measured arm), on (always fused),
# off (always NCCL+cuBLAS), thr512 (M<=512 off, else fused), rf (random forest, rf_baseline_v1.py),
# pred (argmin predicted), hyb (pred, but measure the top-2 when the predicted margin < eps; eps is
# the p90 relative arm error on the TRAINING rows), hyb+fb (hyb, and also measure when the pick runs
# a Flux GEMM on a fallback config: the plan's rule), hyb+pcie (hyb, and also measure when the pick
# runs a Flux GEMM on a registry config tuned for a PCIe topology: revised rule, chosen AFTER seeing
# the first G1 results, so it must be re-validated on unseen shapes in G3). A margin probe measures
# the top-2 predicted arms; a risk probe measures the pick and the best arm without a Flux GEMM. A
# probe's outcome is the measured table value (optimistic: a real 20-iteration probe has noise).
# Regret = sum(t_chosen - t_oracle) / sum(t_oracle) over the test points (policy_eval_v1.py);
# mixed = 70% decode (M<=512) + 30% prefill (M>=1024) expected time, per layer then pooled.
# Usage: python3 eval_predictor_v1.py [--out results/g1_predictor] [--axes A,B,B',D,D',IN]
################################################################################
import argparse
import csv
import math
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from predictor_data_v1 import GRID, HELD, REPO, W, f2_alpha_sync, load_rows  # noqa: E402
from rf_baseline_v1 import fit_forests, predict_forests  # noqa: E402

sys.path.insert(0, os.path.join(REPO, "common", "cost_model"))
from predictor import AG_ARMS, FLUX_GEMM_ARMS, RS_ARMS, fit_profile, predict_ag, predict_rs  # noqa: E402
from predictor.overlap import fused_ag_time  # noqa: E402

ARMS = {"ag": AG_ARMS, "rs": RS_ARMS}
POLICIES = ["on", "off", "thr512", "rf", "pred", "hyb", "hyb+fb", "hyb+pcie"]


def fit_from_rows(train, alpha_sync, label):
    ag = [r for r in train if r["side"] == "ag"]
    rs = [r for r in train if r["side"] == "rs"]
    nccl = {"ag": [(r["M"] * r["K"] * 2, r["comps"]["nccl"]) for r in ag],
            "rs": [(r["M"] * r["N"] * 2, r["comps"]["nccl_rs"]) for r in rs],
            "ar": [(r["M"] * r["N"] * 2, r["comps"]["nccl_ar"]) for r in rs]}
    return fit_profile(
        W, nccl, [(r["M"] * r["K"] * 2, r["comps"]["fluxag"]) for r in ag], alpha_sync,
        [(r["M"], r["n"], r["K"], None, r["comps"]["cublas"]) for r in ag] +
        [(r["M"], r["N"], r["k"], None, r["comps"]["cublas"]) for r in rs],
        [(r["M"], r["n"], r["K"], r["cfg"]["tile"], r["comps"]["fluxgemm"]) for r in ag],
        [(r["M"], r["n"], r["K"], r["cfg"], r["comps"]["fluxag"], r["comps"]["fluxgemm"], r["arms"]["A"])
         for r in ag],
        [(r["M"], r["N"], r["k"], r["cfg"], r["arms"]["A"]) for r in rs],
        meta={"fitted_on": label, "method": "G1: op-map component columns + training fused arms",
              "source": "ws/fusion-dispatch/results/final_{ag,rs}_points.csv",
              "alpha_sync_source": "results/f2_ag_latency_v1 (gpu, M<=64)"})


def predict_row(prof, cfgs, r):
    if r["side"] == "ag":
        return predict_ag(prof, cfgs, r["M"], r["n"], r["K"])
    return predict_rs(prof, cfgs, r["M"], r["N"], r["k"])


def choose(policy, r, pred, rf, eps):
    arms = ARMS[r["side"]]
    meas = r["arms"]
    if policy == "on":
        return "A", False
    if policy == "off":
        return "B", False
    if policy == "thr512":
        return ("B" if r["M"] <= 512 else "A"), False
    if policy == "rf":
        return min(arms, key=lambda a: rf[a]), False
    rank = sorted(arms, key=lambda a: pred[a])
    pick = rank[0]
    if policy == "pred":
        return pick, False
    cand = set()
    if (pred[rank[1]] - pred[pick]) / pred[pick] < eps:  # predicted margin within model error
        cand |= {rank[0], rank[1]}
    risky = pick in FLUX_GEMM_ARMS[r["side"]] and (
        (policy == "hyb+fb" and r["cfg"]["source"] == "fallback") or
        (policy == "hyb+pcie" and r["cfg"].get("tuned_for") == "pcie"))
    if risky:  # config-cliff risk: verify against the best path that does not run a Flux GEMM
        cand |= {pick, next(a for a in rank if a not in FLUX_GEMM_ARMS[r["side"]])}
    if cand:
        return min(cand, key=lambda a: meas[a]), True
    return pick, False


def regret_stats(points):
    """points: list of (row, chosen_arm). Pooled regret, worst point, #wrong (>5%)."""
    if not points:
        return None
    num = den = 0.0
    worst, wrong = (0.0, None), 0
    for r, a in points:
        orc = min(r["arms"].values())
        t = r["arms"][a]
        num += t - orc
        den += orc
        rel = (t - orc) / orc
        if rel > worst[0]:
            worst = (rel, f"{r['layer']}@{r['M']}")
        wrong += rel > 0.05
    return num / den, worst, wrong, len(points)


def mixed_regret(points):
    by = {}
    for r, a in points:
        by.setdefault(r["layer"], []).append((r, a))
    num = den = 0.0
    for pts in by.values():
        dec = [(r, a) for r, a in pts if r["M"] <= 512]
        pre = [(r, a) for r, a in pts if r["M"] >= 1024]
        if not dec or not pre:
            continue
        def ex(ps, f):
            return sum(f(r, a) for r, a in ps) / len(ps)
        tp = 0.7 * ex(dec, lambda r, a: r["arms"][a]) + 0.3 * ex(pre, lambda r, a: r["arms"][a])
        to = 0.7 * ex(dec, lambda r, a: min(r["arms"].values())) + 0.3 * ex(pre, lambda r, a: min(r["arms"].values()))
        num += tp - to
        den += to
    return num / den if den else None


def anchors(prof, cfgs, rows, say):
    """CLAUDE.md 5.1.2: compare model outputs with known anchors (params.json) and physical bounds."""
    import json
    P = json.load(open(os.path.join(REPO, "common", "cost_model", "params.json")))
    W_ = prof.W
    nb = 4096 * 12288 * 2
    fa, nc = prof.comm.flux_ag(nb, W_), prof.comm.nccl("ag", nb)
    checks = [
        ("Flux AG  M=4096 K=12288 [ms]", fa, P["flux_ag_comm_ms"]["flux_all2all_pull"], "params flux_ag_comm_ms (Phase 0, sustained clock)"),
        ("Flux AG ingress per GPU [GB/s]", nb * (W_ - 1) / W_ / fa / 1e6,
         P["flux_ag_comm_ms"]["flux_all2all_pull_ingress_per_gpu_GBps"]["value"], "params (Phase 0)"),
        ("NCCL AG  M=4096 K=12288 [ms]", nc, P["flux_dispatch"]["nccl_ag_M4096_K12288_gpu_aligned_ms"]["value"], "params flux_dispatch (E0 v2)"),
        ("Flux AG  M=64 K=12288 [ms]", prof.comm.flux_ag(64 * 12288 * 2, W_),
         P["flux_dispatch"]["ag_latency_ms"]["flux_small_m_floor"]["value"], "params flux_small_m_floor"),
        ("NCCL AG  M=64 K=12288 [ms]", prof.comm.nccl("ag", 64 * 12288 * 2), 0.0512, "E1 G-FC1 M=64 gpu"),
        ("cuBLAS G-FC1 M=64 [ms]", prof.gemm["cublas"].time(64, 6144, 12288), 0.1198, "E1 G-FC1 M=64 gpu"),
    ]
    big = 16384 * 12288 * 2
    for name in ("NCCL", "Flux"):
        t = prof.comm.nccl("ag", big) if name == "NCCL" else prof.comm.flux_ag(big, W_)
        checks.append((f"{name} AG ingress/GPU at 403 MB [GB/s]", big * (W_ - 1) / W_ / t / 1e6,
                       P["nvlink"]["all_to_all_per_gpu_GBps"]["value"], "params nvlink all_to_all (1 GiB/GPU)"))
    say("[anchors, profile fit on all gpu rows] (flag if |dev| > 20%)")
    for name, v, ref, src in checks:
        dev = v / ref - 1
        say(f"    {name:<38} model {v:9.4f}  anchor {ref:9.4f}  dev {dev * 100:+6.1f}% {'<-- CHECK' if abs(dev) > 0.2 else ''}  ({src})")
    for k, g in prof.gemm.items():
        say(f"    gemm {k:<8} eta={g.eta:.3f} (<=1)  bw={g.bw / 1e12:.3f} TB/s (<= 2.039 HBM peak)  t0={g.t0 * 1e3:.1f} us")
    # overlap vs nsys points (results/v8_nsys, e2_nsys): hidden comm = D - A, from measured components
    say("    overlap at nsys points: hidden comm (D - A) / Flux AG, E1 measured vs schedule model (measured components)")
    for lay, M in [("G-QKV", 64), ("L-QKV", 64), ("L-GU", 512), ("G-FC1", 1024), ("G-QKV", 2048), ("L-QKV", 2048),
                   ("L-GU", 3072), ("G-FC1", 3072), ("G-QKV", 4096), ("L-QKV", 4096), ("L-GU", 4096), ("G-FC1", 4096)]:
        r = next(x for x in rows if x["side"] == "ag" and x["layer"] == lay and x["M"] == M and x["mode"] == "gpu")
        tA = fused_ag_time(M, r["n"], r["K"], W_, r["cfg"], r["comps"]["fluxag"], r["comps"]["fluxgemm"], prof.fused_ag)
        hm = (r["arms"]["D"] - r["arms"]["A"]) / r["comps"]["fluxag"]
        hp = (r["comps"]["fluxag"] + r["comps"]["fluxgemm"] - tA) / r["comps"]["fluxag"]
        say(f"      {lay:<6} M={M:<5} cfg={r['cfg']['source']:<8} measured {hm * 100:6.1f}%   model {hp * 100:6.1f}%")


def p0_crosscheck(prof, cfgs, say):
    """Shapes the model never saw: Phase 0 anchors P0-4096 / P0-8192 (E0 v2, gpu mode), and the
    diag-overlap question (N=4096 overlaps ~2% in Phase 0, N=8192 76%)."""
    it = {}
    for r in csv.DictReader(open(os.path.join(os.path.dirname(HERE), "results", "e0_anchor_v2", "summary_items.csv"))):
        if r["layer"].startswith("P0") and r["mode"] == "gpu":
            it[(r["layer"], int(r["M"]), r["item"])] = float(r["median_ms"])
    names = {"A": "A_fused", "B": "B_nccl_cublas", "C": "C_fluxag_cublas", "D": "D_fluxag_fluxgemm"}
    say("[unseen shapes: Phase 0 anchors, results/e0_anchor_v2 gpu] hidden comm = (D - A) / Flux AG")
    for lay, N in (("P0-4096", 4096), ("P0-8192", 8192)):
        n, K = N // W, 12288
        for M in sorted({k[1] for k in it if k[0] == lay}):
            g = lambda x: it[(lay, M, x)]  # noqa: E731
            cfg = cfgs.get("ag", M, n, K)
            t_mech = fused_ag_time(M, n, K, W, cfg, g("c_flux_ag"), g("c_fluxgemm"), prof.fused_ag)
            arms, _, _ = predict_ag(prof, cfgs, M, n, K)
            tiles = -(-M // cfg["tile"][0]) * -(-n // cfg["tile"][1])
            best = min(names, key=lambda a: g(names[a]))
            say(f"    {lay} M={M:<5} {cfg['source']:<8} {cfg['tile']} {cfg['sk']} tiles={tiles} ({tiles / 108:.2f} waves)"
                f"  A meas {g('A_fused'):.4f} sched-model {t_mech:.4f} full-pred {arms['A']:.4f}"
                f"  hidden meas {(g('D_fluxag_fluxgemm') - g('A_fused')) / g('c_flux_ag') * 100:5.1f}%"
                f" model {(g('c_flux_ag') + g('c_fluxgemm') - t_mech) / g('c_flux_ag') * 100:5.1f}%"
                f"  best meas {best} pred {min(arms, key=arms.get)}")
    M, n, K = 4096, 512, 12288
    fa, fg = it[("P0-4096", M, "c_flux_ag")], it[("P0-4096", M, "c_fluxgemm")]
    say("    what-if [推論] P0-4096 M=4096 with other schedules, same standalone GEMM time (diag-overlap target < 0.55 ms):")
    for name, cfg in [("fallback 128x128x64 stream-K rasterM", dict(tile=(128, 128, 64), stages=3, sk="SK", raster="M")),
                      ("128x128x64 data-parallel rasterN", dict(tile=(128, 128, 64), stages=3, sk="DP", raster="N")),
                      ("64x128x64 data-parallel rasterN", dict(tile=(64, 128, 64), stages=4, sk="DP", raster="N"))]:
        t = fused_ag_time(M, n, K, W, cfg, fa, fg, prof.fused_ag)
        say(f"      {name:<38} fused {t:.4f} ms  hidden {(fa + fg - t) / fa * 100:5.1f}%")


def axes_def(rows):
    out = []
    for mode in ("gpu", "steady"):
        R = [r for r in rows if r["mode"] == mode]
        out.append(("A", mode, [r for r in R if r["M"] in GRID], [r for r in R if r["M"] in HELD]))
        out.append(("B", mode, [r for r in R if r["model"] == "G"], [r for r in R if r["model"] == "L"]))
        out.append(("B'", mode, [r for r in R if r["model"] == "L"], [r for r in R if r["model"] == "G"]))
        out.append(("IN", mode, R, R))
    out.append(("D", "gpu->steady", [r for r in rows if r["mode"] == "gpu"], [r for r in rows if r["mode"] == "steady"]))
    out.append(("D'", "steady->gpu", [r for r in rows if r["mode"] == "steady"], [r for r in rows if r["mode"] == "gpu"]))
    return out


def rel_errors(prof, cfgs, rows):
    errs = []
    for r in rows:
        pred = predict_row(prof, cfgs, r)[0]
        errs += [abs(pred[a] / r["arms"][a] - 1) for a in ARMS[r["side"]] if r["arms"][a]]
    return errs


def q(v, p):
    s = sorted(v)
    return s[min(len(s) - 1, int(p * len(s)))] if s else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(HERE), "results", "g1_predictor"))
    ap.add_argument("--axes", default="A,B,B',D,D',IN")
    ap.add_argument("--eps_pct", type=float, default=90, help="percentile of training arm error used as eps")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows, cfgs = load_rows()
    alpha = f2_alpha_sync()
    want = set(a.axes.split(","))
    log = open(os.path.join(a.out, "eval_log.txt"), "w")

    def say(*x):
        s = " ".join(str(v) for v in x)
        print(s, flush=True)
        log.write(s + "\n")
    say(f"G1 predictor evaluation  {time.strftime('%Y-%m-%d %H:%M:%S')}  rows={len(rows)}  alpha_sync={alpha:.4f} ms")

    # mechanism check: fused-AG simulation fed with MEASURED components (no component-model error)
    for mode in ("gpu", "steady"):
        prof = fit_from_rows([r for r in rows if r["mode"] == mode], alpha, f"all {mode}")
        prof.save(os.path.join(a.out, f"profile_g1_all_{mode}.json"))
        if mode == "gpu":
            anchors(prof, cfgs, rows, say)
            p0_crosscheck(prof, cfgs, say)
        errs = []
        for r in rows:
            if r["side"] != "ag" or r["mode"] != mode:
                continue
            t = fused_ag_time(r["M"], r["n"], r["K"], W, r["cfg"], r["comps"]["fluxag"], r["comps"]["fluxgemm"],
                              prof.fused_ag)
            errs.append((abs(t / r["arms"]["A"] - 1), r["layer"], r["M"]))
        e = [x[0] for x in errs]
        say(f"[mechanism {mode}] fused AG from measured components: kappa={prof.fused_ag['kappa']:.3f} "
            f"d_tail={prof.fused_ag['d_tail']:.4f}  MAPE {statistics.mean(e) * 100:.1f}%  median "
            f"{statistics.median(e) * 100:.1f}%  p90 {q(e, .9) * 100:.1f}%  worst " +
            ", ".join(f"{l}@{m} {x * 100:.0f}%" for x, l, m in sorted(errs, reverse=True)[:5]))

    sum_rows, pt_rows, sweep_rows = [], [], []
    for axis, mode, train, test in axes_def(rows):
        if axis not in want:
            continue
        t0 = time.time()
        prof = fit_from_rows(train, alpha, f"{axis} {mode}")
        tr_err = rel_errors(prof, cfgs, train)
        eps = q(tr_err, a.eps_pct / 100)
        forests = fit_forests(train, ARMS)
        say(f"\n=== axis {axis} [{mode}]  train {len(train)} rows  test {len(test)} rows  "
            f"eps(p{a.eps_pct:.0f} train arm err) = {eps * 100:.1f}%  fit {time.time() - t0:.0f}s")
        say(f"    fused_ag kappa={prof.fused_ag['kappa']:.3f} d_tail={prof.fused_ag['d_tail']:.4f}  fused_rs alpha={prof.fused_rs['alpha_rs']:.4f} "
            f"beta_scat={prof.fused_rs['beta_scat'] * 1e3 / 1e9:.0f} GB/s")
        say("    gemm " + "; ".join(f"{k}: t0={g.t0 * 1e3:.1f}us bw={g.bw / 1e12:.2f}TB/s eta={g.eta:.3f} p={g.p:.2f} "
                                   f"tile={g.tm}x{g.tn}" for k, g in prof.gemm.items()))
        preds = {}
        for r in test:
            arms, comps, _ = predict_row(prof, cfgs, r)
            preds[id(r)] = (arms, comps, predict_forests(forests, r, ARMS[r["side"]]))
        for side in ("ag", "rs"):
            T = [r for r in test if r["side"] == side]
            if not T:
                continue
            # prediction quality
            arm_err = {x: [abs(preds[id(r)][0][x] / r["arms"][x] - 1) for r in T] for x in ARMS[side]}
            comp_err = {}
            for r in T:
                for c, v in preds[id(r)][1].items():
                    m = r["comps"].get({"nccl_rs": "nccl_rs", "fluxrs_gemm": None}.get(c, c))
                    if m:
                        comp_err.setdefault(c, []).append(abs(v / m - 1))
            near = [r for r in T if abs(r["arms"]["A"] / min(v for k, v in r["arms"].items() if k != "A") - 1) < 0.15]
            near_err = [abs(preds[id(r)][0][x] / r["arms"][x] - 1) for r in near for x in ARMS[side]]
            rf_err = [abs(preds[id(r)][2][x] / r["arms"][x] - 1) for r in T for x in ARMS[side]]
            say(f"  [{side}] arm MAPE: " + "  ".join(f"{x} {statistics.mean(v) * 100:.1f}%" for x, v in arm_err.items()) +
                f"  | near-crossover ({len(near)} pts) {statistics.mean(near_err) * 100 if near_err else float('nan'):.1f}%"
                f"  | RF arm MAPE {statistics.mean(rf_err) * 100:.1f}%")
            say("        component MAPE: " + "  ".join(f"{c} {statistics.mean(v) * 100:.1f}%" for c, v in comp_err.items()))
            # decision accuracy (fused vs off) excluding measured ties
            nt = [r for r in T if r["verdict"] in ("off", "fused")]
            acc = sum((min(ARMS[side], key=lambda x: preds[id(r)][0][x]) == "A") == (r["verdict"] == "fused") for r in nt)
            say(f"        fused-vs-off decision accuracy (non-tie): {acc}/{len(nt)}")
            # policies
            say(f"        {'policy':<9}{'regret':>9}{'decode':>9}{'prefill':>9}{'mixed':>9}{'worst':>18}{'>5%':>6}{'probes':>9}")
            for pol in POLICIES:
                ch = []
                probes = 0
                for r in T:
                    arms_p, _, rf = preds[id(r)]
                    c, pr = choose(pol, r, arms_p, rf, eps)
                    probes += pr
                    ch.append((r, c))
                st = regret_stats(ch)
                dec = regret_stats([(r, c) for r, c in ch if r["M"] <= 512])
                pre = regret_stats([(r, c) for r, c in ch if r["M"] >= 1024])
                mix = mixed_regret(ch)
                say(f"        {pol:<9}{st[0] * 100:8.2f}%{(dec[0] * 100 if dec else float('nan')):8.2f}%"
                    f"{(pre[0] * 100 if pre else float('nan')):8.2f}%{(mix * 100 if mix is not None else float('nan')):8.2f}%"
                    f"{st[1][0] * 100:9.1f}% {str(st[1][1]):<8}{st[2]:>6}{probes:>5}/{len(T)}")
                sum_rows.append([axis, mode, side, pol, f"{st[0]:.5f}", f"{dec[0]:.5f}" if dec else "",
                                 f"{pre[0]:.5f}" if pre else "", f"{mix:.5f}" if mix is not None else "",
                                 f"{st[1][0]:.4f}", st[1][1], st[2], probes, len(T), f"{eps:.4f}"])
            # eps sweep (diagnostic only; the headline uses the training-chosen eps)
            for e in (0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30):
                ch, probes = [], 0
                for r in T:
                    c, pr = choose("hyb", r, preds[id(r)][0], None, e)
                    probes += pr
                    ch.append((r, c))
                st = regret_stats(ch)
                sweep_rows.append([axis, mode, side, e, f"{st[0]:.5f}", probes, len(T)])
            for lay in sorted({r["layer"] for r in T}):
                L = sorted((r for r in T if r["layer"] == lay), key=lambda r: r["M"])
                say(f"        picks {lay:<7} M:      " + " ".join(f"{r['M']:>5}" for r in L))
                say(f"        {'':<13} oracle: " + " ".join(f"{min(r['arms'], key=r['arms'].get):>5}" for r in L))
                say(f"        {'':<13} pred:   " + " ".join(f"{min(ARMS[side], key=lambda x: preds[id(r)][0][x]):>5}" for r in L))
                say(f"        {'':<13} verdict:" + " ".join(f"{r['verdict'][:5]:>5}" for r in L))
            for r in T:
                arms_p, comps_p, rf = preds[id(r)]
                pt_rows.append([axis, mode, side, r["layer"], r["M"], r["cfg"]["source"],
                                min(r["arms"], key=r["arms"].get),
                                min(ARMS[side], key=lambda x: arms_p[x]),
                                min(ARMS[side], key=lambda x: rf[x])] +
                               [f"{r['arms'][x]:.4f}" for x in AG_ARMS if x in r["arms"]] + [""] * (4 - len(r["arms"])) +
                               [f"{arms_p[x]:.4f}" for x in AG_ARMS if x in arms_p] + [""] * (4 - len(arms_p)))
    with open(os.path.join(a.out, "eval_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["axis", "mode", "side", "policy", "regret", "regret_decode", "regret_prefill", "regret_mixed",
                    "worst_rel", "worst_point", "n_wrong_gt5pct", "probes", "n_points", "eps"])
        w.writerows(sum_rows)
    with open(os.path.join(a.out, "eval_points.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["axis", "mode", "side", "layer", "M", "flux_cfg", "oracle", "pred_pick", "rf_pick",
                    "meas_A", "meas_B", "meas_C", "meas_D", "pred_A", "pred_B", "pred_C", "pred_D"])
        w.writerows(pt_rows)
    with open(os.path.join(a.out, "eps_sweep.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["axis", "mode", "side", "eps", "regret_hyb", "probes", "n_points"])
        w.writerows(sweep_rows)
    say(f"\nwrote {a.out}/eval_summary.csv, eval_points.csv, eps_sweep.csv, profile_g1_all_*.json")
    log.close()


if __name__ == "__main__":
    main()
