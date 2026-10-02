################################################################################
# fusion-dispatch G2: evaluate hardware profiles fitted ONLY from calibration microbenchmarks
# (fit_calibration_v1.py) against every existing op-level measurement. Nothing is fitted here, so
# all 320 rows (8 layers x 20 M x 2 modes) are out-of-sample.
# Cases: profile_gpu -> gpu rows, profile_steady -> steady rows, and the two cross combinations.
# Policies and regret: same definitions as eval_predictor_v1.py (on / off / thr512 / pred / hyb /
# hyb+pcie; no random forest here: the calibration set has no op-level arms to train it on).
# eps (probe threshold) = p90 in-sample residual of the calibration fit (stored in the profile).
# Also repeats the G1 checks with the calibrated profile: anchors and the unseen Phase 0 shapes.
# Usage: python3 eval_profile_v1.py [--profiles common/cost_model/hw_profiles] [--out results/g2_calibration]
################################################################################
import argparse
import copy
import csv
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eval_predictor_v1 as ev  # noqa: E402
from predictor_data_v1 import REPO, W, load_rows  # noqa: E402

sys.path.insert(0, os.path.join(REPO, "common", "cost_model"))
from predictor import HardwareProfile  # noqa: E402

POLICIES = ["on", "off", "thr512", "pred", "hyb", "hyb+pcie"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", default=os.path.join(REPO, "common", "cost_model", "hw_profiles"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(HERE), "results", "g2_calibration"))
    a = ap.parse_args()
    rows, cfgs = load_rows()
    profs = {m: HardwareProfile.load(os.path.join(a.profiles, f"css-host-158_tp{W}_{m}.json")) for m in ("gpu", "steady")}
    log = open(os.path.join(a.out, "eval_log.txt"), "w")

    def say(*x):
        s = " ".join(str(v) for v in x)
        print(s, flush=True)
        log.write(s + "\n")
    say(f"G2 evaluation of calibration-only profiles  {time.strftime('%Y-%m-%d %H:%M:%S')}  rows={len(rows)}")
    for m, p in profs.items():
        say(f"  profile {m}: {os.path.relpath(a.profiles, REPO)}/css-host-158_tp{W}_{m}.json  fitted {p.meta.get('date')}  "
            f"kappa={p.fused_ag['kappa']:.3f} d_tail={p.fused_ag['d_tail']:.4f}  calibration p90 residual "
            f"{p.meta['calibration_residual_p90'] * 100:.1f}%")
    ev.anchors(profs["gpu"], cfgs, rows, say, label="calibration-only gpu profile")
    ev.p0_crosscheck(profs["gpu"], cfgs, say, label="calibration-only gpu profile")

    # the kernel-start constant (t_k0, local_bw) comes from nsys runs of evaluation layers, i.e. it is
    # the one input that is not from the calibration set: check that decisions do not depend on it
    say("[sensitivity: nsys kernel-start constant] AG side, matched mode, predictor alone")
    for m in ("gpu", "steady"):
        T = [r for r in rows if r["mode"] == m and r["side"] == "ag"]
        line = []
        for label, tk0, lbw in (("nsys", None, None), ("t_k0 x0.7", 0.7, None), ("t_k0 x1.3", 1.3, None),
                                ("no local-copy term", None, 1e18), ("const 0.040 ms", 0.040, 1e18)):
            p = copy.deepcopy(profs[m])
            if tk0 is not None:
                p.fused_ag["t_k0"] = tk0 if label.startswith("const") else p.fused_ag["t_k0"] * tk0
            if lbw is not None:
                p.fused_ag["local_bw"] = lbw
            ch = []
            for r in T:
                arms_ = ev.predict_row(p, cfgs, r)[0]
                ch.append((r, min(arms_, key=arms_.get)))
            line.append(f"{label}: {ev.regret_stats(ch)[0] * 100:.2f}%")
        say(f"    {m:<6} " + "  ".join(line))

    sum_rows, pt_rows, sweep_rows = [], [], []
    for pm, dm in (("gpu", "gpu"), ("steady", "steady"), ("gpu", "steady"), ("steady", "gpu")):
        prof = profs[pm]
        eps = prof.meta["calibration_residual_p90"]
        test = [r for r in rows if r["mode"] == dm]
        preds = {id(r): ev.predict_row(prof, cfgs, r) for r in test}
        say(f"\n=== profile {pm} -> rows {dm}  ({len(test)} rows, all out-of-sample)  eps={eps * 100:.1f}%")
        for side in ("ag", "rs"):
            T = [r for r in test if r["side"] == side]
            arms = ev.ARMS[side]
            arm_err = {x: [abs(preds[id(r)][0][x] / r["arms"][x] - 1) for r in T] for x in arms}
            comp_err = {}
            for r in T:
                for c, v in preds[id(r)][1].items():
                    m = r["comps"].get(c) if c != "fluxrs_gemm" else None
                    if m:
                        comp_err.setdefault(c, []).append(abs(v / m - 1))
            near = [r for r in T if abs(r["arms"]["A"] / min(v for k, v in r["arms"].items() if k != "A") - 1) < 0.15]
            near_err = [abs(preds[id(r)][0][x] / r["arms"][x] - 1) for r in near for x in arms]
            say(f"  [{side}] arm MAPE: " + "  ".join(f"{x} {statistics.mean(v) * 100:.1f}%" for x, v in arm_err.items()) +
                f"  | near-crossover ({len(near)} pts) {statistics.mean(near_err) * 100:.1f}%")
            say("        component MAPE: " + "  ".join(f"{c} {statistics.mean(v) * 100:.1f}%" for c, v in comp_err.items()))
            nt = [r for r in T if r["verdict"] in ("off", "fused")]
            acc = sum((min(arms, key=lambda x: preds[id(r)][0][x]) == "A") == (r["verdict"] == "fused") for r in nt)
            say(f"        fused-vs-off decision accuracy (non-tie): {acc}/{len(nt)}")
            say(f"        {'policy':<9}{'regret':>9}{'decode':>9}{'prefill':>9}{'mixed':>9}{'worst':>18}{'>5%':>6}{'probes':>9}")
            for pol in POLICIES:
                ch, probes = [], 0
                for r in T:
                    c, pr = ev.choose(pol, r, preds[id(r)][0], None, eps)
                    probes += pr
                    ch.append((r, c))
                st = ev.regret_stats(ch)
                dec = ev.regret_stats([(r, c) for r, c in ch if r["M"] <= 512])
                pre = ev.regret_stats([(r, c) for r, c in ch if r["M"] >= 1024])
                mix = ev.mixed_regret(ch)
                say(f"        {pol:<9}{st[0] * 100:8.2f}%{dec[0] * 100:8.2f}%{pre[0] * 100:8.2f}%{mix * 100:8.2f}%"
                    f"{st[1][0] * 100:9.1f}% {str(st[1][1]):<8}{st[2]:>6}{probes:>5}/{len(T)}")
                sum_rows.append([pm, dm, side, pol, f"{st[0]:.5f}", f"{dec[0]:.5f}", f"{pre[0]:.5f}", f"{mix:.5f}",
                                 f"{st[1][0]:.4f}", st[1][1], st[2], probes, len(T), f"{eps:.4f}"])
            sweep = []
            for e in (0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20):
                ch, probes = [], 0
                for r in T:
                    c, pr = ev.choose("hyb+pcie", r, preds[id(r)][0], None, e)
                    probes += pr
                    ch.append((r, c))
                sweep.append(f"{e * 100:.0f}%: {ev.regret_stats(ch)[0] * 100:.2f}% ({probes})")
                sweep_rows.append([pm, dm, side, e, f"{ev.regret_stats(ch)[0]:.5f}", probes, len(T)])
            say("        eps sweep (hyb+pcie, diagnostic): " + "  ".join(sweep))
            for lay in sorted({r["layer"] for r in T}):
                L = sorted((r for r in T if r["layer"] == lay), key=lambda r: r["M"])
                say(f"        picks {lay:<7} M:      " + " ".join(f"{r['M']:>5}" for r in L))
                say(f"        {'':<13} oracle: " + " ".join(f"{min(r['arms'], key=r['arms'].get):>5}" for r in L))
                say(f"        {'':<13} pred:   " + " ".join(f"{min(arms, key=lambda x: preds[id(r)][0][x]):>5}" for r in L))
            for r in T:
                ap_ = preds[id(r)][0]
                pt_rows.append([pm, dm, side, r["layer"], r["M"], r["cfg"]["source"], r["cfg"].get("tuned_for", ""),
                                min(r["arms"], key=r["arms"].get), min(arms, key=lambda x: ap_[x])] +
                               [f"{r['arms'][x]:.4f}" if x in r["arms"] else "" for x in ev.AG_ARMS] +
                               [f"{ap_[x]:.4f}" if x in ap_ else "" for x in ev.AG_ARMS])
    with open(os.path.join(a.out, "eval_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["profile", "rows", "side", "policy", "regret", "regret_decode", "regret_prefill", "regret_mixed",
                    "worst_rel", "worst_point", "n_wrong_gt5pct", "probes", "n_points", "eps"])
        w.writerows(sum_rows)
    with open(os.path.join(a.out, "eval_points.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["profile", "rows", "side", "layer", "M", "flux_cfg", "tuned_for", "oracle", "pred_pick",
                    "meas_A", "meas_B", "meas_C", "meas_D", "pred_A", "pred_B", "pred_C", "pred_D"])
        w.writerows(pt_rows)
    with open(os.path.join(a.out, "eps_sweep.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["profile", "rows", "side", "eps", "regret_hyb_pcie", "probes", "n_points"])
        w.writerows(sweep_rows)
    say(f"\nwrote {os.path.relpath(a.out, REPO)}/eval_log.txt, eval_summary.csv, eval_points.csv, eps_sweep.csv")
    log.close()


if __name__ == "__main__":
    main()
