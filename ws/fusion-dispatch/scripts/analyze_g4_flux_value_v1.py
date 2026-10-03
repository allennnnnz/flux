################################################################################
# fusion-dispatch G4 follow-up: where does Flux help at block level, and what does the decider add over
# a simple phase rule? Post-hoc analysis of the G4 block oracle (nothing is re-measured or refitted).
# Input : results/g4_block_oracle/raw_<model>_tp<TP>_<phase>_<mode>_L4.csv (validate_block_v4.py, 200 rounds)
#         results/g4_block_tables/layout_g4.csv (pre-registered layout = auto_g4)
#         results/g4_block_tables/table_g4_<cfg>_<phase>.json (pre-registered per-layer paths of sp_g4)
# Output: results/g4_block_oracle/flux_value_log.txt, flux_value_points.csv
#   1. decomposition per point: layout cost   = sp_nccl / tp_ar_vllm - 1   (SP vs vLLM default, no Flux)
#                               Flux in SP    = min(sp_flux, sp_rsflux) / sp_nccl - 1   (same layout)
#                               also vs tp_ar (TP + plain NCCL all-reduce, the pre-custom-AR baseline)
#   2. policies scored like eval_g4_block_v1.py (regret vs the best deployable policy, saving vs vLLM):
#        rule_phase  decode -> tp_ar_vllm, prefill -> sp_flux ("decode vLLM, prefill Flux")
#        auto_g4     the pre-registered decider
#   3. share of the sp_g4 per-layer prefill decisions that use a Flux path
# Usage: python3 analyze_g4_flux_value_v1.py
################################################################################
import csv
import glob
import json
import os
import statistics
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
R = os.path.join(WS, "results")
DEPLOY = ["tp_ar_vllm", "sp_nccl", "sp_flux", "sp_rsflux", "sp_g3", "sp_g4"]
FLUX_PATHS = {"flux", "fluxag", "fluxag_fluxgemm"}


def main():
    med = defaultdict(dict)
    for f in glob.glob(os.path.join(R, "g4_block_oracle", "raw_*.csv")):
        base = os.path.basename(f)[4:-4]
        model, tp = base.split("_tp")[0], int(base.split("_tp")[1].split("_")[0])
        v = defaultdict(list)
        for r in csv.DictReader(open(f)):
            if r["kept"] == "1":
                v[(r["phase"], int(r["M"]), r["policy"])].append(float(r["rank_max_ms"]))
        for (phase, M, pol), x in v.items():
            med[(model, tp, phase, M)][pol] = statistics.median(x)
    lay = {(r["model"], int(r["tp"]), r["phase"], int(r["M"])): r["choice_final"] for r in
           csv.DictReader(open(os.path.join(R, "g4_block_tables", "layout_g4.csv")))}
    out = open(os.path.join(R, "g4_block_oracle", "flux_value_log.txt"), "w")

    def say(s=""):
        print(s, flush=True)
        out.write(s + "\n")

    rows = []
    say("[1] decomposition per point (block time, rank-max median ms)")
    say(f"{'config':<22}{'M':>6}{'tp_ar':>8}{'vllm':>8}{'sp_nccl':>9}{'flux_sp':>9}  {'layout cost':>11}{'Flux in SP':>11}"
        f"{'vs tp_ar':>9}{'vs vllm':>9}")
    for k in sorted(med, key=lambda k: (k[0], -k[1], k[2] != "decode", k[3])):
        m = med[k]
        flux_sp = min(m[p] for p in ("sp_flux", "sp_rsflux") if p in m)
        dep = {p: m[p] for p in DEPLOY if p in m}
        orc = min(dep.values())
        rule = m["tp_ar_vllm"] if k[2] == "decode" else m["sp_flux"]
        auto = m[lay[k]]
        lc, fs = m["sp_nccl"] / m["tp_ar_vllm"] - 1, flux_sp / m["sp_nccl"] - 1
        say(f"{k[0] + ' tp' + str(k[1]) + ' ' + k[2]:<22}{k[3]:>6}{m['tp_ar']:>8.3f}{m['tp_ar_vllm']:>8.3f}{m['sp_nccl']:>9.3f}"
            f"{flux_sp:>9.3f}  {lc * 100:>10.1f}%{fs * 100:>10.1f}%{(flux_sp / m['tp_ar'] - 1) * 100:>8.1f}%"
            f"{(flux_sp / m['tp_ar_vllm'] - 1) * 100:>8.1f}%")
        rows.append({"model": k[0], "tp": k[1], "phase": k[2], "M": k[3], "tp_ar": m["tp_ar"], "tp_ar_vllm": m["tp_ar_vllm"],
                     "sp_nccl": m["sp_nccl"], "flux_in_sp": flux_sp, "layout_cost": lc, "flux_gain_in_sp": fs,
                     "oracle": orc, "rule_phase": rule, "auto_g4": auto})
    for phase in ("decode", "prefill"):
        P = [r for r in rows if r["phase"] == phase]
        lcs, fss = [r["layout_cost"] for r in P], [r["flux_gain_in_sp"] for r in P]
        say(f"  {phase}: layout cost {min(lcs) * 100:+.1f}% .. {max(lcs) * 100:+.1f}%   Flux in SP {min(fss) * 100:+.1f}% .. "
            f"{max(fss) * 100:+.1f}%   vLLM custom AR vs NCCL AR "
            f"{min(r['tp_ar_vllm'] / r['tp_ar'] - 1 for r in P) * 100:+.1f}% .. {max(r['tp_ar_vllm'] / r['tp_ar'] - 1 for r in P) * 100:+.1f}%")
    dec = [r for r in rows if r["phase"] == "decode"]
    say("  decode, Flux in SP vs sp_nccl by M: " + "  ".join(
        f"M={M}: {min(r['flux_gain_in_sp'] for r in dec if r['M'] == M) * 100:+.1f}..{max(r['flux_gain_in_sp'] for r in dec if r['M'] == M) * 100:+.1f}%"
        for M in sorted({r["M"] for r in dec})))

    say("\n[2] policies (regret = sum(t - t_best) / sum(t_best); saving = 1 - sum(t) / sum(t_vllm))")
    for gname, gf in (("all", lambda r: True), ("decode", lambda r: r["phase"] == "decode"),
                      ("prefill", lambda r: r["phase"] == "prefill")):
        P = [r for r in rows if gf(r)]
        o, v = sum(r["oracle"] for r in P), sum(r["tp_ar_vllm"] for r in P)
        for pol in ("tp_ar_vllm", "rule_phase", "auto_g4"):
            t = sum(r[pol] for r in P)
            worst = max(P, key=lambda r: r[pol] / r["oracle"])
            say(f"  {gname:<8}{pol:<12} regret {(t / o - 1) * 100:6.2f}%  saving vs vLLM {(1 - t / v) * 100:5.1f}%  "
                f"worst {(worst[pol] / worst['oracle'] - 1) * 100:5.1f}% at {worst['model']} tp{worst['tp']} {worst['phase']} M={worst['M']}")

    say("\n[3] sp_g4 per-layer first choice in prefill (pre-registered tables)")
    n_all = n_flux = n_fused = 0
    for f in sorted(glob.glob(os.path.join(R, "g4_block_tables", "table_g4_*_prefill.json"))):
        t = json.load(open(f))
        picks = [e["rank"][i][0][0] for kind in ("ag", "rs") for e in t[kind].values() for i in range(len(e["M"]))]
        nf = sum(p in FLUX_PATHS for p in picks)
        n_all, n_flux, n_fused = n_all + len(picks), n_flux + nf, n_fused + sum(p == "flux" for p in picks)
        say(f"  {os.path.basename(f)}: {nf}/{len(picks)} Flux ({sum(p == 'flux' for p in picks)} fused)")
    say(f"  total: {n_flux}/{n_all} decisions use a Flux path ({n_flux / n_all * 100:.0f}%), {n_fused} fused kernel")

    with open(os.path.join(R, "g4_block_oracle", "flux_value_points.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    out.close()


if __name__ == "__main__":
    main()
