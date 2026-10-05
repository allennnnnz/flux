################################################################################
# fusion-dispatch E4a2 (2026-10-05): layout decision on the slow interconnect, decider v2-xnode.
# Changes vs e4a_layout_v1.py, each from a failure located in E4a (reports/20261005_e4a_slow_link_layout.md):
#   1. communication curves are NOT forced monotone and nearby sizes are NOT merged (cross-node NCCL
#      all-reduce has cliffs: TP4 1.5 MiB 0.750 ms vs 3 MiB 0.404 ms; pooling smeared them)
#   2. calibration at the exact message sizes the block uses (calibrate_xnode_v2.py --cases)
#   3. G4's probe rule is ON: |predicted delta| < EPS x t_sp -> short layout probe (30 rounds, both layouts),
#      the faster one is the final decision. EPS = 3% (G4's lower bound), fixed here before any E4a2 data.
#   4. FRESH test points: decode (CUDA graph) M = 64, 192, 320, 448; prefill (eager) M = 768, 1536, 3072.
#      The 24 E4a points are only a sanity check now (their answers were seen while designing 1-3).
# Subcommands:
#   fit <cal_dir> <tag>
#   predict <tag> <model> <tp> <set: new|old>       -> predictions_<tag>_<model>_tp<tp>_<set>.csv
#   finalize <tag> <model> <tp>                      -> decisions_<tag>_<model>_tp<tp>_new.csv (after probes)
#   eval <tag> <model> <tp> <set> [old_oracle_tag]   -> eval_<tag>_<model>_tp<tp>_<set>.txt / _summary.csv
# Frozen rules as in v1: R1 (10-03; Flux unavailable -> R2), R1p (decode vLLM, prefill SP), R2 (vLLM), R3 (SP).
################################################################################
import csv
import glob
import hashlib
import json
import os
import statistics
import subprocess
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(WS))
sys.path.insert(0, os.path.join(REPO, "common", "cost_model"))
from predictor.block import BlockProfile, layout_times  # noqa: E402
from predictor.curves import Curve  # noqa: E402

PROF = os.environ.get("E4A_PROF_DIR", os.path.join(REPO, "common", "cost_model", "hw_profiles"))  # override: self-test
OUT = os.environ.get("E4A_OUT_DIR", os.path.join(WS, "results", "e4a2_layout"))                  # override: self-test
OLD = os.path.join(WS, "results", "e4a_layout")
HIDDEN = {"qwen2.5-32b": 5120, "llama3-8b": 4096}
SETS = {"new": {"decode": [64, 192, 320, 448], "prefill": [768, 1536, 3072]},
        "old": {"decode": [32, 128, 256, 384, 512], "prefill": [1024, 2048, 4096]}}
MODE = {"decode": "gpu", "prefill": "steady"}
EPS = 0.03
DEPLOY = ["tp_ar_vllm", "sp_nccl"]


def prof_path(tag, mode):
    return os.path.join(PROF, f"xnode158-159_{tag}_{mode}_block.json")


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def fit(cal_dir, tag):
    meta = json.load(open(os.path.join(cal_dir, "meta_xnode_cal.json")))
    W = meta["world"]
    rows = list(csv.DictReader(open(os.path.join(cal_dir, "summary_xnode_cal.csv"))))
    for mode in ("gpu", "steady"):
        R = [r for r in rows if r["mode"] == mode]
        pts = {it: [(int(r["M"]) * int(r["H"]) * 2, float(r["median_ms"])) for r in R if r["item"] == it]
               for it in ("vllm_ar", "nccl_ag", "nccl_rs")}
        norm = BlockProfile._linfit([(int(r["M"]) * int(r["H"]) * 4, float(r["median_ms"])) for r in R if r["item"] == "rmsnorm"])
        add = BlockProfile._linfit([(int(r["M"]) * int(r["H"]) * 6, float(r["median_ms"])) for r in R if r["item"] == "add"])
        curve = {k: Curve.fit(v, rel_merge=0.0, monotone=False) for k, v in pts.items()}
        bp = BlockProfile(W, curve["vllm_ar"], norm, add,
                          {"source": os.path.relpath(cal_dir, REPO), "mode": mode, "world": W, "monotone": False,
                           "rel_merge": 0.0, "interconnect": "TP across css-host-158 + css-host-159 (RoCE, NCCL_IB_TC=104)"})
        d = bp.to_dict()
        d["nccl_ag"], d["nccl_rs"] = curve["nccl_ag"].to_dict(), curve["nccl_rs"].to_dict()
        json.dump(d, open(prof_path(tag, mode), "w"), indent=1)
        print(f"wrote {os.path.relpath(prof_path(tag, mode), REPO)}  (W={W}, {len(curve['vllm_ar'].xs)} AR sizes)")


def load(tag, mode):
    d = json.load(open(prof_path(tag, mode)))
    bp = BlockProfile(d["W"], Curve.from_dict(d["ar"]), tuple(d["rmsnorm_t0_ms_bw_Bpms"]), tuple(d["add_t0_ms_bw_Bpms"]),
                      d.get("meta"))
    return bp, Curve.from_dict(d["nccl_ag"]), Curve.from_dict(d["nccl_rs"])


def gemm(model, tp):
    g, srcs = {}, []
    for p in (os.path.join(WS, "results", "g4_block_gemm", f"gemm_{model}_tp{tp}.csv"),
              os.path.join(OUT, "gemm", f"gemm_{model}_tp{tp}.csv")):
        if os.path.exists(p):
            srcs.append(p)
            for r in csv.DictReader(open(p)):
                if r["item"] == "c_cublas":
                    g[(r["mode"], r["layer"], int(r["M"]))] = float(r["median_ms"])
    return g, srcs


def predict(tag, model, tp, which):
    H, (G, srcs) = HIDDEN[model], gemm(model, tp)
    rows = []
    for phase, Ms in SETS[which].items():
        bp, ag, rs = load(tag, MODE[phase])
        assert bp.W == tp
        for M in Ms:
            nb = M * H * 2
            cub = [G[(MODE[phase], l, M)] for l in ("qkv", "o", "gate_up", "down")]
            sp = [ag(nb) + cub[0], cub[1] + rs(nb), ag(nb) + cub[2], cub[3] + rs(nb)]
            t_tp, t_sp, delta = layout_times(bp, M, H, cub, sp)
            dec = "sp_nccl" if delta > 0 else "tp_ar_vllm"
            probe = int(abs(delta) < EPS * t_sp)
            rows.append([model, tp, phase, M, f"{t_tp:.4f}", f"{t_sp:.4f}", f"{delta:.4f}", dec, probe,
                         "tp_ar_vllm", "tp_ar_vllm" if M <= 512 else "sp_nccl", "tp_ar_vllm", "sp_nccl"])
    path = os.path.join(OUT, f"predictions_{tag}_{model}_tp{tp}_{which}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "tp", "phase", "M", "t_tp_pred_ms", "t_sp_pred_ms", "delta_ms", "decider_model", "probe",
                    "R1", "R1p", "R2", "R3"])
        w.writerows(rows)
    head = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    json.dump({"script": "ws/fusion-dispatch/scripts/e4a_layout_v2.py predict", "tag": tag, "set": which, "eps": EPS,
               "git_head_before_commit": head,
               "profiles_sha256": {os.path.relpath(prof_path(tag, m), REPO): sha(prof_path(tag, m)) for m in ("gpu", "steady")},
               "gemm_sha256": {os.path.relpath(p, REPO): sha(p) for p in srcs},
               "rule": "frozen before the probe and oracle runs; nothing refitted afterwards"},
              open(path.replace(".csv", "_meta.json"), "w"), indent=1)
    print(f"wrote {os.path.relpath(path, REPO)}")
    for r in rows:
        print(f"  {r[2]:<7} M={r[3]:<5} TP+AR {r[4]}  SP {r[5]}  delta {r[6]:>8} -> {r[7]:<10} probe={r[8]}  (R1p {r[10]})")
    probes = {ph: [r[3] for r in rows if r[2] == ph and r[8]] for ph in SETS[which]}
    with open(path.replace(".csv", "_probes.txt"), "w") as f:
        for ph, Ms in probes.items():
            f.write(f"{ph} {','.join(map(str, Ms))}\n")


def medians(pattern):
    med = defaultdict(dict)
    for f in glob.glob(pattern):
        v = defaultdict(list)
        for r in csv.DictReader(open(f)):
            if r["kept"] == "1":
                v[(r["phase"], int(r["M"]), r["policy"])].append(float(r["rank_max_ms"]))
        for (ph, M, p), x in v.items():
            med[(ph, M)][p] = statistics.median(x)
    return med


def finalize(tag, model, tp):
    pred = list(csv.DictReader(open(os.path.join(OUT, f"predictions_{tag}_{model}_tp{tp}_new.csv"))))
    pm = medians(os.path.join(OUT, f"probe_{tag}_{model}_tp{tp}_*", "raw_*.csv"))
    out = []
    for r in pred:
        k = (r["phase"], int(r["M"]))
        final, note = r["decider_model"], ""
        if r["probe"] == "1":
            m = pm.get(k, {})
            if all(p in m for p in DEPLOY):
                final = min(DEPLOY, key=lambda p: m[p])
                note = f"probe tp_ar_vllm={m['tp_ar_vllm']:.4f} sp_nccl={m['sp_nccl']:.4f}"
            else:
                note = "probe MISSING (model decision kept)"
        out.append([r["model"], r["tp"], r["phase"], r["M"], r["decider_model"], final, note])
    path = os.path.join(OUT, f"decisions_{tag}_{model}_tp{tp}_new.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "tp", "phase", "M", "decider_model", "decider", "note"])
        w.writerows(out)
    print(f"wrote {os.path.relpath(path, REPO)}")
    for r in out:
        print(f"  {r[2]:<7} M={r[3]:<5} model {r[4]:<10} final {r[5]:<10} {r[6]}")


def evaluate(tag, model, tp, which, old_tag=None):
    pred = {(r["phase"], int(r["M"])): r for r in
            csv.DictReader(open(os.path.join(OUT, f"predictions_{tag}_{model}_tp{tp}_{which}.csv")))}
    if which == "new":
        fin = {(r["phase"], int(r["M"])): r["decider"] for r in
               csv.DictReader(open(os.path.join(OUT, f"decisions_{tag}_{model}_tp{tp}_new.csv")))}
        med = medians(os.path.join(OUT, f"oracle_{tag}_{model}_tp{tp}_*", "raw_*.csv"))
        pols = ["decider", "decider_model", "R1", "R1p", "R2", "R3"]
    else:  # sanity: new model on the E4a points, against the E4a oracle (no new measurement)
        fin = None
        med = medians(os.path.join(OLD, f"oracle_{old_tag}_{model}_tp{tp}_*", "raw_*.csv"))
        oldp = {(r["phase"], int(r["M"])): r["decider"] for r in
                csv.DictReader(open(os.path.join(OLD, f"predictions_{old_tag}_{model}_tp{tp}.csv")))}
        pols = ["decider_model", "decider_e4a_v1", "R1", "R1p", "R2", "R3"]
    out = open(os.path.join(OUT, f"eval_{tag}_{model}_tp{tp}_{which}.txt"), "w")

    def say(s=""):
        print(s, flush=True)
        out.write(s + "\n")
    say(f"E4a2 {tag} {model} TP={tp} set={which} ({'FRESH points, pre-registered' if which == 'new' else 'SANITY ONLY: E4a points, answers seen before the model change'})")
    say(f"{'phase':<8}{'M':>6}{'vLLM':>9}{'SP':>9}{'best':>12}{'model':>12}{'final':>12}{'pred dT':>9}{'meas dT':>9} probe")
    pts = []
    for k in sorted(med, key=lambda k: (k[0] != "decode", k[1])):
        m = med[k]
        if not all(p in m for p in DEPLOY) or k not in pred:
            continue
        p = pred[k]
        ch = {"decider_model": p["decider_model"], "R1": p["R1"], "R1p": p["R1p"], "R2": p["R2"], "R3": p["R3"]}
        if which == "new":
            ch["decider"] = fin[k]
        else:
            ch["decider_e4a_v1"] = oldp[k]
        orc = min(DEPLOY, key=lambda x: m[x])
        pts.append({"k": k, "t": m, "orc": orc, "ch": ch})
        say(f"{k[0]:<8}{k[1]:>6}{m['tp_ar_vllm']:>9.3f}{m['sp_nccl']:>9.3f}{orc:>12}{p['decider_model']:>12}"
            f"{ch.get('decider', ch.get('decider_e4a_v1')):>12}{float(p['delta_ms']):>9.3f}{m['tp_ar_vllm'] - m['sp_nccl']:>9.3f} {p['probe']}")
    say("\npolicy           regret   saving vs vLLM   worst point   right")
    summary = []
    for g, gf in (("all", lambda p: True), ("decode", lambda p: p["k"][0] == "decode"),
                  ("prefill", lambda p: p["k"][0] == "prefill")):
        P = [p for p in pts if gf(p)]
        if not P:
            continue
        o, v = sum(p["t"][p["orc"]] for p in P), sum(p["t"]["tp_ar_vllm"] for p in P)
        say(f"[{g}] {len(P)} points")
        for pol in pols:
            t = sum(p["t"][p["ch"][pol]] for p in P)
            wst = max(P, key=lambda p: p["t"][p["ch"][pol]] / p["t"][p["orc"]])
            wr = wst["t"][wst["ch"][pol]] / wst["t"][wst["orc"]] - 1
            right = sum(p["ch"][pol] == p["orc"] for p in P)
            say(f"  {pol:<15}{(t / o - 1) * 100:7.2f}%{(1 - t / v) * 100:13.1f}%{wr * 100:9.1f}% at {wst['k'][0]} M={wst['k'][1]:<5}{right:>4}/{len(P)}")
            summary.append([g, pol, f"{t / o - 1:.5f}", f"{1 - t / v:.5f}", f"{wr:.4f}", f"{wst['k'][0]} {wst['k'][1]}", right, len(P)])
    with open(os.path.join(OUT, f"eval_{tag}_{model}_tp{tp}_{which}_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "policy", "regret", "saving_vs_vllm", "worst_rel", "worst_point", "right", "n"])
        w.writerows(summary)
    out.close()


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    c, a = sys.argv[1], sys.argv[2:]
    if c == "fit":
        fit(a[0], a[1])
    elif c == "predict":
        predict(a[0], a[1], int(a[2]), a[3])
    elif c == "finalize":
        finalize(a[0], a[1], int(a[2]))
    elif c == "eval":
        evaluate(a[0], a[1], int(a[2]), a[3], a[4] if len(a) > 4 else None)
