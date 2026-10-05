################################################################################
# fusion-dispatch E4a (2026-10-05): does the LAYOUT decision move on a slow interconnect, and does the
# decider find it? TP spans two nodes (css-host-158 + css-host-159, RoCE, ~7 GB/s per GPU), so the same
# model shapes as G4 run with half of every collective crossing the slow link. Flux paths are not used
# (Flux ops cannot run across these nodes at full width yet, see results/e4_flux_smoke/), so the
# deployable layouts are: tp_ar_vllm (vLLM default; cross-node vLLM uses PyNccl) and sp_nccl
# (sequence parallel, NCCL all_gather / reduce_scatter + cuBLAS).
#
#   fit <cal_dir> <tag>          calibrate_xnode_v1.py output -> hw_profiles/xnode158-159_<tag>_<mode>_block.json
#                                (vLLM all-reduce, NCCL all_gather / reduce_scatter curves, RMSNorm / add)
#   predict <tag> <model> <tp>   decider = predictor/block.py layout model with those curves and the
#                                single-GPU cuBLAS times already measured for these shapes
#                                (results/g4_block_gemm/, same A100 model on both nodes). Writes the
#                                pre-registered decisions of the decider and of the frozen rules:
#                                  R1  frozen 2026-10-03 (reports/20261003_g4_flux_value.md section 3):
#                                      M <= 512 vLLM default, M > 512 SP all-Flux; Flux unavailable -> R2
#                                  R2  always vLLM default
#                                  R3  always SP (here sp_nccl)
#                                  R1p registered 2026-10-05 before the oracle run (after a 10-round, 2-point
#                                      smoke test, results/e4_smoke_block/, disclosed): M <= 512 vLLM default,
#                                      M > 512 SP with the best non-Flux path (= sp_nccl). It is the "decode
#                                      vLLM, prefill SP" rule without its Flux part, the strongest simple rival.
#   eval <tag> <model> <tp>      score every policy against the oracle (best measured deployable layout).
# Profiles get their own file names: the css-host-158 single-node profiles are never overwritten.
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

PROF = os.environ.get("E4A_PROF_DIR", os.path.join(REPO, "common", "cost_model", "hw_profiles"))  # override: self-test only
OUT = os.environ.get("E4A_OUT_DIR", os.path.join(WS, "results", "e4a_layout"))  # override: self-test only
HIDDEN = {"qwen2.5-32b": 5120, "llama3-8b": 4096}
PHASES = {"decode": ("gpu", [32, 128, 256, 384, 512]), "prefill": ("steady", [1024, 2048, 4096])}
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
        nm = [(int(r["M"]), int(r["H"]), float(r["median_ms"])) for r in R if r["item"] == "rmsnorm"]
        ad = [(int(r["M"]), int(r["H"]), float(r["median_ms"])) for r in R if r["item"] == "add"]
        bp = BlockProfile.fit(W, pts["vllm_ar"], nm, ad,
                              meta={"source": os.path.relpath(cal_dir, REPO), "mode": mode, "world": W,
                                    "interconnect": "TP across css-host-158 + css-host-159 (RoCE, NCCL_IB_TC=104)"})
        d = bp.to_dict()
        d["nccl_ag"] = Curve.fit(pts["nccl_ag"]).to_dict()
        d["nccl_rs"] = Curve.fit(pts["nccl_rs"]).to_dict()
        os.makedirs(PROF, exist_ok=True)
        json.dump(d, open(prof_path(tag, mode), "w"), indent=1)
        print(f"wrote {os.path.relpath(prof_path(tag, mode), REPO)}  (W={W}, {len(pts['vllm_ar'])} AR points)")


def load(tag, mode):
    d = json.load(open(prof_path(tag, mode)))
    bp = BlockProfile(d["W"], Curve.from_dict(d["ar"]), tuple(d["rmsnorm_t0_ms_bw_Bpms"]),
                      tuple(d["add_t0_ms_bw_Bpms"]), d.get("meta"))
    return bp, Curve.from_dict(d["nccl_ag"]), Curve.from_dict(d["nccl_rs"])


def gemm(model, tp):
    g = {}
    for r in csv.DictReader(open(os.path.join(WS, "results", "g4_block_gemm", f"gemm_{model}_tp{tp}.csv"))):
        if r["item"] == "c_cublas":
            g[(r["mode"], r["layer"], int(r["M"]))] = float(r["median_ms"])
    return g


def predict(tag, model, tp):
    os.makedirs(OUT, exist_ok=True)
    H, G = HIDDEN[model], gemm(model, tp)
    rows = []
    for phase, (mode, Ms) in PHASES.items():
        bp, ag, rs = load(tag, mode)
        assert bp.W == tp, f"profile world {bp.W} != tp {tp}"
        for M in Ms:
            nb = M * H * 2
            cub = [G[(mode, l, M)] for l in ("qkv", "o", "gate_up", "down")]
            sp = [ag(nb) + cub[0], cub[1] + rs(nb), ag(nb) + cub[2], cub[3] + rs(nb)]
            t_tp, t_sp, delta = layout_times(bp, M, H, cub, sp)
            dec = "sp_nccl" if delta > 0 else "tp_ar_vllm"
            r1p = "tp_ar_vllm" if M <= 512 else "sp_nccl"
            rows.append([model, tp, phase, M, f"{t_tp:.4f}", f"{t_sp:.4f}", f"{delta:.4f}", dec,
                         "tp_ar_vllm", r1p, "tp_ar_vllm", "sp_nccl"])
    path = os.path.join(OUT, f"predictions_{tag}_{model}_tp{tp}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "tp", "phase", "M", "t_tp_pred_ms", "t_sp_pred_ms", "delta_ms", "decider",
                    "R1", "R1p", "R2", "R3"])
        w.writerows(rows)
    gpath = os.path.join(WS, "results", "g4_block_gemm", f"gemm_{model}_tp{tp}.csv")
    head = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    json.dump({"script": "ws/fusion-dispatch/scripts/e4a_layout_v1.py predict", "tag": tag, "model": model, "tp": tp,
               "git_head_before_commit": head,
               "profiles_sha256": {os.path.relpath(prof_path(tag, m), REPO): sha(prof_path(tag, m)) for m in ("gpu", "steady")},
               "gemm_sha256": {os.path.relpath(gpath, REPO): sha(gpath)},
               "rule": "frozen before the oracle run; nothing refitted afterwards"},
              open(path.replace(".csv", "_meta.json"), "w"), indent=1)
    print(f"wrote {os.path.relpath(path, REPO)}")
    for r in rows:
        print(f"  {r[2]:<7} M={r[3]:<5} TP+AR {r[4]} ms  SP {r[5]} ms  -> decider {r[7]}   (R1p {r[9]})")


def evaluate(tag, model, tp):
    pred = {(r["phase"], int(r["M"])): r for r in
            csv.DictReader(open(os.path.join(OUT, f"predictions_{tag}_{model}_tp{tp}.csv")))}
    med, kept = {}, {}
    for f in glob.glob(os.path.join(OUT, f"oracle_{tag}_{model}_tp{tp}_*", "raw_*.csv")):
        v = defaultdict(list)
        for r in csv.DictReader(open(f)):
            if r["kept"] == "1":
                v[(r["phase"], int(r["M"]), r["policy"])].append(float(r["rank_max_ms"]))
        for (ph, M, p), x in v.items():
            med.setdefault((ph, M), {})[p] = statistics.median(x)
            kept[(ph, M)] = len(x)
    out = open(os.path.join(OUT, f"eval_{tag}_{model}_tp{tp}.txt"), "w")

    def say(s=""):
        print(s, flush=True)
        out.write(s + "\n")
    say(f"E4a {tag} {model} TP={tp} (two nodes): pre-registered decisions vs measured oracle")
    say(f"{'phase':<8}{'M':>6}{'vLLM':>9}{'SP':>9}{'best':>12}{'decider':>12}{'pred dT':>9}{'meas dT':>9}  kept")
    pts = []
    for k in sorted(med, key=lambda k: (k[0] != "decode", k[1])):
        m = med[k]
        if not all(p in m for p in DEPLOY) or k not in pred:
            continue
        p = pred[k]
        orc = min(DEPLOY, key=lambda x: m[x])
        pts.append({"k": k, "t": m, "orc": orc, "ch": {"decider": p["decider"], "R1": p["R1"], "R1p": p["R1p"],
                                                        "R2": p["R2"], "R3": p["R3"]}})
        say(f"{k[0]:<8}{k[1]:>6}{m['tp_ar_vllm']:>9.3f}{m['sp_nccl']:>9.3f}{orc:>12}{p['decider']:>12}"
            f"{float(p['delta_ms']):>9.3f}{m['tp_ar_vllm'] - m['sp_nccl']:>9.3f}  {kept[k]}")
    say("\npolicy      regret   saving vs vLLM   worst point   (regret = sum(t - t_best) / sum(t_best))")
    summary = []
    for g, gf in (("all", lambda p: True), ("decode", lambda p: p["k"][0] == "decode"),
                  ("prefill", lambda p: p["k"][0] == "prefill")):
        P = [p for p in pts if gf(p)]
        if not P:
            continue
        o, v = sum(p["t"][p["orc"]] for p in P), sum(p["t"]["tp_ar_vllm"] for p in P)
        say(f"[{g}] {len(P)} points")
        for pol in ("decider", "R1", "R1p", "R2", "R3"):
            t = sum(p["t"][p["ch"][pol]] for p in P)
            wst = max(P, key=lambda p: p["t"][p["ch"][pol]] / p["t"][p["orc"]])
            wr = wst["t"][wst["ch"][pol]] / wst["t"][wst["orc"]] - 1
            say(f"  {pol:<9}{(t / o - 1) * 100:7.2f}%{(1 - t / v) * 100:13.1f}%{wr * 100:10.1f}% at {wst['k'][0]} M={wst['k'][1]}")
            summary.append([g, pol, f"{t / o - 1:.5f}", f"{1 - t / v:.5f}", f"{wr:.4f}", f"{wst['k'][0]} {wst['k'][1]}", len(P)])
    with open(os.path.join(OUT, f"eval_{tag}_{model}_tp{tp}_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "policy", "regret", "saving_vs_vllm", "worst_rel", "worst_point", "n"])
        w.writerows(summary)
    out.close()


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "fit":
        fit(sys.argv[2], sys.argv[3])
    elif cmd == "predict":
        predict(sys.argv[2], sys.argv[3], int(sys.argv[4]))
    elif cmd == "eval":
        evaluate(sys.argv[2], sys.argv[3], int(sys.argv[4]))
