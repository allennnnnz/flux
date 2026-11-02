################################################################################
# fusion-dispatch C1 (2026-11-02): the Compass selector (arXiv 2609.10549, Algorithm 1, Eq. 2-5) re-implemented from
# the paper (no code is published), and the C1 evaluation against our decider.
#   fit <cal_dir> <model> <tp> <tag>
#       eta_obs(d) = (T_dec(d) - (T_comp + T_comm)/d) / ((d-1) * max(T_comp, T_comm)/d)  (Eq. 2 solved for eta) for every
#       calibration (op, M, d >= 2); least squares eta = alpha + beta*d (Eq. 3) per kind (ag, rs) -> compass_params_*.json
#   predict <tag> <model> <tp> <e4a3_tag>
#       per test point and op: T_comp = measured single-GPU cuBLAS at that M (the same files our decider used),
#       T_comm = our calibrated NCCL curve at the exact size (the same profiles our decider used) -> both selectors get
#       identical inputs; lambda = T_comp/T_comm; d* = P2(sqrt((1 - alpha + 1/lambda)/beta)) (Eq. 5, P2 = nearest power
#       of two), clamped to {1,2,4,8,16} dividing M/W; kept only if T_dec(d*) < T_dec(1) (Eq. 4), else 1.
#       Fusion (TA-IntraFusion) does not exist across nodes -> Compass reduces to choosing d (Algorithm 1 line 16 moot).
#       -> compass_table_*.json (validate_block_v5 --compass_table), compass_choices_*.csv
#       prefill: oracle = min over every measured deploy policy (tp_ar_vllm, sp_nccl, sp_dec*, sp_compass);
#       decode: E4a3's oracle (tp_ar_vllm vs sp_nccl), Compass = sp_nccl wherever its decode choices are all d = 1.
#       Policies: decider (E4a3 final), compass, decider+compass (decider's layout; inside SP, Compass's d), vLLM default,
#       SP d=1, fixed d (sp_dec<d>).
################################################################################
import csv
import glob
import json
import math
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(WS))
sys.path.insert(0, HERE)
import e4a_layout_v2 as e4  # noqa: E402  (E4A_OUT_DIR must point at the E4a3 results)

OUT = os.environ.get("C1_OUT_DIR", os.path.join(WS, "results", "c1_compass"))
MODELS = {"qwen2.5-32b": (5120, 40, 8, 128, 27648), "llama3-8b": (4096, 32, 8, 128, 14336)}
DS = [1, 2, 4, 8, 16]
LAYER = {"qkv": "qkv", "o": "o", "gate_up": "gate_up", "down": "down"}


def ops_of(model, W):
    H, NQ, NKV, HD, FFN = MODELS[model]
    return {"qkv": ("ag", (NQ // W + 2 * NKV // W) * HD, H), "gate_up": ("ag", 2 * FFN // W, H),
            "o": ("rs", H, NQ // W * HD), "down": ("rs", H, FFN // W)}


def t_dec(d, tc, tm, a, b):
    if d == 1:
        return tc + tm
    return (d - 1) * (max(tc, tm) / d) * (a + b * d) + tc / d + tm / d


def fit(cal_dir, model, tp, tag):
    rows = list(csv.DictReader(open(os.path.join(cal_dir, f"compass_cal_{model}.csv"))))
    t = {(r["op"], int(r["M"]), r["item"]): float(r["median_ms"]) for r in rows}
    kinds = {r["op"]: r["kind"] for r in rows}
    out = {}
    for kind in ("ag", "rs"):
        xs, ys = [], []
        for (op, M, item), v in t.items():
            if kinds[op] != kind or not item.startswith("dec"):
                continue
            d = int(item[3:])
            tc, tm = t[(op, M, "comp")], t[(op, M, "comm")]
            eta = (v - (tc + tm) / d) / ((d - 1) * max(tc, tm) / d)
            xs.append(d)
            ys.append(eta)
        n = len(xs)
        mx, my = sum(xs) / n, sum(ys) / n
        sxx = sum((x - mx) ** 2 for x in xs)
        b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
        a = my - b * mx
        ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
        ss_tot = sum((y - my) ** 2 for y in ys)
        out[kind] = {"alpha": a, "beta": b, "n": n, "r2": 1 - ss_res / ss_tot if ss_tot else None,
                     "eta_by_d": {str(d): statistics.median([y for x, y in zip(xs, ys) if x == d]) for d in sorted(set(xs))}}
    out["source"] = os.path.relpath(cal_dir, REPO)
    path = os.path.join(OUT, f"compass_params_{tag}_{model}_tp{tp}.json")
    json.dump(out, open(path, "w"), indent=1)
    print(f"wrote {os.path.relpath(path, REPO)}: " + "; ".join(
        f"{k} alpha={out[k]['alpha']:.3f} beta={out[k]['beta']:.4f} r2={out[k]['r2']:.2f} n={out[k]['n']}" for k in ("ag", "rs")))


def choose(tc, tm, a, b, m):
    lam = tc / tm
    arg = (1 - a + 1 / lam) / b if b > 0 else -1
    if arg <= 0:
        return 1, lam, None
    dc = 2 ** round(math.log2(math.sqrt(arg)))
    allowed = [d for d in DS if d <= dc and m % d == 0]
    d = max(allowed) if allowed else 1
    if d > 1 and not t_dec(d, tc, tm, a, b) < t_dec(1, tc, tm, a, b):
        d = 1
    return d, lam, dc


def predict(tag, model, tp, e4a3_tag):
    H = MODELS[model][0]
    P = json.load(open(os.path.join(OUT, f"compass_params_{tag}_{model}_tp{tp}.json")))
    G, gsrc = e4.gemm(model, tp)
    ops = ops_of(model, tp)
    table, rows = {}, []
    for phase, Ms in e4.SETS["new"].items():
        bp, ag, rs = e4.load(e4a3_tag, e4.MODE[phase])
        for M in Ms:
            nb = M * H * 2
            for op, (kind, n, k) in ops.items():
                tc = G[(e4.MODE[phase], LAYER[op], M)]
                tm = ag(nb) if kind == "ag" else rs(nb)
                a, b = P[kind]["alpha"], P[kind]["beta"]
                d, lam, dc = choose(tc, tm, a, b, M // tp)
                table.setdefault(str(M), {})[f"{kind}:{n}:{k}"] = d
                rows.append([model, tp, phase, M, op, kind, n, k, f"{tc:.4f}", f"{tm:.4f}", f"{lam:.3f}",
                             dc, d, f"{t_dec(d, tc, tm, a, b):.4f}", f"{t_dec(1, tc, tm, a, b):.4f}"])
    jp = os.path.join(OUT, f"compass_table_{tag}_{model}_tp{tp}.json")
    json.dump(table, open(jp, "w"), indent=1)
    cp = os.path.join(OUT, f"compass_choices_{tag}_{model}_tp{tp}.csv")
    with open(cp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "tp", "phase", "M", "op", "kind", "n", "k", "T_comp_ms", "T_comm_ms", "lambda",
                    "d_cand_eq5", "d_final", "T_dec_pred_ms", "T_seq_pred_ms"])
        w.writerows(rows)
    print(f"wrote {os.path.relpath(jp, REPO)}, {os.path.relpath(cp, REPO)}")
    for r in rows:
        print(f"  {r[2]:<7} M={r[3]:<5} {r[4]:<8} Tcomp {r[8]} Tcomm {r[9]} lambda {r[10]:>7} d_eq5={r[11]} -> d={r[12]}")


def medians(pattern):
    v = {}
    for f in glob.glob(pattern):
        for r in csv.DictReader(open(f)):
            if r["kept"] == "1":
                v.setdefault((int(r["M"]), r["policy"]), []).append(float(r["rank_max_ms"]))
    return {k: statistics.median(x) for k, x in v.items()}


def evaluate(tag, model, tp, e4a3_tag):
    """Both phases from the C1 oracle runs (every policy interleaved in one run per phase); oracle = min over all
    measured deploy policies (tp_ar, the torch-bridge correctness reference, excluded as in E4a)."""
    dec = {(r["phase"], int(r["M"])): r["decider"] for r in csv.DictReader(
        open(os.path.join(e4.OUT, f"decisions_{e4a3_tag}_{model}_tp{tp}_new.csv")))}
    pts = []
    for phase in ("decode", "prefill"):
        med = medians(os.path.join(OUT, f"oracle_{tag}_{model}_tp{tp}_{phase}", "raw_*.csv"))
        for M in sorted({M for M, _ in med}):
            t = {p: v for (m_, p), v in med.items() if m_ == M and p != "tp_ar"}
            sp_in = "sp_compass" if phase == "prefill" else "sp_nccl"
            c = {"decider": dec[(phase, M)], "compass (literal)": "sp_compass", "compass (prefill only)": sp_in,
                 "decider+compass": sp_in if dec[(phase, M)] == "sp_nccl" else "tp_ar_vllm",
                 "vLLM default": "tp_ar_vllm", "SP d=1": "sp_nccl"}
            for d in DS[1:]:
                c[f"SP d={d}"] = f"sp_dec{d}" if f"sp_dec{d}" in t else None
            pts.append({"k": (phase, M), "t": t, "c": c})
    return pts


def report(tag, cfgs):
    allp, notes = [], []
    for model, tp, e4a3_tag in cfgs:
        pts = evaluate(tag, model, tp, e4a3_tag)
        for p in pts:
            p["cfg"] = f"{model} TP{tp}"
        allp += pts
    pols = ["decider", "compass (literal)", "compass (prefill only)", "decider+compass", "vLLM default", "SP d=1"] + \
        [f"SP d={d}" for d in DS[1:]]
    L = [f"C1 {tag}: Compass selector vs our decider (cross-node, E4a3 points; decider = E4a3's pre-registered decisions)",
         "oracle per point = min over all policies measured in the same interleaved run (vLLM default, SP d=1, SP d=2..16",
         "where d divides M/W, Compass's per-op d); regret = sum(chosen)/sum(best) - 1",
         f"{'policy':<17}{'regret':>8}{'worst':>9}  {'where':<34}{'right':>6}"]
    res = {}
    for g, gf in (("all", lambda p: True), ("decode", lambda p: p["k"][0] == "decode"),
                  ("prefill", lambda p: p["k"][0] == "prefill")):
        L.append(f"[{g}]")
        for pol in pols:
            P = [p for p in allp if gf(p) and p["c"].get(pol) is not None and p["c"][pol] in p["t"]]
            if len([p for p in allp if gf(p)]) != len(P):
                L.append(f"  {pol:<15}  (not defined on every point: {len(P)}/{len([p for p in allp if gf(p)])})")
                continue
            o = sum(min(p["t"].values()) for p in P)
            s = sum(p["t"][p["c"][pol]] for p in P)
            w = max(P, key=lambda p: p["t"][p["c"][pol]] / min(p["t"].values()))
            wr = w["t"][w["c"][pol]] / min(w["t"].values()) - 1
            right = sum(p["t"][p["c"][pol]] == min(p["t"].values()) for p in P)
            res[(g, pol)] = (s / o - 1, wr, right, len(P))
            L.append(f"  {pol:<15}{(s / o - 1) * 100:7.2f}%{wr * 100:8.1f}%  {w['cfg']} {w['k'][0]} M={w['k'][1]:<14}{right:>3}/{len(P)}")
    L.append("")
    L.append("points (ms, 4-layer block, rank-max median of 200 interleaved rounds):")
    for p in allp:
        if True:
            best = min(p["t"], key=p["t"].get)
            L.append(f"  {p['cfg']:<18} M={p['k'][1]:<5} " + " ".join(f"{q}={v:.3f}" for q, v in sorted(p["t"].items()))
                     + f"  best={best} decider={p['c']['decider']}")
    out = "\n".join(L)
    print(out)
    open(os.path.join(OUT, f"eval_{tag}.txt"), "w").write(out + "\n")
    return res


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    c, a = sys.argv[1], sys.argv[2:]
    if c == "fit":
        fit(a[0], a[1], int(a[2]), a[3])
    elif c == "predict":
        predict(a[0], a[1], int(a[2]), a[3])
    elif c == "report":  # report <tag> <model:tp:e4a3_tag,...>
        cfgs = [(x.split(":")[0], int(x.split(":")[1]), x.split(":")[2]) for x in a[1].split(",")]
        report(a[0], cfgs)
