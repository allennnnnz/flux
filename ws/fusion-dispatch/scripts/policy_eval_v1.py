################################################################################
# fusion-dispatch E3: evaluate dispatch policies against the measured map.
# Design 7. Input: <prefix>_points.csv from analyze_v1.py.
#
# Arms: A (fused), B (nccl+cublas), C (fluxag+cublas), D (fluxag+fluxgemm).
# Policies are FIT on the power-of-2 grid only and evaluated on:
#   in-sample grid, held-out M (never used for fitting), and three traces
#   (decode-heavy, prefill-heavy, mixed 70/30) drawn uniformly from measured M.
#   pi_on       always A
#   pi_off      one fixed non-fused arm (best total over the grid)
#   pi_thr      per layer: M < M* -> fixed off arm, else A (M*, arm fit on grid)
#   pi_lut      per layer: best arm of the nearest grid point in log2(M)
#   pi_lut_floor  same, largest grid point <= M
#   oracle      best measured arm at every M
# Metrics: aggregate regret sum(t_pi - t_oracle)/sum(t_oracle); worst point;
#          wrong decisions (chosen arm > 5% slower than oracle).
# Usage: python3 policy_eval_v1.py <points.csv> --mode gpu|steady [--out file.csv]
################################################################################
import argparse
import bisect
import csv
import math
import timeit
from collections import defaultdict

GRID = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
HELD = [24, 72, 136, 264, 520, 1032, 3072, 6144]
ARMS = {"A": "t_fused", "B": "t_B", "C": "t_C", "D": "t_D"}
OFF = ["B", "C", "D"]


def load(path, mode):
    t = defaultdict(dict)  # t[layer][M] = {arm: ms}
    for r in csv.DictReader(open(path)):
        if r["mode"] != mode:
            continue
        t[r["layer"]][int(r["M"])] = {a: float(r[c]) for a, c in ARMS.items() if r[c]}
    return t


def fit(tl):
    grid = [m for m in GRID if m in tl]
    total = lambda choose: sum(tl[m][choose(m)] for m in grid)  # noqa: E731
    off_arm = min(OFF, key=lambda a: total(lambda m: a))
    best_thr, best_cost = None, math.inf
    for cut in grid + [grid[-1] * 2]:  # M < cut -> off
        for a in OFF:
            c = total(lambda m: a if m < cut else "A")
            if c < best_cost:
                best_thr, best_cost = (cut, a), c
    lut = {m: min(tl[m], key=tl[m].get) for m in grid}
    lg = [math.log2(m) for m in grid]

    def nearest(M):
        x = math.log2(M)
        i = min(range(len(grid)), key=lambda j: abs(lg[j] - x))
        return lut[grid[i]]

    def floor_(M):
        i = bisect.bisect_right(grid, M) - 1
        return lut[grid[max(i, 0)]]
    cut, thr_arm = best_thr
    return {
        "pi_on": lambda M: "A",
        "pi_off": lambda M: off_arm,
        "pi_thr": lambda M: thr_arm if M < cut else "A",
        "pi_lut": nearest,
        "pi_lut_floor": floor_,
    }, {"off_arm": off_arm, "thr_cut": cut, "thr_arm": thr_arm, "lut": lut}


def evaluate(tl, pols, Ms):
    Ms = [m for m in Ms if m in tl]
    orc = {m: min(tl[m].values()) for m in Ms}
    out = {}
    for name, p in pols.items():
        tp = {m: tl[m][p(m)] for m in Ms}
        regret = sum(tp[m] - orc[m] for m in Ms) / sum(orc.values())
        worst = max(Ms, key=lambda m: (tp[m] - orc[m]) / orc[m])
        wrong = sum(1 for m in Ms if tp[m] > 1.05 * orc[m])
        out[name] = (regret, worst, (tp[worst] - orc[worst]) / orc[worst], wrong, len(Ms))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("points")
    ap.add_argument("--mode", default="gpu")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    t = load(a.points, a.mode)
    measured = sorted({m for tl in t.values() for m in tl})
    dec = [m for m in measured if m <= 512]
    pre = [m for m in measured if m >= 1024]
    rows = []
    print(f"mode={a.mode}")
    for layer, tl in sorted(t.items()):
        pols, info = fit(tl)
        pols_full = dict(pols)
        print(f"\n== {layer}: fixed off arm {info['off_arm']}; threshold: M < {info['thr_cut']} -> "
              f"{info['thr_arm']}; LUT " + " ".join(f"{m}:{v}" for m, v in info["lut"].items()))
        sets = {"grid(in-sample)": GRID, "held-out": HELD, "decode<=512": dec, "prefill>=1024": pre}
        res = {k: evaluate(tl, pols_full, v) for k, v in sets.items()}
        # mixed trace: 70% decode steps, 30% prefill chunks, uniform within each
        orc = {m: min(tl[m].values()) for m in tl}
        for name, p in pols_full.items():
            def exp(ms):
                ms = [m for m in ms if m in tl]
                return (sum(tl[m][p(m)] for m in ms) / len(ms), sum(orc[m] for m in ms) / len(ms))
            (dp, do), (pp, po) = exp(dec), exp(pre)
            mix = (0.7 * dp + 0.3 * pp - (0.7 * do + 0.3 * po)) / (0.7 * do + 0.3 * po)
            res.setdefault("mixed 70/30", {})[name] = (mix, None, None, None, None)
        print(f"  {'policy':<13}" + "".join(f"{k:>22}" for k in res))
        for name in pols_full:
            line = f"  {name:<13}"
            for k in res:
                reg, worst, wr, wrong, n = res[k][name]
                line += f"{reg * 100:>9.2f}%" + (f" ({wrong}/{n} wrong)" if wrong is not None else " " * 13)
            print(line)
            for k in res:
                reg, worst, wr, wrong, n = res[k][name]
                rows.append([a.mode, layer, name, k, f"{reg:.5f}", worst if worst else "",
                             f"{wr:.4f}" if wr is not None else "", wrong if wrong is not None else "",
                             n if n else ""])
        wm = {k: max(res[k]["pi_lut"][2] or 0, 0) for k in ["held-out"]}
        print(f"  pi_lut worst held-out point: M={res['held-out']['pi_lut'][1]} "
              f"(+{wm['held-out'] * 100:.1f}% vs oracle)")
    # decision overhead: host-side lookup cost
    grid = GRID
    lut = {m: "A" for m in grid}
    f = lambda: lut[grid[max(bisect.bisect_right(grid, 3000) - 1, 0)]]  # noqa: E731
    n = 1_000_000
    per = timeit.timeit(f, number=n) / n * 1e6
    print(f"\ndecision overhead (bisect + dict lookup, CPython): {per:.3f} us/call")
    if a.out:
        with open(a.out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["mode", "layer", "policy", "eval_set", "regret", "worst_M", "worst_rel", "wrong", "n"])
            w.writerows(rows)
            w.writerow(["overhead_us_per_call", f"{per:.4f}"])


if __name__ == "__main__":
    main()
