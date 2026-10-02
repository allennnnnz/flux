################################################################################
# fusion-dispatch analysis: raw per-round CSVs -> per-point verdicts.
# Reads raw_<layer>.csv written by dispatch_map_v1.py. Kept rounds only.
#
# For every (layer, M, mode):
#   * per-item rank-max median / p10 / p90
#   * best off path = argmin median over {B, C, D}
#   * delta = per-round (A - best_off), median / p10 / p90
#     verdict: 'off' if p10 > 0, 'fused' if p90 < 0, else 'tie' (design 5.6)
#   * decomposition medians (design 6): A-D, D-C, C-B
#   * self-consistency (design 5.4): composite vs sum of its components,
#     and A below max(own comm, own gemm) -> faster than perfect overlap -> STOP
#
# Usage: python3 analyze_v1.py <results_dir> [<results_dir> ...] [--out summary_prefix]
################################################################################

import argparse
import csv
import glob
import os
import statistics
from collections import defaultdict

OFF = ["B_nccl_cublas", "C_fluxag_cublas", "D_fluxag_fluxgemm"]
COMPOSITES = [  # (composite, components) measured in the same round; checked when all present
    ("D_fluxag_fluxgemm", ["c_flux_ag", "c_fluxgemm"]),
    ("B_nccl_cublas", ["c_nccl_ag", "c_cublas"]),      # AG side (dispatch_map_v2)
    ("C_fluxag_cublas", ["c_flux_ag", "c_cublas"]),
    ("B_nccl_cublas", ["c_nccl_rs", "c_cublas"]),      # RS side (dispatch_map_rs_v1)
    ("E_allreduce", ["c_nccl_ar", "c_cublas"]),
]


def q(v, p):
    s = sorted(v)
    return s[max(0, min(len(s) - 1, int(p * len(s))))]


def load(dirs):
    # data[(layer,N,K,M,mode)][item][round] = rank_max_ms ; cpu likewise
    data = defaultdict(lambda: defaultdict(dict))
    cpu = defaultdict(lambda: defaultdict(dict))
    clk = defaultdict(list)
    for d in dirs:
        for path in sorted(glob.glob(os.path.join(d, "raw_*.csv"))):
            with open(path) as f:
                for row in csv.DictReader(f):
                    if row["kept"] != "1":
                        continue
                    key = (row["layer"], int(row["N"]), int(row["K"]), int(row["M"]), row["mode"])
                    r = int(row["round"])
                    data[key][row["item"]][r] = float(row["rank_max_ms"])
                    cpu[key][row["item"]][r] = float(row["cpu_launch_rank_max_ms"])
                    if row["item"] == "A_fused" or "A_fused" not in data[key]:
                        pass
                    clk[key].append(int(row["min_sm_clock_mhz"]))
    return data, cpu, clk


def med(d):
    return statistics.median(d.values()) if d else float("nan")


def per_round(a, bs):
    rounds = set(a)
    for b in bs:
        rounds &= set(b)
    return [a[r] - sum(b[r] for b in bs) for r in sorted(rounds)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    data, cpu, clk = load(args.dirs)

    item_rows, point_rows, flags = [], [], []
    for key in sorted(data, key=lambda k: (k[0], k[4], k[3])):
        layer, N, K, M, mode = key
        items = data[key]
        clocks = [c for c in clk[key] if c > 0]
        modal_clk = statistics.mode(clocks) if clocks else None
        for it, d in sorted(items.items()):
            v = list(d.values())
            cv = list(cpu[key][it].values())
            item_rows.append([layer, N, K, M, mode, it, len(v), f"{statistics.median(v):.4f}",
                              f"{q(v, .1):.4f}", f"{q(v, .9):.4f}", f"{statistics.median(cv):.4f}"])
        if "A_fused" not in items:
            continue
        offs = [o for o in OFF if o in items]
        best = min(offs, key=lambda o: med(items[o]))
        dl = per_round(items["A_fused"], [items[best]])
        p10, p50, p90 = q(dl, .1), statistics.median(dl), q(dl, .9)
        verdict = "off" if p10 > 0 else ("fused" if p90 < 0 else "tie")
        tA, tb = med(items["A_fused"]), med(items[best])

        def dmed(a, b):
            if a in items and b in items:
                return statistics.median(per_round(items[a], [items[b]]))
            return float("nan")
        row = [layer, N, K, M, mode, modal_clk, f"{tA:.4f}", best, f"{tb:.4f}",
               f"{p50:.4f}", f"{p10:.4f}", f"{p90:.4f}", verdict, f"{tb / tA:.3f}",
               f"{dmed('A_fused', 'D_fluxag_fluxgemm'):.4f}",
               f"{dmed('D_fluxag_fluxgemm', 'C_fluxag_cublas'):.4f}",
               f"{dmed('C_fluxag_cublas', 'B_nccl_cublas'):.4f}"]
        for o in ["B_nccl_cublas", "C_fluxag_cublas", "D_fluxag_fluxgemm"]:
            row.append(f"{med(items[o]):.4f}" if o in items else "")
        for c in ["c_nccl_ag", "c_flux_ag", "c_cublas", "c_fluxgemm"]:
            row.append(f"{med(items[c]):.4f}" if c in items else "")
        point_rows.append(row)

        # self-consistency checks
        for comp, parts in COMPOSITES:
            if comp in items and all(p in items for p in parts):
                dd = statistics.median(per_round(items[comp], [items[p] for p in parts]))
                rel = dd / med(items[comp])
                if abs(rel) > 0.10:
                    flags.append(f"CONSISTENCY {layer} M={M} {mode}: {comp} - sum(parts) = "
                                 f"{dd:.4f} ms ({rel:+.1%})")
        if "c_flux_ag" in items and "c_fluxgemm" in items:
            floor = max(med(items["c_flux_ag"]), med(items["c_fluxgemm"]))
            if tA < 0.98 * floor:
                flags.append(f"STOP {layer} M={M} {mode}: A={tA:.4f} < max(flux_ag, fluxgemm)="
                             f"{floor:.4f} (faster than perfect overlap)")

    hdr_p = ["layer", "N", "K", "M", "mode", "modal_sm_clock", "t_fused", "best_off", "t_best_off",
             "delta_med(A-best)", "delta_p10", "delta_p90", "verdict", "speedup_off_over_fused",
             "A-D", "D-C", "C-B", "t_B", "t_C", "t_D",
             "c_nccl_ag", "c_flux_ag", "c_cublas", "c_fluxgemm"]
    hdr_i = ["layer", "N", "K", "M", "mode", "item", "n_kept", "median_ms", "p10_ms", "p90_ms",
             "cpu_launch_med_ms"]
    w = max(len(h) for h in hdr_p)
    print(" | ".join(["layer", "M", "mode", "clk", "fused", "best_off", "t_off", "Δ(A-off)",
                      "p10", "p90", "verdict", "A-D", "D-C", "C-B"]))
    for r in point_rows:
        print(" | ".join(str(x) for x in [r[0], r[3], r[4], r[5], r[6], r[7], r[8], r[9], r[10],
                                          r[11], r[12], r[14], r[15], r[16]]))
    for f_ in flags:
        print(f_)
    if args.out:
        with open(args.out + "_points.csv", "w", newline="") as f:
            cw = csv.writer(f)
            cw.writerow(hdr_p)
            cw.writerows(point_rows)
        with open(args.out + "_items.csv", "w", newline="") as f:
            cw = csv.writer(f)
            cw.writerow(hdr_i)
            cw.writerows(item_rows)
        with open(args.out + "_flags.txt", "w") as f:
            f.write("\n".join(flags) + ("\n" if flags else ""))
        print(f"wrote {args.out}_points.csv / _items.csv / _flags.txt")


if __name__ == "__main__":
    main()
