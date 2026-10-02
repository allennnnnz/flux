################################################################################
# fusion-dispatch G1: load the existing op-level measurements as one table of rows.
# AG side: results/final_ag_points.csv (E1 + E1 rerun + V7 merged; components c_* per point).
# RS side: results/final_rs_points.csv; c_nccl_rs / c_nccl_ar / E_allreduce are not in that file,
#          so they are read from the summary_items.csv of the run each point came from ('source').
# Flux config (registry hit or fallback tile) is attached per point.
# Layer shapes: AG (N, K) from dispatch_map_v2.LAYERS, each rank (N/8, K);
#               RS (N, K) from dispatch_map_rs_v1.LAYERS, each rank (N, K/8).
################################################################################
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(WS))
sys.path.insert(0, os.path.join(REPO, "common", "cost_model"))
from predictor import FluxConfigs  # noqa: E402

AG_LAYERS = {"G-FC1": (49152, 12288), "G-QKV": (36864, 12288), "L-QKV": (10240, 8192),
             "L-GU": (57344, 8192)}
RS_LAYERS = {"G-FC2": (12288, 49152), "G-O": (12288, 12288), "L-down": (8192, 28672),
             "L-O": (8192, 8192)}
W = 8
GRID = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
HELD = [24, 72, 136, 264, 520, 1032, 3072, 6144]


def _f(x):
    return float(x) if x not in ("", None, "nan") else None


def load_rows(ws=WS, tp=W):
    cfgs = FluxConfigs(tp)
    rows = []
    for r in csv.DictReader(open(os.path.join(ws, "results", "final_ag_points.csv"))):
        N, K = AG_LAYERS[r["layer"]]
        M, n = int(r["M"]), N // tp
        rows.append(dict(
            side="ag", layer=r["layer"], model=r["layer"][0], M=M, mode=r["mode"],
            clock=int(r["modal_sm_clock"]), n=n, K=K, verdict=r["verdict"],
            arms={"A": _f(r["t_fused"]), "B": _f(r["t_B"]), "C": _f(r["t_C"]), "D": _f(r["t_D"])},
            comps={"nccl": _f(r["c_nccl_ag"]), "fluxag": _f(r["c_flux_ag"]),
                   "cublas": _f(r["c_cublas"]), "fluxgemm": _f(r["c_fluxgemm"])},
            cfg=cfgs.get("ag", M, n, K), source=r["source"]))
    cache = {}
    for r in csv.DictReader(open(os.path.join(ws, "results", "final_rs_points.csv"))):
        src = r["source"]
        if src not in cache:
            p = os.path.join(ws, src + "_items.csv")
            cache[src] = {(x["layer"], x["M"], x["mode"], x["item"]): float(x["median_ms"])
                          for x in csv.DictReader(open(p))}
        it = cache[src]
        key = (r["layer"], r["M"], r["mode"])
        N, K = RS_LAYERS[r["layer"]]
        M, k = int(r["M"]), K // tp
        rows.append(dict(
            side="rs", layer=r["layer"], model=r["layer"][0], M=M, mode=r["mode"],
            clock=int(r["modal_sm_clock"]), N=N, k=k, verdict=r["verdict"],
            arms={"A": _f(r["t_fused"]), "B": _f(r["t_B"])},
            comps={"nccl_rs": it.get(key + ("c_nccl_rs",)), "nccl_ar": it.get(key + ("c_nccl_ar",)),
                   "cublas": _f(r["c_cublas"]), "fluxgemm_only": _f(r["c_fluxgemm"]),
                   "E_allreduce": it.get(key + ("E_allreduce",))},
            cfg=cfgs.get("rs", M, N, k), source=r["source"]))
    return rows, cfgs


def f2_alpha_sync(ws=WS, max_m=64):
    """Flux AllGather synchronisation cost at W=8 from the F2 microbenchmark (gpu mode):
    flux_default - (W / (W-1)) * (W-1 copies on one stream without sync), median over M <= max_m."""
    import statistics
    from collections import defaultdict
    vals = []
    for K in (8192, 12288):
        d = defaultdict(list)
        for r in csv.DictReader(open(os.path.join(ws, "results", "f2_ag_latency_v1", f"raw_K{K}.csv"))):
            if r["kept"] == "1" and r["mode"] == "gpu" and int(r["M"]) <= max_m:
                d[(int(r["M"]), r["item"])].append(float(r["rank_max_ms"]))
        for M in sorted({m for m, _ in d}):
            fd = statistics.median(d[(M, "flux_default")])
            ce = statistics.median(d[(M, "proto_ce_1s")])
            vals.append(fd - W / (W - 1) * ce)
    return statistics.median(vals)


if __name__ == "__main__":
    rows, cfgs = load_rows()
    print(len(rows), "rows;", sum(r["side"] == "ag" for r in rows), "AG,", sum(r["side"] == "rs" for r in rows), "RS")
    print("alpha_sync (F2, gpu, M<=64) =", round(f2_alpha_sync(), 4), "ms")
    reg = sum(1 for r in rows if r["cfg"]["source"] == "registry")
    print("points on a registry config:", reg, "of", len(rows))
