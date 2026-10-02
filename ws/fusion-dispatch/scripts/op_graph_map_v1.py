################################################################################
# fusion-dispatch V5: per-op dispatch map measured as CUDA-graph REPLAY (what vLLM decode
# runs), to check whether the gpu-mode table used for graph deployment picks the same path.
# F0.4 found graph replay adds 7-15 us to NCCL paths but only 3-4 us to Flux GemmRS, which
# could shift near-tie decisions.
#   AG layers (dispatch_map_v2.Case):   B_nccl_cublas, C_fluxag_cublas, D_fluxag_fluxgemm
#                                       (A_fused = AGKernel cannot be captured, F0.4)
#   RS layers (dispatch_map_rs_v1.CaseRS): A_fused (GemmRS), B_nccl_cublas
# Per (layer, M): capture every item once (3 warmups on a side stream first), then >=200 rounds;
# each round replays every item once in a shared random order with L2 flush + GPU sleep pad +
# 1-elem NCCL all_reduce alignment before it; per-round rank-max; SM-clock filter.
# Output: raw CSV in the dispatch_map format (mode="graph") + a comparison with a table JSON.
# Usage: ./launch.sh op_graph_map_v1.py --layers L-QKV,L-GU,L-O,L-down --Ms 8,...,512 --table t.json --out_dir d
################################################################################
import argparse
import csv
import json
import os
import random
import statistics
import sys
import time

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "common", "measure"))
import dispatch_map_v2 as dm  # noqa: E402
import dispatch_map_rs_v1 as rs  # noqa: E402
from clock_logger import ClockLogger  # noqa: E402
from flux.testing import initialize_distributed  # noqa: E402

AG_ITEMS = ["B_nccl_cublas", "C_fluxag_cublas", "D_fluxag_fluxgemm"]
RS_ITEMS = ["A_fused", "B_nccl_cublas"]
PATH = {"ag": {"B_nccl_cublas": "nccl", "C_fluxag_cublas": "fluxag", "D_fluxag_fluxgemm": "fluxag_fluxgemm"},
        "rs": {"A_fused": "flux", "B_nccl_cublas": "nccl"}}


def capture(fn, w):
    cur = torch.cuda.current_stream()
    s = torch.cuda.Stream()
    s.wait_stream(cur)
    with torch.cuda.stream(s):
        for _ in range(3):
            fn(w)
    cur.wait_stream(s)
    torch.cuda.synchronize()
    dist.barrier(TP)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn(w)
    torch.cuda.synchronize()
    return g


def main():
    Ms = [int(x) for x in A.Ms.split(",")]
    table = json.load(open(A.table))
    rng = random.Random(A.seed)
    FLUSH = torch.empty(128 * 1024 * 1024 // 4, device="cuda")
    ALIGN = torch.zeros(1, device="cuda")
    os.makedirs(A.out_dir, exist_ok=True) if RANK == 0 else None
    rows, cmp_rows = [], []
    for L in A.layers.split(","):
        kind = "ag" if L in dm.LAYERS else "rs"
        if kind == "ag":
            layer = dm.Layer(L, max(Ms), 1)
            N, K = dm.LAYERS[L]
            key = f"{N // W}x{K}"
            items = AG_ITEMS
        else:
            layer = rs.LayerRS(L, max(Ms), 1)
            N, K = rs.LAYERS[L]
            key = f"{N}x{K // W}"
            items = RS_ITEMS
        w = layer.weights[0]
        for M in Ms:
            case = dm.Case(layer, M) if kind == "ag" else rs.CaseRS(layer, M)
            graphs = {it: capture(case.fns[it], w) for it in items}
            ev = [{it: (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for it in items}
                  for _ in range(A.rounds)]
            orders, windows = [], []
            for _ in range(A.warmup):
                for it in items:
                    graphs[it].replay()
            torch.cuda.synchronize()
            with ClockLogger(LOCAL_RANK, 0.02) as clk:
                for r in range(A.rounds):
                    order = items[:]
                    rng.shuffle(order)
                    orders.append(order)
                    t0 = time.perf_counter()
                    for it in order:
                        FLUSH.zero_()
                        torch.cuda._sleep(400_000)
                        dist.all_reduce(ALIGN, group=TP)
                        ev[r][it][0].record()
                        graphs[it].replay()
                        ev[r][it][1].record()
                    torch.cuda.synchronize()
                    windows.append((t0, time.perf_counter()))
                clocks = [clk.min_clock_between(*x) for x in windows]
            t = torch.tensor([[ev[r][it][0].elapsed_time(ev[r][it][1]) for it in items] for r in range(A.rounds)],
                             device="cuda", dtype=torch.float64)
            ck = torch.tensor([c if c else -1 for c in clocks], device="cuda", dtype=torch.float64)
            gt, gk = [torch.zeros_like(t) for _ in range(W)], [torch.zeros_like(ck) for _ in range(W)]
            dist.all_gather(gt, t, group=TP)
            dist.all_gather(gk, ck, group=TP)
            del graphs
            if RANK == 0:
                T, Kc = torch.stack(gt).cpu().max(0).values, torch.stack(gk).cpu()
                kept = [True] * A.rounds
                for rk in range(W):
                    vals = [int(v) for v in Kc[rk].tolist() if v > 0]
                    if vals:
                        mo = statistics.mode(vals)
                        kept = [k and not (0 < Kc[rk, r].item() < 0.95 * mo) for r, k in enumerate(kept)]
                ks = [r for r in range(A.rounds) if kept[r]]
                med = {it: statistics.median(T[r, j].item() for r in ks) for j, it in enumerate(items)}
                for r in range(A.rounds):
                    for j, it in enumerate(items):
                        rows.append([L, N, K, M, "graph", r, it, orders[r].index(it), f"{T[r, j].item():.5f}",
                                     int(kept[r])])
                best = min(med, key=med.get)
                # per-round difference best vs runner-up (tie if p10..p90 contains 0)
                sec = sorted(med, key=med.get)[1]
                jb, js = items.index(best), items.index(sec)
                d = sorted(T[r, jb].item() - T[r, js].item() for r in ks)
                tie = d[len(d) // 10] <= 0 <= d[9 * len(d) // 10]
                ent = table[kind].get(key)
                tchoice = None
                if ent and M in ent["M"]:
                    ranked = [p for p, _ in ent["rank"][ent["M"].index(M)]]
                    tchoice = next(p for p in ranked if not (kind == "ag" and p == "flux"))
                gchoice = PATH[kind][best]
                cmp_rows.append([L, M, gchoice, tchoice, "tie" if tie else "decided",
                                 " ".join(f"{PATH[kind][i]}={med[i]:.4f}" for i in items), len(ks)])
                print(f"[{L} M={M} graph] kept {len(ks)}  " + "  ".join(f"{i}={v:.4f}" for i, v in med.items())
                      + f"  graph-best={gchoice} ({'tie' if tie else 'decided'})  table={tchoice}",
                      file=sys.stderr, flush=True)
            del case
            torch.cuda.empty_cache()
    if RANK == 0:
        with open(os.path.join(A.out_dir, "raw_graph.csv"), "w", newline="") as f:
            w_ = csv.writer(f)
            w_.writerow(["layer", "N", "K", "M", "mode", "round", "item", "order_pos", "rank_max_ms", "kept"])
            w_.writerows(rows)
        with open(os.path.join(A.out_dir, "compare_table.csv"), "w", newline="") as f:
            w_ = csv.writer(f)
            w_.writerow(["layer", "M", "graph_best", "table_choice", "graph_margin", "graph_medians_ms", "kept"])
            w_.writerows(cmp_rows)
        agree = sum(1 for r in cmp_rows if r[2] == r[3])
        agree_or_tie = sum(1 for r in cmp_rows if r[2] == r[3] or r[4] == "tie")
        print(f"\ngraph-best == table choice: {agree}/{len(cmp_rows)}; agree or graph tie: {agree_or_tie}/{len(cmp_rows)}",
              file=sys.stderr, flush=True)
    dist.barrier(TP)
    torch.cuda.synchronize()
    os._exit(0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="L-QKV,L-GU,L-O,L-down")
    ap.add_argument("--Ms", default="8,16,32,64,128,256,384,512")
    ap.add_argument("--table", required=True)
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20261002)
    ap.add_argument("--out_dir", required=True)
    A = ap.parse_args()
    TP = initialize_distributed()
    os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    torch.use_deterministic_algorithms(False)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    RANK, W = TP.rank(), TP.size()
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    ns = argparse.Namespace(atol=0.05, rtol=0.05, steady_L=1, pad_cycles=400_000, clock_period=0.02)
    dm.TP_GROUP, dm.RANK, dm.W, dm.NNODES, dm.ARGS, dm.LOCAL_RANK = TP, RANK, W, 1, ns, LOCAL_RANK
    rs.TP, rs.RANK, rs.W, rs.ARGS = TP, RANK, W, argparse.Namespace(atol=0.1, rtol=0.05)
    main()
