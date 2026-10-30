"""FS1d (2026-10-30; plan addendum 3): does a running cuBLAS GEMM slow copy-engine peer copies on 8x A100 NVSwitch,
with no Flux involved?

One process drives all 8 GPUs. Per case (TP8 all-gather + GEMM shape of a layer at token count M):
  copies : each GPU j pushes its shard A_j [M/8, K] to the 7 other GPUs (dst.copy_(src) runs on the SOURCE device's
           current stream, CLAUDE.md trap) on a side stream of device j, rotated order j+1, j+2, ...
  gemm   : each GPU runs torch.mm(full [M, K], W [N/8, K]^T) on its main stream
  both   : copies and GEMM enqueued together (copies first), no dependency between them
Per device: copy duration (first copy start -> last copy end, side stream events) and GEMM duration (main stream);
report the max over devices, median over --rounds rounds (modes interleaved in shuffled order).
copy slowdown = copies_in_both / copies_alone; gemm slowdown = gemm_in_both / gemm_alone.
"""
import argparse
import json
import os
import random
import statistics

import torch

DT = torch.bfloat16
CASES = {"L-QKV": (10240, 8192), "G-FC1": (49152, 12288), "Q-GU": (55296, 5120), "Q-QKV": (7168, 5120), "L-GU": (57344, 8192)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cases", default="L-QKV:1024,L-QKV:2048,L-QKV:8192,Q-QKV:2048,Q-GU:1024,L-GU:1024,G-FC1:1024,G-FC1:8192")
    ap.add_argument("--rounds", type=int, default=20)
    args = ap.parse_args()
    G = torch.cuda.device_count()
    for i in range(G):
        for j in range(G):
            if i != j:
                assert torch.cuda.can_device_access_peer(i, j)
    side = [torch.cuda.Stream(device=d) for d in range(G)]
    main_s = [torch.cuda.current_stream(d) for d in range(G)]
    res = []
    for case in args.cases.split(","):
        name, M = case.split(":")
        M = int(M)
        N, K = CASES[name]
        n, m = N // G, M // G
        A = [torch.randn((m, K), device=d).to(DT) for d in range(G)]
        full = [torch.empty((M, K), dtype=DT, device=d) for d in range(G)]          # copy destinations
        fref = [torch.randn((M, K), device=d).to(DT) for d in range(G)]            # GEMM input (no race)
        W = [(torch.randn((n, K), device=d) * 0.01).to(DT) for d in range(G)]
        out = [torch.empty((M, n), dtype=DT, device=d) for d in range(G)]

        def run(mode):
            for d in range(G):
                torch.cuda.synchronize(d)
            ev = {}
            for j in range(G):
                with torch.cuda.device(j):
                    s0 = torch.cuda.Event(enable_timing=True)
                    s0.record(main_s[j])
                    ev[j] = {"s0": s0}
                    if mode in ("copies", "both"):
                        side[j].wait_event(s0)
                        cs, ce = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        cs.record(side[j])
                        with torch.cuda.stream(side[j]):
                            for k in range(1, G):
                                i = (j + k) % G
                                full[i][j * m:(j + 1) * m].copy_(A[j], non_blocking=True)
                        ce.record(side[j])
                        ev[j]["c"] = (cs, ce)
            for j in range(G):
                with torch.cuda.device(j):
                    if mode in ("gemm", "both"):
                        gs, ge = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        gs.record(main_s[j])
                        torch.mm(fref[j], W[j].t(), out=out[j])
                        ge.record(main_s[j])
                        ev[j]["g"] = (gs, ge)
            for d in range(G):
                torch.cuda.synchronize(d)
            c = max(e["c"][0].elapsed_time(e["c"][1]) for e in ev.values()) if "c" in ev[0] else None
            g = max(e["g"][0].elapsed_time(e["g"][1]) for e in ev.values()) if "g" in ev[0] else None
            return c, g

        for mode in ("copies", "gemm", "both"):
            for _ in range(3):
                run(mode)
        meas = {"copies": [], "gemm": [], "both_c": [], "both_g": []}
        for r in range(args.rounds):
            order = ["copies", "gemm", "both"]
            random.Random(r).shuffle(order)
            for mode in order:
                c, g = run(mode)
                if mode == "copies":
                    meas["copies"].append(c)
                elif mode == "gemm":
                    meas["gemm"].append(g)
                else:
                    meas["both_c"].append(c)
                    meas["both_g"].append(g)
        med = {k: statistics.median(v) for k, v in meas.items()}
        rec = {"case": name, "M": M, **med, "copy_slowdown": med["both_c"] / med["copies"],
               "gemm_slowdown": med["both_g"] / med["gemm"],
               "shard_MB": m * K * 2 / 1e6, "copy_GBps_alone_per_gpu": 7 * m * K * 2 / (med["copies"] * 1e6)}
        res.append(rec)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
        del A, full, fref, W, out
        torch.cuda.empty_cache()
    os.makedirs(args.out, exist_ok=True)
    json.dump(res, open(os.path.join(args.out, "fs1d_raw.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
