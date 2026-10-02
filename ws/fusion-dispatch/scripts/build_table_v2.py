################################################################################
# fusion-dispatch G4 (prep): build a dispatch table in the dispatcher_v1.py format from the
# PREDICTOR instead of from measured maps, plus the list of entries that need a short probe.
# The runtime (dispatcher_v1.FluxDispatcher: rank-0 broadcast, sha256 check, graph-mode skip of
# AGKernel) is reused unchanged; only where the ranking comes from changes.
#   AG layers: flux=A (AGKernel), nccl=B, fluxag=C, fluxag_fluxgemm=D;  RS layers: flux=A, nccl=B
# Probe rule (as in G1-G3): predicted margin top-1 vs top-2 < eps, or the pick runs a Flux GEMM on a
# registry config tuned for a PCIe topology (then compare with the best path without a Flux GEMM).
# Usage: python3 build_table_v2.py --model llama3-70b --tp 8 --mode gpu --out table.json [--Ms ...]
################################################################################
import argparse
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "common", "cost_model"))
from predictor import FLUX_GEMM_ARMS, FluxConfigs, HardwareProfile, predict_ag, predict_rs  # noqa: E402

# hidden, q heads, kv heads, head_dim, ffn
MODELS = {"llama3-70b": (8192, 64, 8, 128, 28672), "qwen2.5-72b": (8192, 64, 8, 128, 29568),
          "llama3-8b": (4096, 32, 8, 128, 14336)}
PATH = {"A": "flux", "B": "nccl", "C": "fluxag", "D": "fluxag_fluxgemm"}
DEFAULT_MS = [8, 16, 24, 32, 64, 72, 128, 136, 256, 264, 384, 512, 520, 1024, 1032, 2048, 3072, 4096, 6144, 8192, 16384]


def layers(model, tp):
    H, nq, nkv, hd, ffn = MODELS[model]
    return {"ag": {"qkv": ((nq + 2 * nkv) * hd // tp, H), "gate_up": (2 * ffn // tp, H)},
            "rs": {"o": (H, nq * hd // tp), "down": (H, ffn // tp)}}


def build(prof, cfgs, model, tp, Ms, eps):
    table, probes = {"ag": {}, "rs": {}}, []
    for kind, shapes in layers(model, tp).items():
        for lname, (a, b) in shapes.items():
            ent = {"layer": lname, "M": [], "rank": [], "verdict": [], "source": []}
            for M in Ms:
                if M % tp:
                    continue
                arms, _, cfg = (predict_ag if kind == "ag" else predict_rs)(prof, cfgs, M, a, b)
                rank = sorted(arms, key=arms.get)
                margin = (arms[rank[1]] - arms[rank[0]]) / arms[rank[0]]
                cand = set()
                if margin < eps:
                    cand |= {rank[0], rank[1]}
                if rank[0] in FLUX_GEMM_ARMS[kind] and cfg.get("tuned_for") == "pcie":
                    cand |= {rank[0], next(x for x in rank if x not in FLUX_GEMM_ARMS[kind])}
                ent["M"].append(M)
                ent["rank"].append([[PATH[x], round(arms[x], 4)] for x in rank])
                ent["verdict"].append("predicted")
                ent["source"].append(f"{cfg['source']}/{cfg.get('tuned_for', '')}")
                if cand:
                    probes.append({"kind": kind, "key": f"{a}x{b}", "layer": lname, "M": M,
                                   "candidates": [PATH[x] for x in sorted(cand, key=arms.get)],
                                   "reason": ("margin" if margin < eps else "") + ("+pcie" if len(cand) > 2 or
                                              (rank[0] in FLUX_GEMM_ARMS[kind] and cfg.get("tuned_for") == "pcie") else "")})
            table[kind][f"{a}x{b}"] = ent
    return table, probes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--tp", type=int, required=True)
    ap.add_argument("--mode", default="gpu", choices=["gpu", "steady"])
    ap.add_argument("--Ms", default=",".join(map(str, DEFAULT_MS)))
    ap.add_argument("--eps", type=float, default=None, help="default: profile calibration p90 residual")
    ap.add_argument("--profiles", default=os.path.join(REPO, "common", "cost_model", "hw_profiles"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    path = os.path.join(a.profiles, f"css-host-158_tp{a.tp}_{a.mode}.json")
    prof = HardwareProfile.load(path)
    eps = a.eps if a.eps is not None else prof.meta["calibration_residual_p90"]
    table, probes = build(prof, FluxConfigs(a.tp), a.model, a.tp, [int(x) for x in a.Ms.split(",")], eps)
    table["meta"] = {"built": time.strftime("%Y-%m-%d %H:%M:%S"), "builder": "ws/fusion-dispatch/scripts/build_table_v2.py",
                     "model": a.model, "world": a.tp, "mode": a.mode, "dtype": "bf16", "eps": eps,
                     "profile": os.path.relpath(path, REPO),
                     "profile_sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(),
                     "probes_pending": probes}
    json.dump(table, open(a.out, "w"), indent=1)
    n = sum(len(e["M"]) for k in ("ag", "rs") for e in table[k].values())
    print(f"wrote {a.out}: {n} entries, {len(probes)} need a probe ({len(probes) / n * 100:.0f}%)")
    for kind in ("ag", "rs"):
        for key, ent in table[kind].items():
            print(f"  {kind} {key} ({ent['layer']}): " + " ".join(f"{m}:{rk[0][0]}" for m, rk in zip(ent["M"], ent["rank"])))


if __name__ == "__main__":
    main()
