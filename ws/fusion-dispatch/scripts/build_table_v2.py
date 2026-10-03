################################################################################
# fusion-dispatch G4 (prep): build a dispatch table in the dispatcher_v1.py format from the
# PREDICTOR instead of from measured maps, plus the list of entries that need a short probe.
# The runtime (dispatcher_v1.FluxDispatcher: rank-0 broadcast, sha256 check, graph-mode skip of
# AGKernel) is reused unchanged; only where the ranking comes from changes.
#   AG layers: flux=A (AGKernel), nccl=B, fluxag=C, fluxag_fluxgemm=D;  RS layers: flux=A, nccl=B
# Probe rule (as in G1-G3): predicted margin top-1 vs top-2 < eps, or the pick runs a Flux GEMM on a
# registry config tuned for a PCIe topology (then compare with the best path without a Flux GEMM).
# Library use (G4): build(..., gemm=measured GEMMs, legacy=G3 method, graph=capturable-only margins),
# apply_probes(...); see build_g4_block_v1.py.
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
          "llama3-8b": (4096, 32, 8, 128, 14336), "qwen2.5-32b": (5120, 40, 8, 128, 27648)}
PATH = {"A": "flux", "B": "nccl", "C": "fluxag", "D": "fluxag_fluxgemm"}
DEFAULT_MS = [8, 16, 24, 32, 64, 72, 128, 136, 256, 264, 384, 512, 520, 1024, 1032, 2048, 3072, 4096, 6144, 8192, 16384]


def layers(model, tp):
    H, nq, nkv, hd, ffn = MODELS[model]
    return {"ag": {"qkv": ((nq + 2 * nkv) * hd // tp, H), "gate_up": (2 * ffn // tp, H)},
            "rs": {"o": (H, nq * hd // tp), "down": (H, ffn // tp)}}


def build(prof, cfgs, model, tp, Ms, eps, gemm=None, legacy=False, graph=False):
    """gemm: {(layer_name, M): {item: ms}} measured standalone GEMMs (G4); legacy: G3 method (joint GemmRS
    model, no measured GEMMs); graph: probe margins among CUDA-graph-capturable paths only (the runtime
    skips AGKernel in graph mode). Returns (table, probes, arms) with arms[(kind, layer, M)] = {arm: ms}."""
    table, probes, all_arms = {"ag": {}, "rs": {}}, [], {}
    for kind, shapes in layers(model, tp).items():
        for lname, (a, b) in shapes.items():
            ent = {"layer": lname, "M": [], "rank": [], "verdict": [], "source": []}
            for M in Ms:
                if M % tp:
                    continue
                g = (gemm or {}).get((lname, M), {})
                if kind == "ag":
                    meas = {"cublas": g.get("c_cublas"), "fluxgemm": g.get("c_fluxgemm")} if g and not legacy else None
                    arms, _, cfg = predict_ag(prof, cfgs, M, a, b, meas=meas)
                else:
                    meas = {"cublas": g.get("c_cublas"), "fluxgemm_only": g.get("c_fluxgemm_only")} if g and not legacy else None
                    arms, _, cfg = predict_rs(prof, cfgs, M, a, b, meas=meas, legacy=legacy)
                all_arms[(kind, lname, M)] = arms
                rank = sorted(arms, key=arms.get)
                cap = [x for x in rank if not (graph and kind == "ag" and x == "A")]
                margin = (arms[cap[1]] - arms[cap[0]]) / arms[cap[0]]
                cand = set()
                if margin < eps:
                    cand |= {cap[0], cap[1]}
                if cap[0] in FLUX_GEMM_ARMS[kind] and cfg.get("tuned_for") == "pcie":
                    cand |= {cap[0], next(x for x in cap if x not in FLUX_GEMM_ARMS[kind])}
                ent["M"].append(M)
                ent["rank"].append([[PATH[x], round(arms[x], 4)] for x in rank])
                ent["verdict"].append("predicted")
                ent["source"].append(f"{cfg['source']}/{cfg.get('tuned_for', '')}")
                if cand:
                    probes.append({"kind": kind, "key": f"{a}x{b}", "layer": lname, "M": M, "margin": round(margin, 4),
                                   "candidates": [x for x in sorted(cand, key=arms.get)]})
            table[kind][f"{a}x{b}"] = ent
    return table, probes, all_arms


def apply_probes(table, probes, measured):
    """measured: {(kind, layer, M): {arm: ms}} from short path probes; the fastest probed candidate moves to
    the front of that entry's ranking. Returns the number of entries whose first choice changed."""
    changed = 0
    for p in probes:
        got = measured.get((p["kind"], p["layer"], p["M"]))
        if not got or any(c not in got for c in p["candidates"]):
            continue
        best = PATH[min(p["candidates"], key=lambda c: got[c])]
        ent = table[p["kind"]][p["key"]]
        i = ent["M"].index(p["M"])
        rk = ent["rank"][i]
        if rk[0][0] != best:
            changed += 1
        ent["rank"][i] = [x for x in rk if x[0] == best] + [x for x in rk if x[0] != best]
        ent["verdict"][i] = "probed"
    return changed


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
    table, probes, _ = build(prof, FluxConfigs(a.tp), a.model, a.tp, [int(x) for x in a.Ms.split(",")], eps)
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
