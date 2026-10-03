################################################################################
# fusion-dispatch G4-block (D-009): pre-registered dispatch tables and LAYOUT decisions for the block
# validation (validate_block_v4.py), on the fresh test set (Qwen2.5-32B TP8 / TP4, Llama-3-8B TP4).
#   decode  : vLLM CUDA-graph buckets M = 32, 128, 256, 384, 512  -> gpu-mode profiles, graph-capturable
#   prefill : eager, M = 1024, 2048, 4096                          -> steady-mode profiles
# Steps (each output committed before the next measurement):
#   tables    sp_g4 table (model comm + overlap, measured GEMMs from results/g4_block_gemm, probe list)
#             and sp_g3 table (G3 method: model only, G2/G3 profiles) per (config, phase)
#   finalize  apply the path probes (results/g4_block_probes, 30 rounds) to the sp_g4 tables, then
#             predict the layout per M with predictor/block.py: vLLM default (tp_ar_vllm) vs sp_g4;
#             layout probe list where |delta| < eps x t_sp
#   layout    apply the layout probes (validate_block_v4 --policies tp_ar_vllm,sp_g4, 30 rounds)
#             -> layout_g4.csv (the final pre-registered layout choice "auto_g4")
# Usage: python3 build_g4_block_v1.py tables | finalize | layout
################################################################################
import csv
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(WS))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "common", "cost_model"))
from build_table_v2 import PATH, apply_probes, build, layers  # noqa: E402
from predictor import FluxConfigs, HardwareProfile  # noqa: E402
from predictor.block import BlockProfile, layout_times  # noqa: E402

CONFIGS = [(8, "qwen2.5-32b"), (4, "qwen2.5-32b"), (4, "llama3-8b")]
PHASES = {"decode": ("gpu", True, [32, 128, 256, 384, 512]), "prefill": ("steady", False, [1024, 2048, 4096])}
HIDDEN = {"qwen2.5-32b": 5120, "llama3-8b": 4096}
HARNESS = {("qwen2.5-32b", "qkv"): "Q32-QKV", ("qwen2.5-32b", "gate_up"): "Q32-GU", ("qwen2.5-32b", "o"): "Q32-O",
           ("qwen2.5-32b", "down"): "Q32-down", ("llama3-8b", "qkv"): "L8-QKV", ("llama3-8b", "gate_up"): "L8-GU",
           ("llama3-8b", "o"): "L8-O", ("llama3-8b", "down"): "L8-down"}
ITEM = {"A": "A_fused", "B": "B_nccl_cublas", "C": "C_fluxag_cublas", "D": "D_fluxag_fluxgemm"}
OUT = os.path.join(WS, "results", "g4_block_tables")
PROF = os.path.join(REPO, "common", "cost_model", "hw_profiles")
INV = {v: k for k, v in PATH.items()}


def gemm(tp, model):
    g = {}
    for r in csv.DictReader(open(os.path.join(WS, "results", "g4_block_gemm", f"gemm_{model}_tp{tp}.csv"))):
        g.setdefault(r["mode"], {}).setdefault((r["layer"], int(r["M"])), {})[r["item"]] = float(r["median_ms"])
    return g


def tag(tp, model, phase):
    return f"{model}_tp{tp}_{phase}"


def tables():
    os.makedirs(OUT, exist_ok=True)
    probes_all = []
    for tp, model in CONFIGS:
        cfgs, G = FluxConfigs(tp), gemm(tp, model)
        for phase, (mode, graph, Ms) in PHASES.items():
            new = HardwareProfile.load(os.path.join(PROF, f"css-host-158_tp{tp}_{mode}_g4.json"))
            old = HardwareProfile.load(os.path.join(PROF, f"css-host-158_tp{tp}_{mode}.json"))
            eps = max(0.03, new.meta["calibration_residual_meas_p90"])
            t4, p4, a4 = build(new, cfgs, model, tp, Ms, eps, gemm=G[mode], graph=graph)
            t3, _, _ = build(old, cfgs, model, tp, Ms, 1.0, legacy=True, graph=graph)
            for t, name in ((t4, "g4"), (t3, "g3")):
                t["meta"] = {"model": model, "world": tp, "phase": phase, "mode": mode, "graph": graph,
                             "method": name, "eps": eps if name == "g4" else None}
            t4["meta"]["probes_pending"] = p4
            json.dump(t4, open(os.path.join(OUT, f"table_g4_{tag(tp, model, phase)}.json"), "w"), indent=1)
            json.dump(t3, open(os.path.join(OUT, f"table_g3_{tag(tp, model, phase)}.json"), "w"), indent=1)
            json.dump({f"{k[0]}|{k[1]}|{k[2]}": v for k, v in a4.items()},
                      open(os.path.join(OUT, f"arms_g4_{tag(tp, model, phase)}.json"), "w"), indent=1)
            for p in p4:
                probes_all.append([tp, model, phase, mode, p["kind"], HARNESS[(model, p["layer"])], p["layer"], p["M"],
                                   "|".join(p["candidates"])])
    with open(os.path.join(OUT, "probes_block.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tp", "model", "phase", "mode", "side", "layer", "lname", "M", "candidates"])
        w.writerows(probes_all)
    n = sum(len(PHASES[p][2]) * 4 for p in PHASES) * len(CONFIGS)
    print(f"tables written to {os.path.relpath(OUT, REPO)}; path probes {len(probes_all)} of {n} entries "
          f"({len(probes_all) / n * 100:.0f}%)")


def probe_medians():
    med = {}
    d0 = os.path.join(WS, "results", "g4_block_probes")
    for d in sorted(os.listdir(d0)) if os.path.isdir(d0) else []:
        if not d.startswith("tp"):
            continue
        path = os.path.join(d0, d)
        subprocess.run([sys.executable, os.path.join(HERE, "analyze_v1.py"), path, "--out", os.path.join(path, "summary")],
                       check=True, capture_output=True)
        for r in csv.DictReader(open(os.path.join(path, "summary_items.csv"))):
            med[(int(d[2:]), r["layer"], int(r["M"]), r["mode"], r["item"])] = float(r["median_ms"])
    return med


def finalize():
    med = probe_medians()
    rows = []
    for tp, model in CONFIGS:
        G = gemm(tp, model)
        H = HIDDEN[model]
        for phase, (mode, graph, Ms) in PHASES.items():
            path = os.path.join(OUT, f"table_g4_{tag(tp, model, phase)}.json")
            t4 = json.load(open(path))
            measured = {}
            for p in t4["meta"]["probes_pending"]:
                hl = HARNESS[(model, p["layer"])]
                got = {c: med.get((tp, hl, p["M"], mode, ITEM[c])) for c in p["candidates"]}
                if all(v is not None for v in got.values()):
                    measured[(p["kind"], p["layer"], p["M"])] = got
            changed = apply_probes(t4, t4["meta"]["probes_pending"], measured)
            t4["meta"]["probes_applied"] = len(measured)
            t4["meta"]["probes_changed_first_choice"] = changed
            json.dump(t4, open(path, "w"), indent=1)
            arms = {tuple(k.split("|")[:2]) + (int(k.split("|")[2]),): v
                    for k, v in json.load(open(os.path.join(OUT, f"arms_g4_{tag(tp, model, phase)}.json"))).items()}
            bp = BlockProfile.load(os.path.join(PROF, f"css-host-158_tp{tp}_{mode}_block.json"))
            eps = t4["meta"]["eps"]
            lay = layers(model, tp)
            for M in Ms:
                cub, sp = [], []
                for kind in ("ag", "rs"):
                    for lname, (a, b) in lay[kind].items():
                        cub.append(G[mode][(lname, M)]["c_cublas"])
                        ent = t4[kind][f"{a}x{b}"]
                        rk = ent["rank"][ent["M"].index(M)]
                        first = next(pth for pth, _ in rk if not (graph and kind == "ag" and pth == "flux"))
                        probed = measured.get((kind, lname, M))
                        x = INV[first]
                        sp.append(probed[x] if probed and x in probed else arms[(kind, lname, M)][x])
                t_tp, t_sp, delta = layout_times(bp, M, H, cub, sp)
                choice = "sp_g4" if delta > 0 else "tp_ar_vllm"
                probe = abs(delta) < eps * t_sp
                rows.append([tp, model, phase, M, f"{t_tp:.4f}", f"{t_sp:.4f}", f"{delta:.4f}", choice, int(probe)])
    with open(os.path.join(OUT, "layout_pred.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tp", "model", "phase", "M", "t_tp_ar_modelled", "t_sp_modelled", "delta", "choice", "layout_probe"])
        w.writerows(rows)
    print(f"tables finalised; layout_pred.csv: {len(rows)} (config, phase, M); layout probes {sum(r[-1] for r in rows)}")
    for r in rows:
        print("  " + " ".join(str(x) for x in r))


def layout():
    rows = list(csv.DictReader(open(os.path.join(OUT, "layout_pred.csv"))))
    meds = {}
    d = os.path.join(WS, "results", "g4_block_layout_probes")
    import glob
    import statistics
    for f in glob.glob(os.path.join(d, "raw_*.csv")):
        model_tp = os.path.basename(f)[4:].split("_decode")[0].split("_prefill")[0]
        v = {}
        for r in csv.DictReader(open(f)):
            if r["kept"] == "1":
                v.setdefault((r["phase"], int(r["M"]), r["policy"]), []).append(float(r["rank_max_ms"]))
        for k, x in v.items():
            meds[(model_tp,) + k] = statistics.median(x)
    out = []
    for r in rows:
        key = f"{r['model']}_tp{r['tp']}"
        final = r["choice"]
        note = ""
        if r["layout_probe"] == "1":
            a = meds.get((key, r["phase"], int(r["M"]), "tp_ar_vllm"))
            b = meds.get((key, r["phase"], int(r["M"]), "sp_g4"))
            if a is not None and b is not None:
                final = "tp_ar_vllm" if a <= b else "sp_g4"
                note = f"probe tp_ar_vllm={a:.4f} sp_g4={b:.4f}"
            else:
                note = "probe MISSING"
        out.append([r["tp"], r["model"], r["phase"], r["M"], r["choice"], final, note])
    with open(os.path.join(OUT, "layout_g4.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tp", "model", "phase", "M", "choice_model", "choice_final", "note"])
        w.writerows(out)
    print(f"wrote layout_g4.csv ({len(out)} rows)")


if __name__ == "__main__":
    {"tables": tables, "finalize": finalize, "layout": layout}[sys.argv[1]]()
