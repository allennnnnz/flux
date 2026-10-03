################################################################################
# fusion-dispatch G4-block steps 6 and 8: block runs with validate_block_v4.py (vLLM venv), each under
# exclusive_guard, L = 4 blocks, decode in CUDA-graph mode (ctx 1024), prefill eager.
#   python3 run_g4_block_v1.py layout   layout probes: policies tp_ar_vllm, sp_g4 at the flagged M,
#                                       30 rounds -> results/g4_block_layout_probes
#   python3 run_g4_block_v1.py oracle   every policy, 200 rounds -> results/g4_block_oracle
################################################################################
import csv
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(WS))
T = os.path.join(WS, "results", "g4_block_tables")
CONFIGS = [(8, "qwen2.5-32b"), (4, "qwen2.5-32b"), (4, "llama3-8b")]
PHASES = {"decode": ("graph", [32, 128, 256, 384, 512]), "prefill": ("eager", [1024, 2048, 4096])}
step = sys.argv[1]
only_phase = sys.argv[2] if len(sys.argv) > 2 else None  # rerun one phase (e.g. after a failure)
out = os.path.join(WS, "results", "g4_block_layout_probes" if step == "layout" else "g4_block_oracle")
os.makedirs(out, exist_ok=True)
flag = set()
if step == "layout":
    for r in csv.DictReader(open(os.path.join(T, "layout_pred.csv"))):
        if r["layout_probe"] == "1":
            flag.add((int(r["tp"]), r["model"], r["phase"], int(r["M"])))
log = open(os.path.join(out, "run_log.txt"), "a")
log.write(f"[G4-block-{step}] start {time.strftime('%F %T')}\n")
t_all = time.time()
for tp, model in CONFIGS:
    for phase, (mode, Ms) in PHASES.items():
        if only_phase and phase != only_phase:
            continue
        ms = [m for m in Ms if step == "oracle" or (tp, model, phase, m) in flag]
        if not ms:
            continue
        name = f"{model}_tp{tp}_{phase}"
        args = ["--model", model, "--phase", phase, "--mode", mode, "--Ms", ",".join(map(str, ms)), "--L", "4",
                "--ctx", "1024", "--table_g4", os.path.join(T, f"table_g4_{name}.json"),
                "--table_g3", os.path.join(T, f"table_g3_{name}.json"), "--out_dir", out, "--warmup", "10",
                "--rounds", "30" if step == "layout" else "200", "--seed", "20261004" if step == "oracle" else "20261003"]
        if step == "layout":
            args += ["--policies", "tp_ar_vllm,sp_g4", "--tag", "layoutprobe"]
        cmd = ["python3", "common/measure/exclusive_guard.py", "--log", os.path.join(out, f"guard_{name}.log"), "--",
               "timeout", "3600", "env", f"TP={tp}", "bash", "ws/fusion-dispatch/scripts/launch_vllm_env.sh",
               "ws/fusion-dispatch/scripts/validate_block_v4.py"] + args
        t0 = time.time()
        rc = subprocess.run(cmd, cwd=REPO, stdout=open(os.path.join(out, f"log_{name}.txt"), "w"),
                            stderr=subprocess.STDOUT).returncode
        verdict = open(os.path.join(out, f"guard_{name}.log")).read().strip().splitlines()[-1]
        log.write(f"[G4-block-{step}] {name} M={ms} exit {rc} {time.time() - t0:.0f}s {verdict}\n")
        log.flush()
log.write(f"[G4-block-{step}] done {time.strftime('%F %T')}  total {time.time() - t_all:.0f}s\n")
log.close()
open(os.path.join(out, "DONE"), "w").close()
