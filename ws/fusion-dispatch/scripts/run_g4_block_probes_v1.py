################################################################################
# fusion-dispatch G4-block step 4: path probes for the sp_g4 table entries listed in
# results/g4_block_tables/probes_block.csv (30 rounds, same protocol as the maps), one process per
# (TP, layer, mode), each under exclusive_guard. Usage: python3 run_g4_block_probes_v1.py
################################################################################
import csv
import os
import subprocess
import time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(WS))
ITEM = {"A": "A_fused", "B": "B_nccl_cublas", "C": "C_fluxag_cublas", "D": "D_fluxag_fluxgemm"}
out = os.path.join(WS, "results", "g4_block_probes")
os.makedirs(out, exist_ok=True)
groups = defaultdict(lambda: [set(), set()])
for r in csv.DictReader(open(os.path.join(WS, "results", "g4_block_tables", "probes_block.csv"))):
    g = groups[(int(r["tp"]), r["layer"], r["side"], r["mode"])]
    g[0].add(int(r["M"]))
    g[1] |= set(r["candidates"].split("|"))
log = open(os.path.join(out, "run_log.txt"), "a")
log.write(f"[G4-block-probe] start {time.strftime('%F %T')}  {len(groups)} runs\n")
t_all = time.time()
for (tp, layer, side, mode), (ms, cand) in sorted(groups.items()):
    d = os.path.join(out, f"tp{tp}")
    os.makedirs(d, exist_ok=True)
    launch = ["./launch.sh"] if tp == 8 else ["ws/fusion-dispatch/scripts/launch_tp.sh", str(tp)]
    script = "dispatch_map_v2.py" if side == "ag" else "dispatch_map_rs_v1.py"
    cmd = ["python3", "common/measure/exclusive_guard.py", "--log", os.path.join(d, f"guard_{layer}_{mode}.log"), "--",
           "timeout", "1800", "pixi", "run", "--manifest-path", "pixi.toml"] + launch + \
          [os.path.join("ws/fusion-dispatch/scripts", script), "--layer", layer, "--Ms", ",".join(map(str, sorted(ms))),
           "--max_m", str(max(ms)), "--modes", mode, "--items", ",".join(ITEM[c] for c in sorted(cand)),
           "--rounds", "30", "--warmup", "30", "--clock_period", "0.02", "--out_dir", d]
    t0 = time.time()
    rc = subprocess.run(cmd, cwd=REPO, stdout=open(os.path.join(d, f"log_{layer}_{mode}.txt"), "w"),
                        stderr=subprocess.STDOUT).returncode
    verdict = open(os.path.join(d, f"guard_{layer}_{mode}.log")).read().strip().splitlines()[-1]
    log.write(f"[G4-block-probe] tp{tp} {layer} {mode} M={sorted(ms)} exit {rc} {time.time() - t0:.0f}s {verdict}\n")
    log.flush()
log.write(f"[G4-block-probe] done {time.strftime('%F %T')}  total {time.time() - t_all:.0f}s\n")
log.close()
open(os.path.join(out, "DONE"), "w").close()
