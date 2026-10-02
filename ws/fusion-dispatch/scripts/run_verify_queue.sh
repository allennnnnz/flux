#!/bin/bash
# Verification queue V4, V5, V9, V10, V8, V7 (reports/20260930_verification.md). Runs after the
# V3 block run (run_v3_block.sh) has finished. Sequential: every item needs all 8 GPUs.
# Run from repo root: bash ws/fusion-dispatch/scripts/run_verify_queue.sh <v3 driver output file>
set -u
R=ws/fusion-dispatch/results
S=ws/fusion-dispatch/scripts
V3OUT=${1:-}
if [ -n "$V3OUT" ]; then until grep -q "eval prefill eager exit" "$V3OUT"; do sleep 20; done; fi
echo "[queue] start $(date +%T)"

# ---- V4: 80 blocks, layout calibrated on the 4-block V3 run (cal copied), evaluated on 80 blocks
mkdir -p $R/v4_block_L80/eval && cp -r $R/v3_block_vllm_ar/cal $R/v4_block_L80/ && cp $R/v3_block_vllm_ar/table_*.json $R/v4_block_L80/
for pm in "decode graph 64,256,384,512 gpu" "prefill eager 1024,4096 steady"; do
  set -- $pm
  timeout 3600 bash $S/launch_vllm_env.sh $S/validate_block_v3.py --phase $1 --mode $2 --Ms $3 \
    --table $R/v4_block_L80/table_$4.json --L 80 --rounds 200 --warmup 5 --seed 20261001 \
    --out_dir $R/v4_block_L80/eval > $R/v4_block_L80/eval/log_$1_$2.txt 2>&1
  echo "[V4] $1 $2 exit $?"; grep -E "^\[" $R/v4_block_L80/eval/log_$1_$2.txt | sed 's/ dispatch counts.*//'
done

# ---- V5: op-level map as CUDA-graph replay vs the gpu-mode table
timeout 3600 pixi run --manifest-path pixi.toml ./launch.sh $S/op_graph_map_v1.py --table $R/f1_block_v2/table_gpu.json \
  --out_dir $R/v5_op_graph > $R/v5_op_graph_log.txt 2>&1
echo "[V5] exit $?"; grep -E "^\[|graph-best ==" $R/v5_op_graph_log.txt

# ---- V9: Triton AllGather prototype under rank skew
mkdir -p $R/v9_triton_stress
for spec in "64 200" "64 2000" "1024 200"; do
  set -- $spec
  timeout 1800 pixi run --manifest-path pixi.toml ./launch.sh $S/triton_ag_stress_v1.py --iters 5000 --M $1 --K 8192 \
    --max_skew_us $2 --out $R/v9_triton_stress/stress_M$1_skew$2.json > $R/v9_triton_stress/log_M$1_skew$2.txt 2>&1
  echo "[V9] M=$1 skew=$2 exit $?"; grep -E "^\[(single|double)\]" $R/v9_triton_stress/log_M$1_skew$2.txt
done

# ---- V10: tuning upside (profiling, no rebuild) on the four Llama layers
mkdir -p $R/v10_tune
for L in L-QKV L-GU L-O L-down; do
  timeout 3600 pixi run --manifest-path pixi.toml ./launch.sh $S/tune_upside_v2.py --layer $L \
    --Ms 8,64,256,512,1024,2048,4096,8192 --out $R/v10_tune/tune_$L.json > $R/v10_tune/log_$L.txt 2>&1
  echo "[V10] $L exit $?"; grep -E "^\[$L" $R/v10_tune/log_$L.txt
done

# ---- V8: nsys timelines for the remaining AG layers
bash $S/run_v8_nsys.sh
echo "[V8] done"

# ---- V7: reruns of points with < 200 kept rounds (list: results/v7_rerun_list.json)
python3 - <<'PY' > $R/v7_rerun_cmds.txt
import json
from collections import defaultdict
g = defaultdict(list)
for h, L, M, mode, kept, rounds, r in json.load(open("ws/fusion-dispatch/results/v7_rerun_list.json")):
    wu = 100 if M == 8 else 30
    g[(h, L, mode, r, wu)].append(M)
for (h, L, mode, r, wu), Ms in sorted(g.items()):
    script = "dispatch_map_v2.py" if h == "ag" else "dispatch_map_rs_v1.py"
    extra = "--max_m 16384" if h == "ag" else "--max_m 16384"
    print(f"{h} {L} {mode} {r} {wu} {script} {','.join(map(str, sorted(Ms)))} {extra}")
PY
while read h L mode r wu script Ms extra mm; do
  mkdir -p $R/v7_rerun/$h
  timeout 3600 pixi run --manifest-path pixi.toml ./launch.sh $S/$script --layer $L --Ms $Ms $extra $mm --modes $mode \
    --rounds $r --warmup $wu --clock_period 0.02 --out_dir $R/v7_rerun/$h > $R/v7_rerun/$h/log_${L}_${mode}_$r.txt 2>&1
  echo "[V7] $h $L $mode rounds=$r Ms=$Ms exit $?"
done < $R/v7_rerun_cmds.txt
echo "[queue] done $(date +%T)"
