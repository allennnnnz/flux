#!/bin/bash
# Verification queue 2 (after run_verify_queue.sh): V4b, V11, V3b. Run from repo root:
#   bash ws/fusion-dispatch/scripts/run_verify_queue2.sh <queue-1 output file>
set -u
R=ws/fusion-dispatch/results
S=ws/fusion-dispatch/scripts
Q1=${1:-}
if [ -n "$Q1" ]; then until grep -q "\[queue\] done" "$Q1"; do sleep 30; done; fi
echo "[queue2] start $(date +%T)"
# ---- V4b: 80-block decode graph, one process per M (V4 hung when several M ran in one process, see V11)
for M in 64 256 384 512; do
  mkdir -p $R/v4_block_L80/eval_M$M
  timeout 3600 bash $S/launch_vllm_env.sh $S/validate_block_v3.py --phase decode --mode graph --Ms $M \
    --table $R/v4_block_L80/table_gpu.json --L 80 --rounds 200 --warmup 5 --seed 20261001 \
    --out_dir $R/v4_block_L80/eval_M$M > $R/v4_block_L80/eval_M$M/log.txt 2>&1
  echo "[V4b] M=$M exit $?"; grep -E "^\[decode" $R/v4_block_L80/eval_M$M/log.txt | sed 's/ dispatch counts.*//'
done
python3 - <<'PY'
import csv, glob
rows, hdr = [], None
for f in sorted(glob.glob("ws/fusion-dispatch/results/v4_block_L80/eval_M*/raw_decode_graph_L80.csv")):
    r = list(csv.reader(open(f)))
    hdr, rows = r[0], rows + r[1:]
if hdr:
    with open("ws/fusion-dispatch/results/v4_block_L80/eval/raw_decode_graph_L80.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(hdr); w.writerows(rows)
    print(f"[V4b] merged {len(rows)} rows")
PY
# ---- V11: NCCL-in-graph hang repro, c10d vs vLLM pynccl
mkdir -p $R/v11_nccl_graph
for b in c10d pynccl; do
  MASTER_PORT=23461 timeout 1200 bash $S/launch_vllm_env.sh $S/nccl_graph_hang_v1.py --backend $b > $R/v11_nccl_graph/log_$b.txt 2>&1
  echo "[V11] $b exit $?"; grep -E "^\[(c10d|pynccl)\]|STUCK" $R/v11_nccl_graph/log_$b.txt | head -6
done
# ---- V3b: vLLM custom all-reduce with a raised ceiling (64 MiB): is the M=512 / prefill win only vLLM's 8 MiB cutoff?
mkdir -p $R/v3b_big_car
for pm in "decode graph 256,384,512 gpu" "prefill eager 1024,2048 steady"; do
  set -- $pm
  timeout 3600 bash $S/launch_vllm_env.sh $S/validate_block_v3.py --phase $1 --mode $2 --Ms $3 \
    --table $R/v3_block_vllm_ar/table_$4.json --L 4 --rounds 200 --warmup 10 --seed 20261004 --car_big_mib 64 \
    --out_dir $R/v3b_big_car > $R/v3b_big_car/log_$1_$2.txt 2>&1
  echo "[V3b] $1 $2 exit $?"; grep -E "^\[" $R/v3b_big_car/log_$1_$2.txt | sed 's/ dispatch counts.*//'
done
echo "[queue2] done $(date +%T)"
