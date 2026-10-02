#!/bin/bash
# Resumable remainder of the verification queues (V7 rest, V4b, V11, V3b, V6 A/B, V9), meant to run
# inside tmux so it survives an SSH / VS Code disconnect. Every finished item leaves a marker in
# results/verify_done/; re-running this script skips marked items.
# Start (repo root):  tmux new-session -d -s fusion_verify 'bash ws/fusion-dispatch/scripts/run_verify_resume.sh'
# Log: ws/fusion-dispatch/results/verify_resume_log.txt
cd /home/rogerlee/flux
R=ws/fusion-dispatch/results
S=ws/fusion-dispatch/scripts
D=$R/verify_done
mkdir -p $D
exec >> $R/verify_resume_log.txt 2>&1
echo "[resume] start $(date '+%F %T')"
mark() { echo "$2" > "$D/$1"; }

# ---- V7: remaining reruns of points with < 200 kept rounds
while read h L mode r wu script Ms extra mm; do
  tag=v7_${h}_${L}_${mode}_${r}_${Ms//,/-}
  [ -e $D/$tag ] && { echo "[skip] $tag"; continue; }
  mkdir -p $R/v7_rerun/$h
  timeout 3600 pixi run --manifest-path pixi.toml ./launch.sh $S/$script --layer $L --Ms $Ms $extra $mm --modes $mode \
    --rounds $r --warmup $wu --clock_period 0.02 --out_dir $R/v7_rerun/$h > $R/v7_rerun/$h/log_${L}_${mode}_$r.txt 2>&1
  rc=$?; echo "[V7] $h $L $mode rounds=$r Ms=$Ms exit $rc $(date +%T)"; [ $rc -eq 0 ] && mark $tag "exit 0"
done < $R/v7_rerun_cmds.txt

# ---- V4b: 80-block decode graph, one process per M
for M in 64 256 384 512; do
  tag=v4b_M$M; [ -e $D/$tag ] && { echo "[skip] $tag"; continue; }
  mkdir -p $R/v4_block_L80/eval_M$M
  timeout 3600 bash $S/launch_vllm_env.sh $S/validate_block_v3.py --phase decode --mode graph --Ms $M \
    --table $R/v4_block_L80/table_gpu.json --L 80 --rounds 200 --warmup 5 --seed 20261001 \
    --out_dir $R/v4_block_L80/eval_M$M > $R/v4_block_L80/eval_M$M/log.txt 2>&1
  rc=$?; echo "[V4b] M=$M exit $rc $(date +%T)"; grep -E "^\[decode" $R/v4_block_L80/eval_M$M/log.txt | sed 's/ dispatch counts.*//'
  mark $tag "exit $rc"
done
python3 - <<'PY'
import csv, glob
rows, hdr = [], None
for f in sorted(glob.glob("ws/fusion-dispatch/results/v4_block_L80/eval_M*/raw_decode_graph_L80.csv")):
    r = list(csv.reader(open(f))); hdr, rows = r[0], rows + r[1:]
if hdr:
    with open("ws/fusion-dispatch/results/v4_block_L80/eval/raw_decode_graph_L80.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(hdr); w.writerows(rows)
    print(f"[V4b] merged {len(rows)} rows")
PY

# ---- V11: NCCL-in-graph hang repro (exit code 3 = watchdog saw a hang; that is a result, not a failure)
mkdir -p $R/v11_nccl_graph
for b in c10d pynccl; do
  tag=v11_$b; [ -e $D/$tag ] && { echo "[skip] $tag"; continue; }
  MASTER_PORT=23461 timeout 1200 bash $S/launch_vllm_env.sh $S/nccl_graph_hang_v1.py --backend $b > $R/v11_nccl_graph/log_$b.txt 2>&1
  rc=$?; echo "[V11] $b exit $rc $(date +%T)"; grep -E "^\[(c10d|pynccl)\]|STUCK" $R/v11_nccl_graph/log_$b.txt | head -6
  mark $tag "exit $rc"
  sleep 20  # let a hung job's ports / GPU memory be released
done

# ---- V3b: vLLM custom all-reduce with a 64 MiB ceiling
mkdir -p $R/v3b_big_car
for pm in "decode graph 256,384,512 gpu" "prefill eager 1024,2048 steady"; do
  set -- $pm
  tag=v3b_$1_$2; [ -e $D/$tag ] && { echo "[skip] $tag"; continue; }
  timeout 3600 bash $S/launch_vllm_env.sh $S/validate_block_v3.py --phase $1 --mode $2 --Ms $3 \
    --table $R/v3_block_vllm_ar/table_$4.json --L 4 --rounds 200 --warmup 10 --seed 20261004 --car_big_mib 64 \
    --out_dir $R/v3b_big_car > $R/v3b_big_car/log_$1_$2.txt 2>&1
  rc=$?; echo "[V3b] $1 $2 exit $rc $(date +%T)"; grep -E "^\[" $R/v3b_big_car/log_$1_$2.txt | sed 's/ dispatch counts.*//'
  [ $rc -eq 0 ] && mark $tag "exit 0"
done

# ---- V6 A/B: which protocol element changes NCCL's reading
tag=v6ab
if [ ! -e $D/$tag ]; then
  timeout 1800 pixi run --manifest-path pixi.toml ./launch.sh $S/v6_protocol_ab.py --out $R/v6_phase0_repro/protocol_ab.json \
    > $R/v6_phase0_repro/log_protocol_ab.txt 2>&1
  rc=$?; echo "[V6ab] exit $rc $(date +%T)"; grep -E "^\[M=" $R/v6_phase0_repro/log_protocol_ab.txt
  [ $rc -eq 0 ] && mark $tag "exit 0"
fi

# ---- V9: Triton AllGather prototype under rank skew
mkdir -p $R/v9_triton_stress
for spec in "64 200" "64 2000" "1024 200"; do
  set -- $spec
  tag=v9_M$1_skew$2; [ -e $D/$tag ] && { echo "[skip] $tag"; continue; }
  timeout 1800 pixi run --manifest-path pixi.toml ./launch.sh $S/triton_ag_stress_v1.py --iters 5000 --M $1 --K 8192 \
    --max_skew_us $2 --out $R/v9_triton_stress/stress_M$1_skew$2.json > $R/v9_triton_stress/log_M$1_skew$2.txt 2>&1
  rc=$?; echo "[V9] M=$1 skew=$2 exit $rc $(date +%T)"; grep -E "^\[(single|double)\]" $R/v9_triton_stress/log_M$1_skew$2.txt
  [ $rc -eq 0 ] && mark $tag "exit 0"
done
echo "[resume] done $(date '+%F %T')"
touch $D/ALL_DONE
