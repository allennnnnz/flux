#!/bin/bash
# G4-block step 1-2: layout calibration (calibrate_block_v1.py, vLLM venv) at TP=8 and TP=4, and the
# standalone GEMM probes at the block M buckets. Each process under exclusive_guard.
cd /home/rogerlee/flux
S=ws/fusion-dispatch/scripts; G=common/measure/exclusive_guard.py
R=ws/fusion-dispatch/results
mkdir -p $R/g4_block_cal_tp8 $R/g4_block_cal_tp4 $R/g4_block_gemm
exec >> $R/g4_block_gemm/run_log.txt 2>&1
echo "[G4-block-prep] start $(date '+%F %T')"
for tp in 8 4; do
  python3 $G --log $R/g4_block_cal_tp$tp/guard.log -- timeout 1200 env TP=$tp bash $S/launch_vllm_env.sh \
    $S/calibrate_block_v1.py --out_dir $R/g4_block_cal_tp$tp > $R/g4_block_cal_tp$tp/log.txt 2>&1
  echo "[G4-block-prep] block-cal tp$tp exit $? $(date '+%T') $(tail -1 $R/g4_block_cal_tp$tp/guard.log)"
done
MS=32,128,256,384,512,1024,2048,4096
for cfg in "8 qwen2.5-32b" "4 qwen2.5-32b" "4 llama3-8b"; do
  set -- $cfg; tp=$1; model=$2
  if [ "$tp" = 8 ]; then L=./launch.sh; else L="$S/launch_tp.sh $tp"; fi
  python3 $G --log $R/g4_block_gemm/guard_${model}_tp$tp.log -- timeout 1800 pixi run --manifest-path pixi.toml $L \
    $S/probe_gemm_v1.py --model $model --Ms $MS --out_dir $R/g4_block_gemm > $R/g4_block_gemm/log_${model}_tp$tp.txt 2>&1
  echo "[G4-block-prep] gemm tp$tp $model exit $? $(date '+%T') $(tail -1 $R/g4_block_gemm/guard_${model}_tp$tp.log)"
done
echo "[G4-block-prep] done $(date '+%F %T')"
touch $R/g4_block_gemm/DONE
