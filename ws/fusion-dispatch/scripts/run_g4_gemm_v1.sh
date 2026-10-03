#!/bin/bash
# G4 step 2: standalone GEMM probes (probe_gemm_v1.py) for the fresh test configs, each under
# exclusive_guard, one process per (TP, model). Usage: bash run_g4_gemm_v1.sh [out_dir]
cd /home/rogerlee/flux
R=${1:-ws/fusion-dispatch/results/g4_gemm}
S=ws/fusion-dispatch/scripts; G=common/measure/exclusive_guard.py
MS=24,96,200,384,768,1536,2560,6144
mkdir -p $R
exec >> $R/run_log.txt 2>&1
echo "[G4-gemm] start $(date '+%F %T')"
for cfg in "8 qwen2.5-32b" "4 qwen2.5-32b" "4 llama3-8b"; do
  set -- $cfg; tp=$1; model=$2
  if [ "$tp" = 8 ]; then L=./launch.sh; else L="$S/launch_tp.sh $tp"; fi
  python3 $G --log $R/guard_${model}_tp$tp.log -- timeout 1800 pixi run --manifest-path pixi.toml $L \
    $S/probe_gemm_v1.py --model $model --Ms $MS --out_dir $R > $R/log_${model}_tp$tp.txt 2>&1
  echo "[G4-gemm] tp$tp $model exit $? $(date '+%T') $(tail -1 $R/guard_${model}_tp$tp.log)"
done
echo "[G4-gemm] done $(date '+%F %T')"
touch $R/DONE
