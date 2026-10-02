#!/bin/bash
# V12: re-measure key points under common/measure/exclusive_guard.py (machine verified exclusive:
# preflight + 1 s monitoring) and compare with the 2026-09-30 numbers. Run from repo root (tmux ok).
cd /home/rogerlee/flux
R=ws/fusion-dispatch/results/v12_exclusive_recheck; S=ws/fusion-dispatch/scripts; G=common/measure/exclusive_guard.py
mkdir -p $R
exec >> $R/run_log.txt 2>&1
echo "[V12] start $(date '+%F %T')"
python3 $G --log $R/guard_op.log -- timeout 1800 pixi run --manifest-path pixi.toml ./launch.sh $S/dispatch_map_v2.py \
  --layer G-FC1 --Ms 64,4096 --max_m 16384 --modes gpu --rounds 200 --warmup 30 --clock_period 0.02 --out_dir $R/op > $R/log_op.txt 2>&1
echo "[V12] op exit $? $(tail -1 $R/guard_op.log)"
for pm in "decode graph 64,512 gpu" "prefill eager 1024 steady"; do
  set -- $pm
  python3 $G --log $R/guard_block_$1.log -- timeout 1800 bash $S/launch_vllm_env.sh $S/validate_block_v3.py --phase $1 --mode $2 \
    --Ms $3 --table ws/fusion-dispatch/results/v3_block_vllm_ar/table_$4.json --L 4 --rounds 200 --warmup 10 --seed 20261001 \
    --out_dir $R/block > $R/log_block_$1.txt 2>&1
  echo "[V12] block $1 $2 exit $? $(tail -1 $R/guard_block_$1.log)"
  grep -E "^\[(decode|prefill)" $R/log_block_$1.txt | sed 's/ dispatch counts.*//'
done
echo "[V12] done $(date '+%F %T')"
