#!/bin/bash
# V9b: Triton AG stress with skew between barrier and gather. V6c: SM clock during the original
# Phase 0 comm script. Run from repo root.
cd /home/rogerlee/flux
R=ws/fusion-dispatch/results; S=ws/fusion-dispatch/scripts
for spec in "64 200 after_barrier" "64 2000 after_barrier" "1024 500 both"; do
  set -- $spec
  timeout 1800 pixi run --manifest-path pixi.toml ./launch.sh $S/triton_ag_stress_v2.py --iters 5000 --M $1 --K 8192 \
    --max_skew_us $2 --skew_at $3 --out $R/v9_triton_stress/stress_v2_M$1_skew$2_$3.json > $R/v9_triton_stress/log_v2_M$1_skew$2_$3.txt 2>&1
  echo "[V9b] M=$1 skew=$2 at=$3 exit $?"; grep -E "^\[(single|double)\]" $R/v9_triton_stress/log_v2_M$1_skew$2_$3.txt
done
# V6c: log SM clock of GPU0 (50 ms) while the unmodified Phase 0 script runs M=4096 (NCCL first, then 7 Flux configs)
mkdir -p $R/v6_phase0_repro
nvidia-smi --query-gpu=timestamp,index,clocks.sm,utilization.gpu,power.draw --format=csv,noheader -i 0,4 -lms 50 > $R/v6_phase0_repro/v6c_clock.csv &
CLK=$!
for rep in 1 2; do
  echo "SCRIPT_START rep=$rep $(date '+%Y/%m/%d %H:%M:%S.%N')" >> $R/v6_phase0_repro/v6c_marks.txt
  timeout 600 pixi run --manifest-path pixi.toml ./launch.sh experiments/2026-09-23-phase0-a100-nvlink-pcie/scripts/flux_comm_baseline.py 4096 12288 \
    --dtype=float16 --warmup=5 --iters=30 --out=$R/v6_phase0_repro/v6c_M4096_rep$rep.csv > $R/v6_phase0_repro/log_v6c_rep$rep.txt 2>&1
  echo "SCRIPT_END rep=$rep $(date '+%Y/%m/%d %H:%M:%S.%N')" >> $R/v6_phase0_repro/v6c_marks.txt
  grep -E "nccl_all_gather|flux_all2all_pull " $R/v6_phase0_repro/log_v6c_rep$rep.txt
done
kill $CLK
echo "[V6c] done"
