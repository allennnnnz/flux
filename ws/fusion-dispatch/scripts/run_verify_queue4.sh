#!/bin/bash
# Verification queue 4 (after queue 3): V9 rerun (triton_ag_stress_v1.py clock fix). Run from repo root.
set -u
R=ws/fusion-dispatch/results; S=ws/fusion-dispatch/scripts
Q3=${1:-}
if [ -n "$Q3" ]; then until grep -q "\[queue3\] done" "$Q3"; do sleep 30; done; fi
for spec in "64 200" "64 2000" "1024 200"; do
  set -- $spec
  timeout 1800 pixi run --manifest-path pixi.toml ./launch.sh $S/triton_ag_stress_v1.py --iters 5000 --M $1 --K 8192 \
    --max_skew_us $2 --out $R/v9_triton_stress/stress_M$1_skew$2.json > $R/v9_triton_stress/log_M$1_skew$2.txt 2>&1
  echo "[V9] M=$1 skew=$2 exit $?"; grep -E "^\[(single|double)\]" $R/v9_triton_stress/log_M$1_skew$2.txt
done
echo "[queue4] done $(date +%T)"
