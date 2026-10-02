#!/bin/bash
# E1 rerun: points whose kept rounds fell below 200 after the clock filter and that
# matter for the decision (near-boundary M = 1024..3072), plus M=8 (first M of each
# run, GPU still ramping from idle -> longer warmup). 350 rounds, 30 warmup.
# The results REPLACE the same (layer, M, mode) points of e1_map_v2 via merge_points_v1.py.
set -u
OUT=ws/fusion-dispatch/results/e1_rerun_v2
mkdir -p "$OUT"
S=ws/fusion-dispatch/scripts/dispatch_map_v2.py
for L in G-FC1 G-QKV L-QKV L-GU; do
  timeout 3600 pixi run --manifest-path pixi.toml ./launch.sh "$S" --layer "$L" --Ms 8,1024,1032,2048,3072 \
    --max_m 16384 --modes gpu,steady --rounds 350 --warmup 30 --clock_period 0.02 \
    --out_dir "$OUT" > "$OUT/log_$L.txt" 2>&1
  echo "$L exit $?"
done
