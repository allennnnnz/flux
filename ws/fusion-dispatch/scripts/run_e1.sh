#!/bin/bash
# E1: dispatch map (design 4, 6). 4 main layers x 20 M x {gpu, steady}.
# M = main grid (powers of 2) + held-out points + cliff-pair partners (design 4.2).
# Run from repo root: bash ws/fusion-dispatch/scripts/run_e1.sh [layer ...]
set -u
OUT=ws/fusion-dispatch/results/${OUT_TAG:-e1_map_v2}
mkdir -p "$OUT"
S=ws/fusion-dispatch/scripts/dispatch_map_v2.py
MS=8,16,24,32,64,72,128,136,256,264,512,520,1024,1032,2048,3072,4096,6144,8192,16384
LAYERS=${@:-G-FC1 G-QKV L-QKV L-GU}
for L in $LAYERS; do
  timeout 3600 pixi run --manifest-path pixi.toml ./launch.sh "$S" --layer "$L" --Ms "$MS" \
    --max_m 16384 --modes gpu,steady --rounds 200 --warmup 10 --clock_period 0.02 \
    --out_dir "$OUT" > "$OUT/log_$L.txt" 2>&1
  echo "$L exit $?"
done
