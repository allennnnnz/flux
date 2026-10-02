#!/bin/bash
# F0.3 / E5: GEMM+ReduceScatter dispatch map. 4 row-parallel layers x 20 M x {gpu, steady}.
# Same M grid as E1 (main + held-out). warmup 30 (E1 lesson: first M saw clock ramp-up).
# Run from repo root: bash ws/fusion-dispatch/scripts/run_e5.sh [layer ...]
set -u
OUT=ws/fusion-dispatch/results/${OUT_TAG:-e5_rs_map_v1}
mkdir -p "$OUT"
S=ws/fusion-dispatch/scripts/dispatch_map_rs_v1.py
MS=8,16,24,32,64,72,128,136,256,264,512,520,1024,1032,2048,3072,4096,6144,8192,16384
LAYERS=${@:-L-O L-down G-O G-FC2}
for L in $LAYERS; do
  timeout 3600 pixi run --manifest-path pixi.toml ./launch.sh "$S" --layer "$L" --Ms "$MS" \
    --max_m 16384 --modes gpu,steady --rounds 200 --warmup 30 --clock_period 0.02 \
    --out_dir "$OUT" > "$OUT/log_$L.txt" 2>&1
  echo "$L exit $?"
done
