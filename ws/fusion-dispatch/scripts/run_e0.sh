#!/bin/bash
# E0: anchors (design 5.4) and GPU-align vs host-barrier comparison (design 5.1).
# Run from repo root: bash ws/fusion-dispatch/scripts/run_e0.sh
set -u
OUT=ws/fusion-dispatch/results/${OUT_TAG:-e0_anchor_v2}
mkdir -p "$OUT"
S=ws/fusion-dispatch/scripts/dispatch_map_v2.py
run() {  # layer Ms [modes]
  timeout 1800 pixi run --manifest-path pixi.toml ./launch.sh "$S" --layer "$1" --Ms "$2" \
    --modes "${3:-host,gpu}" --rounds 200 --warmup 10 --clock_period 0.02 --out_dir "$OUT" \
    > "$OUT/log_$1.txt" 2>&1
  echo "$1 exit $?"
}
run P0-4096 1024,4096,16384
run P0-8192 4096
run G-FC1 64,4096 host,gpu,steady
