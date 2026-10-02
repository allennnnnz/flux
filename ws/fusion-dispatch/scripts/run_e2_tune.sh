#!/bin/bash
# E2a: tuning upside without rebuild (tune_upside_v1.py). Run from repo root.
set -u
OUT=ws/fusion-dispatch/results/e2_tune
mkdir -p "$OUT"
run() {  # layer Ms tag
  timeout 1800 pixi run --manifest-path pixi.toml ./launch.sh ws/fusion-dispatch/scripts/tune_upside_v1.py \
    --layer "$1" --Ms "$2" --out "$OUT/tune_$1_$3.json" > "$OUT/log_$1_$3.txt" 2>&1
  echo "$1 $2 exit $?"; grep -E "^\[" "$OUT/log_$1_$3.txt"
}
run G-FC1 64,512 b
run L-GU 520,4096 a
run L-QKV 64,1024 a
