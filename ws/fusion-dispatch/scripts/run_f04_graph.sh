#!/bin/bash
# F0.4: CUDA graph capture feasibility, one process per item. Run from repo root.
set -u
OUT=ws/fusion-dispatch/results/f04_graph
mkdir -p "$OUT"
for it in ${ITEMS:-B_nccl AR_nccl R_nccl C_fluxag A_fused R_fused}; do
  timeout 600 pixi run --manifest-path pixi.toml ./launch.sh ws/fusion-dispatch/scripts/graph_capture_v1.py \
    --item $it --Ms 64,1024 --out "$OUT/$it.json" > "$OUT/log_$it.txt" 2>&1
  echo "$it exit $?"; grep -E "^\[$it" "$OUT/log_$it.txt"
done
