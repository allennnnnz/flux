#!/bin/bash
# hetero-proxy M0 (2026-10-29): formal run of m0_split_v1.py under the exclusive guard; start inside tmux.
set -u
cd /home/rogerlee/flux
d=ws/hetero-proxy/results/moehy/m0_v1
mkdir -p "$d"
echo "[run] $d $(date -u +%H:%M:%S)"
python3 common/measure/exclusive_guard_v2.py --log "$d/guard.log" -- env CUDA_MODULE_LOADING=EAGER \
  pixi run --manifest-path pixi.toml python ws/hetero-proxy/scripts/moehy/m0_split_v1.py --out "$d" --rounds 30 \
  > "$d/stdout.txt" 2>&1
tail -1 "$d/guard.log"
echo ALLDONE
