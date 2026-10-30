#!/bin/bash
# FS1b (2026-10-30): concurrent AG + GEMM (no waiting) vs fused vs ideal; waits for FS1 to finish first. Run in tmux.
set -u
cd /home/rogerlee/flux
until grep -q ALLDONE ws/hetero-proxy/results/topic/run_fs1_v1.log 2>/dev/null; do sleep 10; done
d=ws/hetero-proxy/results/topic/fs1b_v1
mkdir -p $d
echo "[run] $(date -u +%H:%M:%S)"
timeout 3600 python3 common/measure/exclusive_guard_v2.py --log $d/guard.log -- env CUDA_DEVICE_MAX_CONNECTIONS=8 pixi run --manifest-path pixi.toml \
  ./launch.sh ws/hetero-proxy/scripts/topic/fs1b_concurrent_v1.py --out $d --rounds 10 > $d/stdout.txt 2>&1
echo "rc=$? $(tail -1 $d/guard.log | grep -o 'CLEAN\|CONTAMINATED') $(date -u +%H:%M:%S)"
echo ALLDONE
