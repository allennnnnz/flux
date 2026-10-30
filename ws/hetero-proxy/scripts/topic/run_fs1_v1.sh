#!/bin/bash
# FS step 1 (2026-10-30): config-only recovery of Flux AG+GEMM overlap gap; plan reports/20261030_fs_validation_plan.md. Run in tmux.
set -u
cd /home/rogerlee/flux
d=ws/hetero-proxy/results/topic/fs1_v1
mkdir -p $d
echo "[run] $(date -u +%H:%M:%S)"
timeout 7200 python3 common/measure/exclusive_guard_v2.py --log $d/guard.log -- env CUDA_DEVICE_MAX_CONNECTIONS=8 pixi run --manifest-path pixi.toml \
  ./launch.sh ws/hetero-proxy/scripts/topic/fs1_profile_v1.py --out $d --rounds 5 > $d/stdout.txt 2>&1
echo "rc=$? $(tail -1 $d/guard.log | grep -o 'CLEAN\|CONTAMINATED') $(date -u +%H:%M:%S)"
echo ALLDONE
