#!/bin/bash
# topic search T1b (2026-10-30): only some GPUs copy (GPU0 alone; GPU0+GPU1 = one PCIe switch); default NCCL.
set -u
cd /home/rogerlee/flux
S=ws/hetero-proxy/scripts/topic/t1_bigcoll_pcie_v1.py
for v in 0 0,1; do
  d=ws/hetero-proxy/results/topic/t1b_v1_bg${v/,/_}
  mkdir -p "$d"
  echo "[run] bg_ranks=$v $(date -u +%H:%M:%S)"
  python3 common/measure/exclusive_guard_v2.py --log "$d/guard.log" -- env CUDA_DEVICE_MAX_CONNECTIONS=8 \
    pixi run --manifest-path pixi.toml torchrun --nproc_per_node 8 $S --out "$d" --rounds 30 --sizes_mb 4,16,32,64,256 --bg_ranks $v \
    > "$d/stdout.txt" 2>&1
  tail -1 "$d/guard.log"
done
echo ALLDONE
