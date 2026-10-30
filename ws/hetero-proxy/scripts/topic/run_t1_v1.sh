#!/bin/bash
# topic search T1 (2026-10-30): training-size NCCL collectives vs background PCIe copies; default / Simple / LL128.
# Run inside tmux. Each variant is one guarded torchrun.
set -u
cd /home/rogerlee/flux
S=ws/hetero-proxy/scripts/topic/t1_bigcoll_pcie_v1.py
for v in default Simple LL128; do
  d=ws/hetero-proxy/results/topic/t1_v1_$v
  mkdir -p "$d"
  extra=""
  [ "$v" != default ] && extra="NCCL_PROTO=$v"
  echo "[run] $v $(date -u +%H:%M:%S)"
  python3 common/measure/exclusive_guard_v2.py --log "$d/guard.log" -- env CUDA_DEVICE_MAX_CONNECTIONS=8 $extra \
    pixi run --manifest-path pixi.toml torchrun --nproc_per_node 8 $S --out "$d" --rounds 30 --sizes_mb 1,4,16,32,64,128,256 \
    > "$d/stdout.txt" 2>&1
  tail -1 "$d/guard.log"
done
echo ALLDONE
