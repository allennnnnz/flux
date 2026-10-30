#!/bin/bash
# validation V1 (pacing, channel count) and V2 (NCCL 2.32.3) for topic T; plan: reports/20261030_t_validation_plan.md. Run in tmux.
set -u
cd /home/rogerlee/flux
S=ws/hetero-proxy/scripts/topic/t1_bigcoll_pcie_v2.py
R=ws/hetero-proxy/results/topic
N232=/home/rogerlee/opt/nccl-2.32.3/nvidia/nccl/lib/libnccl.so.2
run() {  # name, env..., -- args...
  local name=$1; shift
  local envs=()
  while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  local d=$R/$name; mkdir -p "$d"
  echo "[run] $name $(date -u +%H:%M:%S)"
  python3 common/measure/exclusive_guard_v2.py --log "$d/guard.log" -- pixi run --manifest-path pixi.toml env CUDA_DEVICE_MAX_CONNECTIONS=8 "${envs[@]}" \
    torchrun --nproc_per_node 8 $S --out "$d" --rounds 30 "$@" > "$d/stdout.txt" 2>&1
  tail -1 "$d/guard.log"
}
run v1a_pacing X=1 -- --sizes_mb 16,32,64 --ops all_reduce,reduce_scatter --modes none,d2h_100,d2h_50,d2h_25,h2d_100
run v1b_base X=1 -- --sizes_mb 16,32,64,128,256 --modes none,d2h_100
run v1b_min32 NCCL_MIN_NCHANNELS=32 -- --sizes_mb 16,32,64,128,256 --modes none,d2h_100
run v1b_max4 NCCL_MAX_NCHANNELS=4 -- --sizes_mb 16,32,64,128,256 --modes none,d2h_100
run v2_nccl232 LD_PRELOAD=$N232 -- --sizes_mb 4,16,32,64,128,256 --modes none,d2h_100
run v2_nccl232_LL128 LD_PRELOAD=$N232 NCCL_PROTO=LL128 -- --sizes_mb 4,16,32,64,128,256 --modes none,d2h_100
echo ALLDONE
