#!/bin/bash
# hetero-proxy pflux V1/V2/V2b (2026-11-02): host-board all-gather / reduce-scatter vs NCCL in PCIe-only mode.
# Usage (repo root, inside tmux): ws/hetero-proxy/scripts/pflux/run_v2_v1.sh smoke|full
set -u
cd /home/rogerlee/flux
mode=${1:-smoke}
d=ws/hetero-proxy/results/pflux/v2_${mode}
mkdir -p "$d"
PCIE="NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_NET_DISABLE_INTRA=1"
rounds=200; [ "$mode" = smoke ] && rounds=2
one() {  # tag, extra env...
  local tag=$1; shift
  rm -f /dev/shm/pflux_* 2>/dev/null
  python3 common/measure/exclusive_guard_v2.py --log "$d/guard_$tag.log" -- \
    env NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH $PCIE "$@" \
    pixi run --manifest-path pixi.toml python -m torch.distributed.run --nproc_per_node 8 \
    ws/hetero-proxy/scripts/pflux/v2_hostboard_v1.py --out "$d" --tag "$tag" --rounds $rounds --job "$(date +%s)_$tag" \
    > "$d/stdout_$tag.txt" 2>&1
  echo "[$tag] exit=$? guard=$(tail -1 $d/guard_$tag.log | grep -o 'CLEAN\|CONTAMINATED')"
  grep -o "via [A-Za-z/]*" "$d/stdout_$tag.txt" | sort | uniq -c | head -5
  rm -f /dev/shm/pflux_* 2>/dev/null
}
one shm
one shmce NCCL_SHM_USE_CUDA_MEMCPY=1
echo DONE
