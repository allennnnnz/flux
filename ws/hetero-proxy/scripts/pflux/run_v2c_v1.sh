#!/bin/bash
# hetero-proxy pflux V2c (2026-11-02): reduce-scatter with CPU-pushed H2D vs NCCL in PCIe-only mode. Repo root, tmux.
set -u
cd /home/rogerlee/flux
mode=${1:-full}
d=ws/hetero-proxy/results/pflux/v2c_${mode}
mkdir -p "$d"
rounds=200; [ "$mode" = smoke ] && rounds=2
rm -f /dev/shm/pflux_* 2>/dev/null
python3 common/measure/exclusive_guard_v2.py --log "$d/guard_shm.log" -- \
  env NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_NET_DISABLE_INTRA=1 \
  pixi run --manifest-path pixi.toml python -m torch.distributed.run --nproc_per_node 8 \
  ws/hetero-proxy/scripts/pflux/v2c_rspush_v1.py --out "$d" --tag shm --rounds $rounds --job "$(date +%s)_c" \
  > "$d/stdout_shm.txt" 2>&1
echo "[shm] exit=$? guard=$(tail -1 $d/guard_shm.log | grep -o 'CLEAN\|CONTAMINATED')"
grep -o "via [A-Za-z/]*" "$d/stdout_shm.txt" | sort | uniq -c | head -5
rm -f /dev/shm/pflux_* 2>/dev/null
echo DONE
