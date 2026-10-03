#!/bin/bash
# fusion-dispatch E4: cross-node NCCL anchors at 1+1, 4+4 and 8+8 GPUs (NCCL defaults; both nodes guarded).
# Usage (repo root, css-host-158, inside tmux): bash ws/fusion-dispatch/scripts/run_e4_anchor_nccl_v1.sh
OUT=ws/fusion-dispatch/results/e4_anchor_nccl
echo "[E4-anchor] start $(date '+%F %T')" | tee -a $OUT/run_log.txt
for n in 1 4 8; do
  XNODE_TIMEOUT=900 bash ws/fusion-dispatch/scripts/run_xnode.sh $OUT/ppn$n $n pixi \
    ws/fusion-dispatch/scripts/xnode_nccl_anchor_v1.py --out $OUT/ppn$n --iters 30 2>&1 | tail -1 | tee -a $OUT/run_log.txt
done
echo "[E4-anchor] done $(date '+%F %T')" | tee -a $OUT/run_log.txt
