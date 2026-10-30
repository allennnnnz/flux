#!/bin/bash
# FS step 3b (2026-10-30): incremental rebuild + install of Flux after the arrival-aware swizzle edit. Run in tmux.
set -u
cd /home/rogerlee/flux
L=ws/hetero-proxy/results/topic/fs3_build_v1.log
echo "[build] start $(date -u +%H:%M:%S)" > $L
t0=$(date +%s)
pixi run --manifest-path pixi.toml make -C build -j 96 install >> $L 2>&1
rc=$?
echo "[build] rc=$rc elapsed $(( $(date +%s) - t0 )) s end $(date -u +%H:%M:%S)" >> $L
echo BUILDDONE >> $L
