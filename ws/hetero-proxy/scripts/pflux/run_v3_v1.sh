#!/bin/bash
# hetero-proxy pflux V3 (2026-11-02): per-GPU bidirectional host copies, copy engine vs SM. Usage: repo root, in tmux.
set -u
cd /home/rogerlee/flux
d=ws/hetero-proxy/results/pflux/v3_${1:-full}
mkdir -p "$d"
rounds=50; [ "${1:-full}" = smoke ] && rounds=2
python3 common/measure/exclusive_guard_v2.py --log "$d/guard.log" -- \
  pixi run --manifest-path pixi.toml python ws/hetero-proxy/scripts/pflux/v3_bidir_v1.py --out "$d" --rounds $rounds \
  > "$d/stdout.txt" 2>&1
echo "exit=$? guard=$(tail -1 $d/guard.log | grep -o 'CLEAN\|CONTAMINATED')"
