#!/bin/bash
# hetero-proxy pflux V3c (2026-11-02): CPU-pushed H2D (GDRCopy) with / without concurrent CE D2H. Repo root, in tmux.
set -u
cd /home/rogerlee/flux
d=ws/hetero-proxy/results/pflux/v3c_${1:-full}
mkdir -p "$d"
rounds=30; [ "${1:-full}" = smoke ] && rounds=2
python3 common/measure/exclusive_guard_v2.py --log "$d/guard.log" -- \
  pixi run --manifest-path pixi.toml python ws/hetero-proxy/scripts/pflux/v3c_gdrpush_v1.py --out "$d" --rounds $rounds \
  > "$d/stdout.txt" 2>&1
echo "exit=$? guard=$(tail -1 $d/guard.log | grep -o 'CLEAN\|CONTAMINATED')"
