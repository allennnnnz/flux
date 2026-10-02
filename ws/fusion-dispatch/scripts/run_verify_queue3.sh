#!/bin/bash
# Verification queue 3 (after queue 2): V6 protocol A/B. Run from repo root with queue-2 output file.
set -u
R=ws/fusion-dispatch/results; S=ws/fusion-dispatch/scripts
Q2=${1:-}
if [ -n "$Q2" ]; then until grep -q "\[queue2\] done" "$Q2"; do sleep 30; done; fi
timeout 1800 pixi run --manifest-path pixi.toml ./launch.sh $S/v6_protocol_ab.py --out $R/v6_phase0_repro/protocol_ab.json \
  > $R/v6_phase0_repro/log_protocol_ab.txt 2>&1
echo "[V6ab] exit $?"; grep -E "^\[M=" $R/v6_phase0_repro/log_protocol_ab.txt
echo "[queue3] done $(date +%T)"
