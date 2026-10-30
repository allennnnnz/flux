#!/bin/bash
# validation V3 matrix (topic T): training-step prototype v2, default NCCL and NCCL_PROTO=LL128. Run in tmux.
set -u
cd /home/rogerlee/flux
S=ws/hetero-proxy/scripts/topic/v3_step_v2.py
R=ws/hetero-proxy/results/topic/v3
mkdir -p $R
run() {  # name scenario seq micro variants proto
  local d=$R/$1; mkdir -p "$d"
  local pe=""; [ "$6" != default ] && pe="NCCL_PROTO=$6"
  echo "[run] $1 $(date -u +%H:%M:%S)"
  timeout 1800 python3 common/measure/exclusive_guard_v2.py --log "$d/guard.log" -- pixi run --manifest-path pixi.toml env CUDA_DEVICE_MAX_CONNECTIONS=8 $pe \
    torchrun --nproc_per_node 8 $S --out "$d" --scenario $2 --seq $3 --micro $4 --rounds 20 --variants $5 > "$d/stdout.txt" 2>&1
  echo "  rc=$? $(tail -1 "$d/guard.log" | grep -o 'CLEAN\|CONTAMINATED')"
}
ALL=none,naive,serial,window,paced50,paced25
LLV=none,naive,window
CASES="gimm:4096:1 gimm:8192:1 gimm:16384:1 gdpu:2048:1 gdpu:2048:4 gdpu:4096:1 gdpu:4096:2 gdpu:4096:4 gdpu:8192:1 gdpu:8192:4 act:2048:1 act:4096:1 act:8192:1"
for c in $CASES; do IFS=: read sc sq mi <<< "$c"; run ${sc}_s${sq}_m${mi} $sc $sq $mi $ALL default; done
for c in $CASES; do IFS=: read sc sq mi <<< "$c"; run ${sc}_s${sq}_m${mi}_LL128 $sc $sq $mi $LLV LL128; done
echo ALLDONE
