#!/bin/bash
# validation V3b (topic T): hybrid schedule (window + paced remainder) on the delayed-update cases; default and LL128.
set -u
cd /home/rogerlee/flux
S=ws/hetero-proxy/scripts/topic/v3_step_v3.py
R=ws/hetero-proxy/results/topic/v3b
mkdir -p $R
for proto in default LL128; do
  for c in 4096:1 4096:2 4096:4 8192:1; do
    IFS=: read sq mi <<< "$c"
    name=gdpu_s${sq}_m${mi}; [ $proto != default ] && name=${name}_LL128
    d=$R/$name; mkdir -p $d
    pe=""; [ $proto != default ] && pe="NCCL_PROTO=$proto"
    echo "[run] $name $(date -u +%H:%M:%S)"
    timeout 1800 python3 common/measure/exclusive_guard_v2.py --log $d/guard.log -- pixi run --manifest-path pixi.toml env CUDA_DEVICE_MAX_CONNECTIONS=8 $pe \
      torchrun --nproc_per_node 8 $S --out $d --scenario gdpu --seq $sq --micro $mi --rounds 20 --variants none,naive,serial,window,hybrid > $d/stdout.txt 2>&1
    echo "  rc=$? $(tail -1 $d/guard.log | grep -o 'CLEAN\|CONTAMINATED')"
  done
done
echo ALLDONE
