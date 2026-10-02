#!/bin/bash
# F2a + F2b: AllGather latency (ag_latency_v1.py). K=8192 (Llama-3-70B) and K=12288 (GPT-3), gpu + steady.
# Run from repo root.
set -u
OUT=ws/fusion-dispatch/results/${OUT_TAG:-f2_ag_latency_v1}
mkdir -p "$OUT"
S=ws/fusion-dispatch/scripts/ag_latency_v1.py
for kn in "8192 1280" "12288 6144"; do
  set -- $kn
  timeout 3600 pixi run --manifest-path pixi.toml ./launch.sh $S --K $1 --n_cols $2 --Ms 8,64,256,512,1024,2048,4096 \
    --modes gpu,steady --rounds 200 --warmup 30 --out_dir $OUT > $OUT/log_K$1.txt 2>&1
  echo "K=$1 exit $?"; grep -E "^\[K=.*(gpu|steady)\]" $OUT/log_K$1.txt
done
