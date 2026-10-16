#!/bin/bash
# hetero-proxy KV tier K2 (2026-10-16): one k2_serve_v1.py run with an SM-clock / power log next to it (one nvidia-smi
# -lms process pinned to an idle NUMA-1 core). A serving run reports the clock rather than filtering it: sustained
# load under the power cap is the deployed state.
# Usage (repo root, inside exclusive_guard_v2):  ws/hetero-proxy/scripts/kvtier/run_k2_v1.sh <out_dir> <k2_serve args...>
set -u
out=$1
shift
mkdir -p "$out"
taskset -c 127 nvidia-smi --query-gpu=timestamp,index,clocks.sm,power.draw,temperature.gpu --format=csv,noheader \
    -lms 500 >> "$out/clocks.csv" &
smi=$!
env -u CUDA_DEVICE_MAX_CONNECTIONS CUDA_MODULE_LOADING=EAGER pixi run --manifest-path pixi.toml \
    ~/.venvs/hetero-proxy-hf/bin/python -m torch.distributed.run --nproc_per_node 8 \
    ws/hetero-proxy/scripts/kvtier/k2_serve_v1.py --out "$out" "$@"
rc=$?
kill $smi
exit $rc
