#!/bin/bash
# hetero-proxy KV tier (2026-10-16): one bench / check run of a kvtier script (KVTIER env, default kvtier_v5.py)
# with an SM-clock log next to it.
# The clock log is one nvidia-smi process (-lms loop, every GPU, pinned to an idle NUMA-1 core) so the 8 ranks'
# cores stay free; the analysis joins its samples with the job windows written by the bench.
# Usage (repo root, inside exclusive_guard_v2):  ws/hetero-proxy/scripts/kvtier/run_bench_v1.sh <out_dir> <kvtier args...>
set -u
out=$1
shift
mkdir -p "$out"
taskset -c 127 nvidia-smi --query-gpu=timestamp,index,clocks.sm,power.draw,temperature.gpu --format=csv,noheader \
    -lms 200 > "$out/clocks.csv" &
smi=$!
env -u CUDA_DEVICE_MAX_CONNECTIONS CUDA_MODULE_LOADING=EAGER pixi run --manifest-path pixi.toml \
    ~/.venvs/hetero-proxy-hf/bin/python -m torch.distributed.run --nproc_per_node 8 \
    ws/hetero-proxy/scripts/kvtier/${KVTIER:-kvtier_v5.py} --out "$out" "$@"
rc=$?
kill $smi
exit $rc
