#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT_DIR}"

EXP_DIR="experiments/2026-09-23-phase0-a100-nvlink-pcie"
DUAL_PATH="${EXP_DIR}/scripts/dual_path_bench.py"
BWTEST="${BWTEST:-/home/rogerlee/cuda-12.5/extras/demo_suite/bandwidthTest}"

echo "== Environment =="
pixi run --manifest-path pixi.toml python - <<'PY'
import torch
import nvidia.nvshmem

print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
PY

echo "== Topology =="
nvidia-smi topo -m
nvidia-smi topo -p2p r
numactl -H

echo "== CUDA bandwidthTest =="
if [[ -x "${BWTEST}" ]]; then
  "${BWTEST}" --memory=pinned --mode=shmoo --start=33554432 --end=33554432 --increment=33554432 --dtoh
  "${BWTEST}" --memory=pinned --mode=shmoo --start=33554432 --end=33554432 --increment=33554432 --htod
  "${BWTEST}" --mode=shmoo --start=33554432 --end=33554432 --increment=33554432 --dtod
else
  echo "Skipping bandwidthTest: ${BWTEST} is not executable"
fi

echo "== Dual-path benchmark =="
pixi run --manifest-path pixi.toml python "${DUAL_PATH}" \
  --bytes-per-gpu $((64 * 1024 * 1024)) \
  --repeats 5

if command -v numactl >/dev/null 2>&1; then
  echo "== Dual-path benchmark, NUMA node 0 =="
  numactl --cpunodebind=0 --membind=0 pixi run --manifest-path pixi.toml python "${DUAL_PATH}" \
    --bytes-per-gpu $((64 * 1024 * 1024)) \
    --repeats 5

  echo "== Dual-path benchmark, NUMA node 1 =="
  numactl --cpunodebind=1 --membind=1 pixi run --manifest-path pixi.toml python "${DUAL_PATH}" \
    --bytes-per-gpu $((64 * 1024 * 1024)) \
    --repeats 5
fi

echo "== Flux AG GEMM benchmarks =="
for m in 64 512 1024 2048 4096 8192; do
  NVSHMEM_REMOTE_TRANSPORT=none pixi run --manifest-path pixi.toml \
    ./launch.sh test/python/ag_gemm/test_ag_kernel.py "${m}" 49152 12288 \
    --dtype=float16 --warmup=2 --iters=5 --verify
done

echo "== Flux GEMM RS benchmarks =="
for m in 64 512 1024 2048 4096 8192; do
  NVSHMEM_REMOTE_TRANSPORT=none pixi run --manifest-path pixi.toml \
    ./launch.sh test/python/gemm_rs/test_gemm_rs.py "${m}" 12288 49152 \
    --dtype=float16 --warmup=2 --iters=5
done
