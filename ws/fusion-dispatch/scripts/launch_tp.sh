#!/bin/bash
# Launch a torchrun job on the first TP GPUs (launch.sh always uses every GPU: its nproc_per_node comes
# from `nvidia-smi --list-gpus`, which ignores CUDA_VISIBLE_DEVICES). Same environment as launch.sh;
# launch.sh itself is not modified (plan appendix A.2).
# Usage (repo root): pixi run --manifest-path pixi.toml ws/fusion-dispatch/scripts/launch_tp.sh <TP> <script> [args]
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." &>/dev/null && pwd)
TP=${1:?TP}; shift
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export LD_LIBRARY_PATH="${REPO}/build/lib:${REPO}/python/flux/lib:${CONDA_PREFIX}/lib:${CONDA_PREFIX}/lib64:${LD_LIBRARY_PATH:-}"
else
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/usr/local/lib:${HOME}/.local/lib"
fi
export NVSHMEM_BOOTSTRAP=UID
export NVSHMEM_DISABLE_CUDA_VMM=1
export NVSHMEM_REMOTE_TRANSPORT=${NVSHMEM_REMOTE_TRANSPORT:-none}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export CUDA_MODULE_LOADING=LAZY
export BYTED_TORCH_BYTECCL=O0
export NCCL_IB_TIMEOUT=${NCCL_IB_TIMEOUT:=23}
export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:=3}
export NVSHMEM_IB_GID_INDEX=3
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((TP - 1)))
CMD="torchrun --node_rank=0 --nproc_per_node=${TP} --nnodes=1 --rdzv_endpoint=127.0.0.1:${MASTER_PORT:-23456} $@"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} ${CMD}"
cd "${REPO}" && ${CMD}
