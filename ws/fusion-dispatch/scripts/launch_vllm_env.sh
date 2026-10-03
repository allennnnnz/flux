#!/bin/bash
# Same environment as ./launch.sh, but runs torchrun from the isolated vLLM 0.8.5.post1 venv
# (/home/rogerlee/venvs/vllm085-flux: vLLM + PyPI torch 2.6.0+cu124, --system-site-packages for Flux).
# Usage (repo root): bash ws/fusion-dispatch/scripts/launch_vllm_env.sh <script.py> [args]
# TP=<n> (optional, G4): run on the first n GPUs (default: all GPUs, unchanged behaviour).
FLUX_SRC_DIR=/home/rogerlee/flux
VENV=/home/rogerlee/venvs/vllm085-flux
export LD_LIBRARY_PATH="${FLUX_SRC_DIR}/build/lib:${FLUX_SRC_DIR}/python/flux/lib:${FLUX_SRC_DIR}/.pixi/envs/default/lib:${LD_LIBRARY_PATH:-}"
export NVSHMEM_BOOTSTRAP=UID
export NVSHMEM_DISABLE_CUDA_VMM=1
export NVSHMEM_REMOTE_TRANSPORT=${NVSHMEM_REMOTE_TRANSPORT:-none}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export CUDA_MODULE_LOADING=LAZY
export BYTED_TORCH_BYTECCL=O0
export NCCL_IB_TIMEOUT=${NCCL_IB_TIMEOUT:=23}
export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:=3}
export NVSHMEM_IB_GID_INDEX=3
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-WARNING}
NPROC=$(nvidia-smi --list-gpus | wc -l)
if [ -n "${TP:-}" ]; then NPROC=$TP; export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((TP - 1))); fi
exec ${VENV}/bin/python -m torch.distributed.run --node_rank=0 --nproc_per_node=$NPROC \
  --nnodes=1 --rdzv_endpoint=127.0.0.1:${MASTER_PORT:-23457} "$@"
