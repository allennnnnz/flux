#!/bin/bash
################################################################################
# fusion-dispatch E4 (cross-node, css-host-158 + css-host-159): start ONE node's share of a 2-node torchrun,
# wrapped in common/measure/exclusive_guard_v2.py. Run by run_xnode.sh on both nodes (do not call by hand).
# Usage: launch_xnode.sh <node_rank> <nproc_per_node> <env: pixi|vllm> <guard_log> <script.py> [args...]
# Environment facts (results/node159_inventory/, 2026-10-03):
#   - 8 RoCE rails (100 GbE, subnets 10.10.<rail>.0/24, GID index 3 = RoCE v2 IPv4); rail 4 is down
#     (css-host-158 mlx5_3 has no IPv4 -> empty GID 3), and css-host-159 has SR-IOV VFs mlx5_8..15 without
#     addresses -> NCCL / NVSHMEM are restricted to the 7 working PFs.
#   - bootstrap / rendezvous over enp29s0f0np0 (same name on both nodes; 10.10.1.158 / 10.10.1.159).
#   - css-host-159 ~/.bashrc puts a self-built NCCL 2.26.2 in LD_LIBRARY_PATH -> LD_LIBRARY_PATH is rebuilt
#     here from scratch (torch loads its bundled 2.21.5 either way; checked 2026-10-03).
#   - css-host-159 has ~/.nccl.conf (NCCL_ALGO=RING, NCCL_PROTO=Simple, NCCL_P2P_LEVEL=NVL, NCCL_IB_HCA=mlx5_3:1; not ours,
#     left untouched). NCCL reads it automatically on 159 only -> the two nodes pick different protocols and the first
#     cross-node collective hangs silently (found 2026-10-03). NCCL_CONF_FILE=/dev/null disables it on both nodes.
#   - every run is bounded by XNODE_TIMEOUT seconds (default 1800) so a node that never joins cannot hang the other.
#   - GPUs: nproc_per_node 8 -> all; 4 -> 0,1,4,5 (one NIC each: mlx5_0,1,4,5); 1 -> 0. XNODE_GPUS overrides.
################################################################################
NODE_RANK=$1; NPROC=$2; ENVK=$3; GLOG=$4; shift 4
FLUX=/home/rogerlee/flux
VENV=/home/rogerlee/venvs/vllm085-flux
HCAS="mlx5_0,mlx5_1,mlx5_2,mlx5_4,mlx5_5,mlx5_6,mlx5_7"
case "${XNODE_GPUS:-}" in
  "") case $NPROC in 8) GPUS=0,1,2,3,4,5,6,7;; 4) GPUS=0,1,4,5;; 2) GPUS=0,4;; 1) GPUS=0;; *) GPUS=$(seq -s, 0 $((NPROC - 1)));; esac;;
  *) GPUS=$XNODE_GPUS;;
esac
export CUDA_VISIBLE_DEVICES=$GPUS
export LD_LIBRARY_PATH="${FLUX}/build/lib:${FLUX}/python/flux/lib:${FLUX}/.pixi/envs/default/lib"
export NCCL_CONF_FILE=/dev/null   # css-host-159 has ~/.nccl.conf (ALGO=RING, PROTO=Simple, IB_HCA=mlx5_3) -> ranks disagree, silent hang
export NCCL_IB_HCA="=${HCAS}"
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TIMEOUT=23
export NCCL_SOCKET_IFNAME=enp29s0f0np0
export GLOO_SOCKET_IFNAME=enp29s0f0np0
export NVSHMEM_BOOTSTRAP=UID
export NVSHMEM_DISABLE_CUDA_VMM=1
export NVSHMEM_REMOTE_TRANSPORT=${NVSHMEM_REMOTE_TRANSPORT:-ibrc}
export NVSHMEM_IB_GID_INDEX=3
export NVSHMEM_HCA_LIST="${HCAS}"
export NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=enp29s0f0np0
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export CUDA_MODULE_LOADING=LAZY
export BYTED_TORCH_BYTECCL=O0
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-WARNING}
case $ENVK in
  pixi) PY=${FLUX}/.pixi/envs/default/bin/python; export CONDA_PREFIX=${FLUX}/.pixi/envs/default;;
  vllm) PY=${VENV}/bin/python;;
  *) echo "env must be pixi or vllm" >&2; exit 2;;
esac
cd ${FLUX}
exec python3 common/measure/exclusive_guard_v2.py --log "${GLOG}" -- \
  timeout --signal=TERM --kill-after=30 ${XNODE_TIMEOUT:-1800} ${PY} -m torch.distributed.run --nnodes=2 --node_rank=${NODE_RANK} --nproc_per_node=${NPROC} \
  --master_addr=10.10.1.158 --master_port=${MASTER_PORT:-29531} "$@"
