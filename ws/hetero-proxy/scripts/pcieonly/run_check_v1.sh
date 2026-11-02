#!/bin/bash
# hetero-proxy PCIe-only check (2026-11-02): NCCL default vs P2P disabled (SHM via host) vs P2P+SHM disabled (NET loopback)
set -u
cd /home/rogerlee/flux
d=ws/hetero-proxy/results/pcieonly/check_v1
mkdir -p "$d"
RUN="pixi run --manifest-path pixi.toml python -m torch.distributed.run --nproc_per_node 8 ws/hetero-proxy/scripts/pcieonly/pcie_only_check_v1.py --out $d"
one() {  # tag, env...
  local tag=$1; shift
  python3 common/measure/exclusive_guard_v2.py --log "$d/guard_$tag.log" -- env NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH "$@" \
    $RUN --tag "$tag" > "$d/stdout_$tag.txt" 2>&1
  echo "[$tag] $(tail -1 $d/guard_$tag.log | grep -o 'CLEAN\|CONTAMINATED')"
  grep "^\[$tag\]" "$d/stdout_$tag.txt"
  grep -o "via [A-Za-z/]*" "$d/stdout_$tag.txt" | sort | uniq -c | head -5
}
case "${1:-all}" in
  all)
    one nvlink env
    one shm env NCCL_P2P_DISABLE=1
    one shm_ce env NCCL_P2P_DISABLE=1 NCCL_SHM_USE_CUDA_MEMCPY=1
    one net env NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 ;;
  noib)  # 2026-11-02 rerun: with P2P off NCCL picked NET/IB for 16 of 40 connections and RoCE loopback timed out
         # (ibv_modify_qp errno 110; rails are separate subnets) -> also disable IB so every link is SHM via host
    one shm_noib env NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
    one shmce_noib env NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_SHM_USE_CUDA_MEMCPY=1 ;;
  intra)  # 2026-11-02 rerun 2: noib still sent 16 of 40 links over NET/Socket (TCP) -> forbid the network inside the node
    one shm_intra env NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_NET_DISABLE_INTRA=1
    one shmce_intra env NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_NET_DISABLE_INTRA=1 NCCL_SHM_USE_CUDA_MEMCPY=1 ;;
esac
