#!/bin/bash
################################################################################
# fusion-dispatch E4: run one 2-node job (css-host-158 = node 0 / master, css-host-159 = node 1), each node
# wrapped in exclusive_guard_v2. Run from the repo root on css-host-158 (long jobs inside tmux).
# Usage: bash ws/fusion-dispatch/scripts/run_xnode.sh <out_dir> <nproc_per_node> <pixi|vllm> <script.py> [args...]
#   - syncs the repo working tree to css-host-159 first (rsync of tracked + untracked source files only; the
#     environments were copied once on 2026-10-03), so both nodes run the same code;
#   - guard logs: <out_dir>/guard_node0.log, guard_node1.log (node 1 log copied back); the run is CLEAN
#     only if both are CLEAN; per-node stdout/stderr: <out_dir>/node{0,1}.out.
#   - XNODE_ENV="K=V ..." is exported on both nodes (e.g. NCCL_DEBUG=INFO); XNODE_GPUS / XNODE_TIMEOUT / MASTER_PORT too.
################################################################################
set -u
OUT=$1; NPROC=$2; ENVK=$3; shift 3
REMOTE=rogerlee@10.2.131.159
SSHO="-o BatchMode=yes"
REPO=/home/rogerlee/flux
mkdir -p "$OUT"
ABS_OUT=$(cd "$OUT" && pwd)
rsync -a -e "ssh $SSHO" --exclude .pixi --exclude build --exclude 'python/flux/lib' --exclude '__pycache__' \
  ${REPO}/ws ${REPO}/common ${REPO}/launch.sh ${REMOTE}:${REPO}/ || { echo "rsync failed" >&2; exit 2; }
ssh $SSHO ${REMOTE} "mkdir -p ${ABS_OUT}"
# Arguments go to node 1 through a remote shell: quote each one (E4a2, 2026-10-05: an unquoted "--cases a;b"
# was split at ';' on css-host-159, node 1 exited, node 0 waited until the timeout).
ARGQ=$(printf '%q ' "$@")
echo "[run_xnode] $(date '+%F %T') start: nproc_per_node=${NPROC} env=${ENVK} $*" | tee -a "${ABS_OUT}/run_xnode.log"
ssh $SSHO ${REMOTE} "env ${XNODE_ENV:-} XNODE_GPUS='${XNODE_GPUS:-}' XNODE_TIMEOUT='${XNODE_TIMEOUT:-1800}' MASTER_PORT='${MASTER_PORT:-29531}' NVSHMEM_REMOTE_TRANSPORT='${NVSHMEM_REMOTE_TRANSPORT:-ibrc}' \
  bash ${REPO}/ws/fusion-dispatch/scripts/launch_xnode.sh 1 ${NPROC} ${ENVK} ${ABS_OUT}/guard_node1.log ${ARGQ}" \
  > "${ABS_OUT}/node1.out" 2>&1 &
RPID=$!
env ${XNODE_ENV:-} bash ${REPO}/ws/fusion-dispatch/scripts/launch_xnode.sh 0 ${NPROC} ${ENVK} ${ABS_OUT}/guard_node0.log "$@" \
  > "${ABS_OUT}/node0.out" 2>&1
RC0=$?
wait $RPID
RC1=$?
rsync -a -e "ssh $SSHO" ${REMOTE}:${ABS_OUT}/ "${ABS_OUT}/" --exclude node0.out --exclude guard_node0.log 2>/dev/null
V0=$(grep -o "END rc=[0-9-]* [A-Z]*" "${ABS_OUT}/guard_node0.log" | tail -1)
V1=$(grep -o "END rc=[0-9-]* [A-Z]*" "${ABS_OUT}/guard_node1.log" | tail -1)
echo "[run_xnode] $(date '+%F %T') done: node0 rc=${RC0} (${V0}); node1 rc=${RC1} (${V1})" | tee -a "${ABS_OUT}/run_xnode.log"
[ $RC0 -eq 0 ] && [ $RC1 -eq 0 ]
