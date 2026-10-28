#!/bin/bash
# hetero-proxy X1d + D8 (2026-10-28): every run under the exclusive guard; start from anywhere inside tmux.
#   x1d_v1            x1d_scope_v1.py, 100 rounds (fence scope, shared switch, NVLink copies, LL data path, kernel end,
#                     event nodes)
#   x1e_nomix         graph dump with NCCL_GRAPH_MIXING_SUPPORT=0 (are the event nodes gone?)
#   x1_nomix          x1_dispatch_v1.py with NCCL_GRAPH_MIXING_SUPPORT=0, default protocol (LL), 100 rounds (vs x1_v1)
#   x1_nomix_{LL128,Simple}  same with NCCL_PROTO forced, 30 rounds (vs x1_proto_*)
set -u
cd /home/rogerlee/flux
PY=/home/rogerlee/.venvs/hetero-proxy-hf/bin/python
R=ws/hetero-proxy/results/xpcie
S=ws/hetero-proxy/scripts/xpcie
PIXI="pixi run --manifest-path pixi.toml"
run() {
  local d=$R/$1; shift
  mkdir -p "$d"
  echo "[run] $d $(date -u +%H:%M:%S)"
  python3 common/measure/exclusive_guard_v2.py --log "$d/guard.log" -- "$@" --out "$d" > "$d/stdout.txt" 2>&1
  tail -1 "$d/guard.log"
}
run x1d_v1 env CUDA_MODULE_LOADING=EAGER $PIXI $PY $S/x1d_scope_v1.py --rounds 100
NOMIX="env -u CUDA_DEVICE_MAX_CONNECTIONS NCCL_GRAPH_MIXING_SUPPORT=0 CUDA_MODULE_LOADING=EAGER"
run x1e_nomix $NOMIX NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,TUNING $PIXI $PY -m torch.distributed.run --nproc_per_node 8 \
  $S/x1e_graphdump_v1.py
run x1_nomix $NOMIX $PIXI $PY -m torch.distributed.run --nproc_per_node 8 $S/x1_dispatch_v1.py --rounds 100
for p in LL128 Simple; do
  run x1_nomix_$p $NOMIX NCCL_PROTO=$p $PIXI $PY -m torch.distributed.run --nproc_per_node 8 $S/x1_dispatch_v1.py --rounds 30
done
echo ALLDONE
