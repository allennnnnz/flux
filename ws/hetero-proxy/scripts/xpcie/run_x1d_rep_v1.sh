#!/bin/bash
# hetero-proxy X1d replication (2026-10-28): x1d_v1 showed hostread under GPU0's own D2H at 5.0 us (stable in all 100
# rounds) while x1b_v1 / x1d_smoke had 1.7 us and x1d_smoke2 2.0 us -> per-process state? Four more processes x 20
# rounds of x1d_scope_v1.py, and the NCCL_GRAPH_MIXING_SUPPORT=1 / 0 comparison repeated in fresh processes (30 rounds).
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
for i in 1 2 3 4; do
  run x1d_rep$i env CUDA_MODULE_LOADING=EAGER $PIXI $PY $S/x1d_scope_v1.py --rounds 20
done
for m in 1 0; do
  run x1_mix${m}_rep env -u CUDA_DEVICE_MAX_CONNECTIONS NCCL_GRAPH_MIXING_SUPPORT=$m CUDA_MODULE_LOADING=EAGER $PIXI $PY \
    -m torch.distributed.run --nproc_per_node 8 $S/x1_dispatch_v1.py --rounds 30
done
echo ALLDONE
