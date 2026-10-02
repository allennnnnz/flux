#!/usr/bin/env bash
# Reproduce the currently-valid Phase 0 measurement set.
#
# The previous version of this script reproduced the WITHDRAWN set: it drove
# dual_path_bench.py (superseded twice), used bandwidthTest with a single 32 MiB
# size, and ran the Flux benchmarks in fp16 under the default ring mode. See
# STATUS.md section 3 for what was withdrawn and why.
#
# Everything below corresponds to a number that still stands. Each block names
# the document that owns its results.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT_DIR}"

EXP="experiments/2026-09-23-phase0-a100-nvlink-pcie"
S="${EXP}/scripts"
R="${EXP}/results"
PIXI="pixi run --manifest-path pixi.toml"

mkdir -p "${R}"/{b_redo,bidirectional,c_staging,d_dual_path,e_flux}

echo "===================================================================="
echo "== Environment                                      (STATUS.md 1) =="
echo "===================================================================="
${PIXI} python - <<'PY'
import torch, flux
import nvidia.nvshmem  # noqa: F401
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("devices", torch.cuda.device_count())
print("peer access complete:", all(
    torch.cuda.can_device_access_peer(i, j)
    for i in range(torch.cuda.device_count())
    for j in range(torch.cuda.device_count()) if i != j))
PY
nvidia-smi topo -m

echo
echo "===================================================================="
echo "== A/B  PCIe link state and bandwidth              (B_REDO.md B) =="
echo "===================================================================="
for bdf in 0000:09:02.0 0000:42:02.0 0000:83:02.0 0000:bf:02.0 \
           0000:0a:00.0 0000:43:00.0 0000:84:00.0 0000:c0:00.0; do
  printf "%s  %s  x%s\n" "${bdf}" \
    "$(cat /sys/bus/pci/devices/${bdf}/current_link_speed 2>/dev/null || echo '?')" \
    "$(cat /sys/bus/pci/devices/${bdf}/current_link_width 2>/dev/null || echo '?')"
done

# single GPU, 8-GPU local/remote NUMA, and the shared-uplink control experiment
for dir in htod dtoh; do
  ${PIXI} python "${S}/global_bandwidth_v2.py" --direction "${dir}" \
    --devices 0 --numa-mode local --size-mib 1024 --copies 20 \
    --warmup 5 --repeats 20 > "${R}/b_redo/${dir}_local_gpu0.csv"
  for devs in "0,1:same_switch" "0,2:diff_switch_same_numa" \
              "0,2,4,6:four_switches" "0,1,2,3,4,5,6,7:all8"; do
    ${PIXI} python "${S}/global_bandwidth_v2.py" --direction "${dir}" \
      --devices "${devs%%:*}" --numa-mode local --size-mib 1024 --copies 20 \
      --warmup 5 --repeats 20 > "${R}/b_redo/${dir}_local_${devs##*:}.csv"
  done
  ${PIXI} python "${S}/global_bandwidth_v2.py" --direction "${dir}" \
    --devices 0,1,2,3,4,5,6,7 --numa-mode remote --size-mib 1024 --copies 20 \
    --warmup 5 --repeats 20 > "${R}/b_redo/${dir}_remote_all8.csv"
done

echo
echo "===================================================================="
echo "== Bidirectional, timed per direction     (BIDIR_CORRECTION.md X) =="
echo "===================================================================="
# NOTE: bidirectional_bandwidth.py (v1) is WITHDRAWN -- it emitted the same
# number for both directions. Use v2.
for cfg in "0:gpu0" "0,1:same_switch" "0,2:diff_switch" "0,1,2,3,4,5,6,7:all8"; do
  ${PIXI} python "${S}/bidirectional_bandwidth_v2.py" \
    --devices "${cfg%%:*}" --size-mib 256 --copies 20 --warmup 5 --repeats 20 \
    --numa-mode local --out "${R}/bidirectional/bidir_v2_local_${cfg##*:}.csv"
done

echo
echo "===================================================================="
echo "== C  Host staging, by topology         (BIDIR_CORRECTION.md X.5) =="
echo "===================================================================="
STAGE_BIN="${STAGE_BIN:-${TMPDIR:-/tmp}/staging_bench}"
if [[ ! -x "${STAGE_BIN}" ]]; then
  PATH=/usr/local/cuda/bin:${PATH} nvcc -O3 -std=c++17 -arch=sm_80 \
    -o "${STAGE_BIN}" "${S}/staging_stream_bench.cu" -lcuda -lnuma
fi
# NOTE: this binary takes "--devices 0,1", NOT "--devices=0,1".
for cfg in "0,1:same_switch" "0,2:diff_switch" "3,4:cross_numa" "0,4:cross_numa_alt"; do
  for chunk in 4 8 16 32; do
    "${STAGE_BIN}" --devices "${cfg%%:*}" --size-mib 1024 --chunk-mib "${chunk}" \
      --warmup 5 --repeats 20 \
      > "${R}/c_staging/staging_v2_${cfg##*:}_chunk${chunk}m.csv"
  done
done
"${STAGE_BIN}" --devices 0,1,2,3,4,5,6,7 --ring --size-mib 1024 --chunk-mib 16 \
  --warmup 5 --repeats 20 > "${R}/c_staging/staging_ring8_chunk16m.csv"

echo
echo "===================================================================="
echo "== D  Dual path, corrected NVLink path    (D_CORRECTION.md D.1-6) =="
echo "===================================================================="
# NOTE: dual_path_bench.py and dual_path_v2.py are WITHDRAWN. v2 placed peer
# copies on the DESTINATION device's stream; PyTorch runs them on the SOURCE
# device's stream, so each GPU's 7 outgoing copies serialized (38 vs 218 GB/s).
${PIXI} python "${S}/dual_path_v3.py" \
  --alphas 0,0.01,0.02,0.03,0.05,0.08,0.1 --size-mib 1024 --copies 20 \
  --warmup 5 --repeats 20 --out "${R}/d_dual_path/dual_path_v3_all8.csv"
${PIXI} python "${S}/dual_path_v3.py" \
  --alphas 0,0.02,0.05,0.08,0.12 --size-mib 1024 --copies 20 \
  --warmup 5 --repeats 15 --stage-ring 0,2,4,6,1,3,5,7 \
  --out "${R}/d_dual_path/dual_path_v3_switchaware_ring.csv"

echo
echo "===================================================================="
echo "== E  Flux-native comm and AG+GEMM     (E_CORRECTION.md, FLUX_*)  =="
echo "===================================================================="
# Flux's own AllGather in isolation. NOTE: nccl_collective_baseline.py is an
# NCCL reference only -- it is NOT a Flux baseline (E_CORRECTION.md E.0).
for M in 1024 2048 4096 8192 16384; do
  ${PIXI} ./launch.sh "${S}/flux_comm_baseline.py" "${M}" 12288 \
    --dtype=float16 --warmup=5 --iters=30 \
    --out="${R}/e_flux/flux_ag_comm_M${M}.csv"
done

# AG+GEMM with the interleaved per-round ECT method. NOTE: the old
# total-minus-gemm_only-across-separate-loops method is WITHDRAWN; it produced
# negative ECT values.
for N in 4096 8192 49152; do
  for RM in all2all ring2d; do
    ${PIXI} ./launch.sh "${S}/flux_ag_gemm_baseline_v2.py" 4096 "${N}" 12288 \
      --dtype=bfloat16 --ring_mode="${RM}" --rounds=200 --warmup=20 \
      --out="${R}/e_flux/ag_gemm_v2_N${N}_${RM}.csv"
  done
done

echo
echo "Done. Results under ${R}. Read docs/PHASE0_STATUS.md for interpretation."
