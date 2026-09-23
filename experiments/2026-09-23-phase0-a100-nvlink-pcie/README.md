# Phase 0: A100 single-node NVLink/PCIe baseline

Date: 2026-09-23

## Purpose

This experiment establishes the single-node baseline before changing Flux transport or scheduling code.

The questions were:

- Is the current machine suitable for a single-node Flux experiment?
- Does the 8x A100 host expose enough topology diversity to test a dual NVLink + PCIe transport idea?
- Can the current pixi environment build and run Flux benchmarks without sudo?
- Is there measurable benefit from intentionally moving part of GPU-GPU traffic through host pinned memory while the main path uses GPU peer copies?

## Environment

- GPU: 8x NVIDIA A100-SXM4-80GB
- Driver: 615.71.09
- CUDA user-mode stack observed by `nvidia-smi`: 13.4
- Python environment: pixi
- PyTorch inside pixi: 2.6.0+cu124
- CUDA visible to PyTorch: 12.4
- `nvidia.nvshmem`: import succeeds inside pixi
- sudo: unavailable

## Topology Findings

`nvidia-smi topo -m` reports all GPU-GPU pairs as `NV12`.

Interpretation:

- The 8 GPUs are connected through NVSwitch/NVLink.
- This is not a topology where only fixed GPU pairs share a direct NVLink.
- A second identical node is not required for Phase 0 single-node baselines.

NUMA placement:

| Devices | NUMA node | CPU affinity |
| --- | --- | --- |
| GPU0-GPU3 | 0 | 0-31,64-95 |
| GPU4-GPU7 | 1 | 32-63,96-127 |

`nvidia-smi topo -p2p r` reports `OK` for all GPU pairs.

## Bandwidth Results

CUDA `bandwidthTest` was used because `nvbandwidth` was not installed.

| Test | Result |
| --- | ---: |
| 8 GPUs simultaneous pinned H2D, 32 MiB each | 162.5 GB/s aggregate |
| 8 GPUs simultaneous pinned D2H, 32 MiB each | 172.8 GB/s aggregate |
| 8 GPUs device-to-device | 11.23 TB/s aggregate |
| Per-GPU H2D, local NUMA bind | 21.6-22.6 GB/s |
| Per-GPU D2H, local NUMA bind | 23.1-23.6 GB/s |

The device-to-device number is much larger than PCIe host staging bandwidth, as expected on an A100 SXM NVSwitch system.

## Dual Path Microbenchmark

The custom benchmark splits each source GPU payload into two parts:

- main path: GPU peer copy, representing NVLink/NVSwitch traffic
- auxiliary path: D2H into pinned CPU memory, then H2D to destination GPU, representing PCIe host staging

Payload: 64 MiB per GPU, 8 GPUs, 5 timed repeats.

| Host memory policy | Best alpha | Best payload GB/s | NVLink-only alpha=0 | PCIe-only alpha=1 | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| default allocation | 0.02 | 254.8 | 245.8 | 42.4 | about +3.6% in this run |
| NUMA node 0 bind | 0.02 | 255.4 | 245.3 | 32.1 | host staging degraded for remote GPUs |
| NUMA node 1 bind | 0.08 | 250.3 | 245.8 | 42.9 | small gain, not stable |

Conclusion:

- A small PCIe-staged fraction can sometimes improve this synthetic copy workload by roughly 2-4%.
- The effect is narrow and NUMA-sensitive.
- On this A100 NVSwitch machine, PCIe is much weaker than the GPU-GPU path, so this is not strong enough evidence to justify changing Flux core kernels directly.
- The result is still useful as a proxy for future PCIe-only or heterogeneous accelerator work, especially for designing backend abstraction and fallback transport logic.

## Flux Benchmark Results

All runs used:

```bash
NVSHMEM_REMOTE_TRANSPORT=none pixi run --manifest-path pixi.toml ./launch.sh ...
```

### AllGather + GEMM

Command template:

```bash
NVSHMEM_REMOTE_TRANSPORT=none pixi run --manifest-path pixi.toml \
  ./launch.sh test/python/ag_gemm/test_ag_kernel.py M 49152 12288 \
  --dtype=float16 --warmup=2 --iters=5 --verify
```

Functional result:

- All tested sizes printed `all close!` and `flux check passed`.
- Bitwise match was observed for M=64, 4096, 8192.
- M=512, 1024, 2048 were all-close but not bitwise identical.

Timing caveat:

- In this Phase 0 run, Flux AG timing lines showed `total 0.000 ms`, so those Flux timing values are invalid and should not be used.
- A previous successful AG run on this environment showed M=4096, N=49152, K=12288, fp16: Flux about 2.889 ms vs PyTorch about 3.300 ms, roughly 1.14x.

Representative PyTorch baseline from the Phase 0 AG runs:

| M | PyTorch total ms | GEMM ms | Comm ms |
| ---: | ---: | ---: | ---: |
| 64 | 0.305 | 0.115 | 0.190 |
| 512 | 0.500 | 0.376 | 0.120 |
| 1024 | 1.020 | 0.816 | 0.205 |
| 2048 | 1.670 | 1.360 | 0.310 |
| 4096 | 3.300 | 2.750 | 0.550 |
| 8192 | 6.460 | 5.400 | 1.060 |

### GEMM + ReduceScatter

Command template:

```bash
NVSHMEM_REMOTE_TRANSPORT=none pixi run --manifest-path pixi.toml \
  ./launch.sh test/python/gemm_rs/test_gemm_rs.py M 12288 49152 \
  --dtype=float16 --warmup=2 --iters=5
```

Functional result:

- All tested sizes printed `all close!` and `flux check passed`.
- The script reported bitwise mismatch for all tested sizes, likely due to reduction ordering. Treat this as numerically correct but not bitwise identical.

Representative timings:

| M | PyTorch ms | Flux ms | Flux speedup |
| ---: | ---: | ---: | ---: |
| 64 | 0.228 | 0.249 | 0.92x |
| 512 | 0.514 | 0.436 | 1.18x |
| 1024 | 1.006 | 0.783 | 1.29x |
| 2048 | 1.667 | 1.458 | 1.14x |
| 4096 | 3.306 | 2.807 | 1.18x |
| 8192 | 6.276 | 5.511 | 1.14x |

## Reproduction

From the repository root:

```bash
pixi install
pixi run --manifest-path pixi.toml python - <<'PY'
import torch
import nvidia.nvshmem
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
PY

bash experiments/2026-09-23-phase0-a100-nvlink-pcie/scripts/run_phase0_repro.sh
```

The reproduction script reruns topology checks, CUDA bandwidth tests when `bandwidthTest` is available, the dual-path microbenchmark, and the Flux AG/RS benchmark commands used above.

## Files

- `scripts/dual_path_bench.py`: custom dual-path NVLink + pinned-host staging microbenchmark.
- `scripts/run_phase0_repro.sh`: commands needed to reproduce this Phase 0 measurement set.

## Decision

Do not proceed directly to Flux core dual-channel scheduling on this machine based only on Phase 0. The observed PCIe auxiliary-path gain is small, unstable, and heavily NUMA-dependent.

The next useful step is to build a transport/backend abstraction and use this A100 node as a proxy for future PCIe-only or heterogeneous devices, while keeping current Flux NVLink behavior intact.
