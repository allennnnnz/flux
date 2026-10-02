# Phase 0 C/D/E Follow-up

Date: 2026-09-23

Adopted B numbers:

- 8-GPU concurrent local NUMA: H2D 12.647 GB/s/GPU, D2H 13.195 GB/s/GPU.
- Single GPU: H2D 21.795 GB/s, D2H 23.924 GB/s.
- PCIe switch uplink: about 25 GB/s, shared by GPU pairs 0/1, 2/3, 4/5, 6/7.

## Added Bidirectional PCIe Measurements

Script: `scripts/bidirectional_bandwidth.py`.

Parameters: 1024 MiB per GPU per direction, 20 copies per measured interval,
5 warmup intervals discarded, 20 formal repeats, local NUMA binding.

| Case | H2D GB/s | D2H GB/s | Per-GPU H2D GB/s | Per-GPU D2H GB/s |
| --- | ---: | ---: | ---: | ---: |
| GPU0 | 12.689 | 12.689 | 12.689 | 12.689 |
| GPU0+GPU1 same switch | 14.683 | 14.683 | 7.341 | 7.341 |
| 8 GPUs | 57.354 | 57.354 | 7.169 | 7.169 |

Cross-NUMA pair script: `scripts/cross_numa_pair.py`.

Case: GPU3 D2H to one pinned host buffer and GPU4 H2D from that buffer, other GPUs idle,
1024 MiB x 20 copies, 5 warmup, 20 repeats.

| Buffer NUMA node | D2H GB/s | H2D GB/s |
| --- | ---: | ---: |
| NUMA 0 | 18.585 | 18.585 |
| NUMA 1 | 19.128 | 19.128 |

## C. Host Staging

Script: `scripts/staging_stream_bench.cu`.

Implementation details:

- Source stream copies D2H into a pinned host chunk.
- Source stream then calls `cuStreamWriteValue32` on a monotonically increasing counter.
- Destination stream calls `cuStreamWaitValue32` before H2D from that chunk.
- Two host chunks are used as a double buffer.
- Host buffers are allocated with `numa_alloc_onnode()` on the source GPU's NUMA node,
  then registered with `cudaHostRegisterMapped`.

Parameters: 1024 MiB per pair, 5 warmup iterations, 20 formal repeats.

### Single Pair, GPU0 -> GPU1

| Chunk MiB | Staging GB/s |
| ---: | ---: |
| 1 | 13.656 |
| 2 | 13.306 |
| 4 | 13.642 |
| 8 | 13.644 |
| 16 | 13.815 |
| 32 | 13.279 |
| 64 | 13.183 |

Best single-pair staging: 13.815 GB/s at 16 MiB chunks.

Single-pair bidirectional PCIe baseline for GPU0: 12.689 GB/s per direction.
Staging efficiency versus that bidirectional per-direction baseline: 13.815 / 12.689 = 108.9%.

### 8-GPU Ring Staging

Ring: 0->1, 1->2, 2->3, 3->4, 4->5, 5->6, 6->7, 7->0.

| Chunk MiB | Aggregate GB/s | Per Pair GB/s |
| ---: | ---: | ---: |
| 1 | 18.727 | 2.341 |
| 2 | 44.034 | 5.504 |
| 4 | 46.276 | 5.785 |
| 8 | 46.728 | 5.841 |
| 16 | 46.988 | 5.874 |
| 32 | 46.306 | 5.788 |
| 64 | 46.062 | 5.758 |

Best 8-GPU staging: 46.988 GB/s aggregate at 16 MiB chunks.

8-GPU bidirectional PCIe baseline: 57.354 GB/s per direction aggregate.
Staging efficiency versus that bidirectional per-direction baseline: 46.988 / 57.354 = 81.9%.

## D. Dual Path

> **WITHDRAWN 2026-09-23 — see `D_CORRECTION.md`.**
> The α=0 NVLink baseline below is 38.4 GB/s per GPU, about 5.7x slower than this machine's
> actual capability (217.66 GB/s per GPU corrected; 270.8 GB/s for a single peer pair). The
> cause was destination-stream placement of peer copies. Every gain figure here divides by
> that baseline and is an artifact. Corrected: every α is a net loss. Kept verbatim for audit.


Script: `scripts/dual_path_v2.py`.

NVLink path: each GPU receives from 7 peer GPUs, so each GPU has 7 NVLink peer ingress streams.
Host staging path: ring staging. Parameters: 1024 MiB per GPU, 20 copies, 5 warmup,
20 formal repeats.

| Alpha | Median s | Payload GB/s | NVLink payload GB/s | Staging payload GB/s |
| ---: | ---: | ---: | ---: | ---: |
| 0.0 | 0.559745 | 306.923 | 306.923 | 0.000 |
| 0.1 | 0.521890 | 329.185 | 296.267 | 32.919 |
| 0.2 | 0.900883 | 190.700 | 152.560 | 38.140 |
| 0.3 | 1.327726 | 129.393 | 90.575 | 38.818 |
| 0.4 | 1.751846 | 98.067 | 58.840 | 39.227 |
| 0.5 | 2.176276 | 78.942 | 39.471 | 39.471 |
| 1.0 | 4.253306 | 40.392 | 0.000 | 40.392 |

Model:

`t(alpha) = max((1-alpha)/B_nvlink, alpha/B_staging)`

Using measured `B_nvlink = 306.923 GB/s` from alpha 0 and best C staging
`B_staging = 46.988 GB/s`, the model predicts:

- Best alpha: `46.988 / (306.923 + 46.988) = 0.133`.
- Predicted best throughput: `306.923 + 46.988 = 353.911 GB/s`.
- Predicted gain over pure NVLink: 15.3%.

Measured:

- Best tested alpha: 0.1.
- Best measured throughput: 329.185 GB/s.
- Measured gain over pure NVLink: 7.3%.

Interpretation: adding a small host-staging fraction helps, but the measured optimum is lower
than the ideal model. Larger alpha values rapidly become staging-bound.

## E. Flux Baselines

> **WITHDRAWN 2026-09-23 — see `E_CORRECTION.md`.**
> Every overlap-efficiency figure below divides a Flux time by an NCCL time. Worse, the
> row labelled `Flux no-overlap ... comm` is itself NCCL `all_gather`, not Flux comm, so
> no number in this section ever measured Flux's own communication path. The section is
> kept verbatim for audit; use `E_CORRECTION.md` for the Flux-native measurements.


All Flux runs used the same shapes as the pixi tasks unless noted.

### AG GEMM

Command output: `results/e_flux/ag_gemm_ring2d.txt`.

Shape: `M=4096, N=49152, K=12288`, dtype float16, ring2d, warmup 5, iters 20.

Rank 0:

- Torch baseline: total 3.307 ms, GEMM 2.749 ms, NCCL AG comm 0.558 ms.
- Flux AG+GEMM: total 2.894 ms, GEMM 2.457 ms, exposed comm 0.437 ms, gemm_only 2.641 ms.
- Flux no-overlap: total 3.401 ms, GEMM 2.495 ms, comm 0.906 ms.
- AG time: 2.894 ms.
- ECT from Flux overlapped run, `total - gemm_only`: 0.253 ms.
- Overlap efficiency using no-overlap comm, `1 - ECT / 0.906`: 72.1%.
- Correctness: torch vs Flux bitwise match and Flux allclose passed on all ranks.

### GEMM Only

Command output: `results/e_flux/gemm_only_single_gpu.txt`.

Shape: `M=4096, N=49152, K=12288`, dtype float16, single GPU, iters 20.

- Torch GEMM: 20.351 ms.
- Flux GEMM-only: 20.918 ms.
- Analytical SOL line: TensorCore 15.858 ms.

This single-GPU GEMM-only shape is not directly comparable to the AG rank-local GEMM timing,
because AG uses local `M/world_size` and local `N/world_size` shards.

### NCCL Pure Communication

> **Mislabelled.** These are valid NCCL numbers but they are not a Flux baseline, and they
> must not be used as the denominator of a Flux overlap efficiency. Flux's own AllGather is
> measured in `E_CORRECTION.md` section E.1 via `flux.AllGatherOp`.


Script: `scripts/nccl_collective_baseline.py`.

AG shape matches AG input shard: `M=4096, K=12288`, dtype float16, world 8.

- NCCL all_gather median: 0.598528 ms.
- Stdev: 0.033505 ms.
- Bytes per rank input: 12,582,912 bytes.

RS shape matches GEMM_RS full output: `M=4096, N=12288`, dtype float16, world 8.

- NCCL reduce_scatter median: 0.666624 ms.
- Stdev: 0.036774 ms.
- Bytes per rank input: 100,663,296 bytes.

The repo's `test_comm_ag.py` and `test_comm_rs.py` pure-comm scripts were attempted but not used:

- `test_comm_ag.py --exp=ring2d` failed because `_run()` does not accept the `iter` keyword
  passed by `run_perf`.
- `test_comm_ag.py --exp=ring1d` failed its correctness assertion.
- `test_comm_rs.py` requires `copy_utils`; after adding `PYTHONPATH=test/python/util`,
  it then required the missing `cuda` Python package.

The failed raw outputs are kept under `results/e_flux/comm_*.txt`.

### GEMM RS

Command output: `results/e_flux/gemm_rs_ring2d.txt`.

Shape: `M=4096, N=12288, K=49152`, dtype float16, ring2d, warmup 5, iters 20.

Rank 0:

- Torch baseline: GEMM 2.780 ms, comm 0.543 ms, total 3.323 ms.
- Flux GEMM_RS: GEMM 2.800 ms, exposed comm 0.005 ms, total 2.806 ms.
- Correctness: Flux allclose passed on all ranks.
- Bitwise: Flux vs torch bitwise check failed on rank 0.

Using the torch comm baseline 0.543 ms and Flux exposed comm 0.005 ms:

- ECT: 0.005 ms.
- Overlap efficiency: `1 - 0.005 / 0.543 = 99.1%`.
