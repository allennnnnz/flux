# Phase 0 E Correction: Flux comm was never measured

Date: 2026-09-23

This supersedes the "NCCL Pure Communication" subsection and every overlap-efficiency
number in `CDE_REPORT.md` section E.

## E.0 The defect

`CDE_REPORT.md` presented NCCL `all_gather` / `reduce_scatter` as the pure-communication
baseline and then used NCCL timings as the denominator of Flux's overlap efficiency.
NCCL and Flux do not share a communication path, so those ratios compare unrelated
quantities.

The defect is larger than the label suggests. The row reported as
`Flux no-overlap: ... comm 0.906 ms` is also NCCL. In
[test_ag_kernel.py:245](../../test/python/ag_gemm/test_ag_kernel.py#L245),
`perf_flux_no_overlap` calls `all_gather_into_tensor_with_fp8`, which is a thin wrapper
over `torch.distributed.all_gather_into_tensor`
([utils.py:228](../../python/flux/testing/utils.py#L228)) — i.e. NCCL — and only the GEMM
is Flux's. The same applies to the `torch` baseline row.

**Consequence: no number anywhere in Phase 0 measured Flux's own communication.**
The 72.1% AG overlap efficiency and the 99.1% GEMM_RS overlap efficiency are both
`1 - (Flux ECT) / (an NCCL timing)` and are withdrawn.

Corroborating evidence that the 0.906 ms denominator was noise rather than signal: across
three ring modes the `flux(no-overlap)` comm value read 0.498, 0.645 and 0.704 ms, even
though that code path is identical NCCL `all_gather` in all three cases.

## E.1 Flux's own AllGather, measured

Flux exposes its all-gather comm path standalone as `flux.AllGatherOp`
([src/pybind/flux_coll_op.cc:37](../../src/pybind/flux_coll_op.cc#L37),
[src/coll/ths_op/all_gather_op.cc:356](../../src/coll/ths_op/all_gather_op.cc#L356)).
`AllGatherOp::run()` is self-contained — it performs the local shard copy, the barrier and
the ring/all2all transfer — and is the same code `AGKernel.forward()` drives internally.

Script: `scripts/flux_comm_baseline.py`. Options are set to match what
`AGKernel.forward()` uses by default for fp16: `input_buffer_copied=False` (so the local
copy is included), `use_cuda_core_local=False`, `fuse_sync=False`.

Parameters: world 8, K=12288, float16, 5 warmup, 20-30 timed iterations, a process-group
barrier before each timed iteration. Reported time is the **rank-max median**, since a
collective is finished only when its slowest rank is finished. Every configuration below
was verified bitwise-identical to NCCL `all_gather` on all 8 ranks before timing.

Full M=4096 (input shard 512x12288 per rank, 100,663,296 output bytes):

| Path | rank-max median ms | algbw GB/s | vs NCCL |
| --- | ---: | ---: | ---: |
| NCCL all_gather | 0.569 | 176.8 | 1.00x |
| Flux All2All, pull | **0.468** | **214.9** | **1.22x** |
| Flux All2All, push | 0.471 | 213.8 | 1.21x |
| Flux Ring1D, pull | 0.496 | 202.8 | 1.15x |
| Flux Ring2D, push | 0.562 | 179.1 | 1.01x |
| Flux Ring1D, push | 0.565 | 178.2 | 1.01x |
| Flux Ring2D, push, CUDA core | 1.066 | 94.4 | 0.53x |

Size sweep, rank-max median ms (raw CSVs in `results/e_flux/flux_ag_comm_M*.csv`):

| Path | M=1024 | M=2048 | M=4096 | M=8192 | M=16384 |
| --- | ---: | ---: | ---: | ---: | ---: |
| NCCL all_gather | 0.224 | 0.331 | 0.569 | 1.042 | 1.928 |
| Flux All2All, push | 0.197 | 0.287 | 0.471 | 0.853 | 1.635 |
| Flux All2All, pull | 0.199 | 0.283 | 0.468 | 0.851 | 1.668 |
| Flux Ring1D, push | 0.219 | 0.342 | 0.565 | 1.006 | 1.832 |
| Flux Ring1D, pull | 0.216 | 0.311 | 0.496 | 0.876 | 1.580 |
| Flux Ring2D, push | 0.232 | 0.340 | 0.562 | 0.988 | 1.810 |
| Flux Ring2D, push, CUDA core | 0.329 | 0.589 | 1.066 | 2.250 | 3.859 |

Verdicts:

- Flux's own AG comm beats NCCL by 13-22% in All2All mode at every size tested. This is a
  real result that the NCCL-as-baseline framing hid entirely.
- **Ring2D — the mode all Phase 0 E runs used — is one of the slowest modes on this
  machine**, roughly equal to NCCL and never better than All2All. Ring2D dispatches to
  `copy_ring_push_2d_pcie` ([all_gather_op.cc:398](../../src/coll/ths_op/all_gather_op.cc#L398)),
  a PCIe-topology-aware schedule; on an NVSwitch box where every pair is NV12 that schedule
  buys nothing and costs ordering. Phase 0 never questioned the choice because it was only
  ever compared against NCCL, not against Flux's other modes.
- Ring2D also has the worst run-to-run stability (rank0 stdev 0.100 ms at M=4096 vs 0.004
  for All2All pull).

## E.2 Known limitation found while measuring

`use_cuda_core_ag=True` fails for fp16 with
`unsupported for input_dtype=FP16,scale_dtype=FP32`. The CUDA-core all-gather kernel
`ag_a2a_mode` is only instantiated for INT8 input with an FP32 scale
([all_gather_impls.cu:127](../../src/coll/all_gather_impls.cu#L127)); the cartesian product
it dispatches over contains only `(_S8{}, _FP32{})`. The All2All CUDA-core path therefore
cannot run for fp16 at all. The Ring2D CUDA-core path does run (it uses a different kernel)
but is 2-4x slower than the copy-engine path.

## E.3 Overlap efficiency cannot be quoted at the old precision

`CDE_REPORT.md` computed ECT as `total - gemm_only` from a single run. Repeating the
AG+GEMM benchmark three times per ring mode (M=4096, N=49152, K=12288, fp16, warmup 5,
iters 20; raw in `results/e_flux/ag_gemm_ringmode_repeats.txt`) shows that quantity is not
stable enough to support a 3-significant-figure efficiency:

| Mode | total ms (3 runs) | gemm_only ms (3 runs) | ECT = total - gemm_only |
| --- | --- | --- | --- |
| all2all | 2.709, 2.897, 2.693 | 2.656, 2.528, 2.805 | 0.053, 0.369, **-0.112** |
| ring1d | 2.885, 2.824, 2.726 | 2.877, 2.530, 2.629 | 0.008, 0.294, 0.097 |
| ring2d | 2.700, 2.690, 2.754 | 2.563, 2.594, 2.545 | 0.137, 0.096, 0.209 |

`gemm_only` alone swings 2.528-2.877 ms, and one all2all run produced a **negative ECT**
because `gemm_only` exceeded `total`. Note also that the script reports two mutually
inconsistent "GEMM alone" numbers — the printed `comm` field uses a separate `flux.GemmOnly`
op, not `AGKernel.gemm_only` ([test_ag_kernel.py:517](../../test/python/ag_gemm/test_ag_kernel.py#L517))
— so `comm` and `total - gemm_only` disagree within the same run (ring2d run1: 0.267 vs 0.137).

With the correct denominator (Flux's own Ring2D AG comm, 0.562 ms) the ring2d ECT range
0.096-0.209 ms maps to an overlap efficiency of 63-83%, against the withdrawn 72.1%. The
honest statement is a range, not a point.

End-to-end at this shape the ring mode barely matters, because the GEMM dominates:
median total 2.700 ms (ring2d), 2.709 ms (all2all), 2.824 ms (ring1d) against a torch
baseline of about 3.30 ms.

## E.4 GEMM_RS: the comm is not separable on this hardware

The 99.1% GEMM_RS overlap efficiency is withdrawn for the denominator reason above, and the
numerator is also misread. On sm80 with `nnodes=1`, `fuse_reduction=false` and NVLink
present, `forward_reduce_scatter_impl` only performs `local_reduction` over
`output_buffer` ([gemm_reduce_scatter.cc:758](../../src/gemm_rs/ths_op/gemm_reduce_scatter.cc#L758)).
The scatter itself happens inside the GEMM epilogue, which writes tiles directly into
peers' symmetric buffers through `output_scatter_ptrs`
([gemm_reduce_scatter.cc:190](../../src/gemm_rs/ths_op/gemm_reduce_scatter.cc#L190)).

So the reported `exposed comm 0.005 ms` is the cost of the final local reduce, not of the
reduce-scatter communication — that cost is already inside the 2.800 ms attributed to
"GEMM". There is no exposed API that separates it, and `forward_barrier` is a no-op on
sm80 single-node. A pure Flux RS comm number is not obtainable on this machine without
instrumenting the kernel.

## E.5 On the repo's own comm scripts

`test_comm_ag.py` and `test_comm_rs.py` were listed in `CDE_REPORT.md` as the intended
pure-comm tools. They would not have served the purpose either: both emulate ring schedules
with `dist.batch_isend_irecv` and `torch.Tensor.copy_` over `flux.create_tensor_list`
buffers. They are topology exploration harnesses, not the kernels Flux ships.

Their failures are nonetheless real bugs, still unfixed:

- `test_comm_ag.py --exp=ring2d`: `perf_2d_ring._run()` takes no parameters
  ([test_comm_ag.py:361](../../test/python/ag_gemm/test_comm_ag.py#L361)) but `run_perf`
  invokes `func(iter=n)` ([utils.py:178](../../python/flux/testing/utils.py#L178)).
- `test_comm_rs.py`: references `WORLD_SIZE` at module scope, before it is defined
  ([test_comm_rs.py:31](../../test/python/gemm_rs/test_comm_rs.py#L31)), so it raises
  `NameError` on import; it also needs `copy_utils` on `PYTHONPATH` and the `cuda` package.

## Files

- `scripts/flux_comm_baseline.py`: Flux-native AllGather comm benchmark.
- `results/e_flux/flux_ag_comm_M*.csv`: per-configuration timings, all sizes.
- `results/e_flux/ag_gemm_ringmode_repeats.txt`: AG+GEMM repeats per ring mode.

## What this changes

1. Use All2All, not Ring2D, as the AG mode on this machine. Phase 0's entire E section ran
   the slowest sensible mode.
2. Flux's AG comm is genuinely 1.2x NCCL. Phase 0 could not see this.
3. Every overlap-efficiency figure in Phase 0 is withdrawn. Re-deriving them needs an ECT
   method that does not subtract two separately-measured noisy GEMM timings.
