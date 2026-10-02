# Flux AG+GEMM Baseline (rebuilt)

Date: 2026-09-23

Replaces the AG+GEMM timings and every overlap-efficiency figure in
`CDE_REPORT.md` section E and `E_CORRECTION.md` E.3.
Script: `scripts/flux_ag_gemm_baseline_v2.py`. Raw: `results/e_flux/ag_gemm_v2_N*.csv`.

## F.1 Why the old ECT could not be quoted

`CDE_REPORT.md` computed exposed comm time as `total - gemm_only` from two separately
averaged loops. Repeating that method showed `gemm_only` swinging 2.528-2.877 ms and one
run producing a **negative ECT**. The source script also emits two mutually inconsistent
"GEMM alone" numbers — the printed `comm` field uses a separate `flux.GemmOnly` op rather
than `AGKernel.gemm_only` — which disagree within a single run (0.267 vs 0.137 ms).

## F.2 Method

- **GEMM-only reference**: `AGKernel.gemm_only()`. This is not a different operator — it
  runs the *same* AG-fused GEMM kernel (`gemm_op`) with the full input already in place and
  a barrier tensor of `torch::ones({world_size})`, i.e. every signal pre-set to true
  ([all_gather_gemm_op.cc:167](../../src/ag_gemm/ths_op/all_gather_gemm_op.cc#L167)).
- **Interleaved**: within each round, the overlapped `forward()` and the `gemm_only()`
  reference run back to back, each with its own CUDA events. ECT is the difference of two
  adjacent measurements, never of two separate averages.
- **200 rounds** per configuration; reported ECT is the median of per-round differences.
- **SM clock** sampled in the background; rounds overlapping a downclock are discarded.
- dtype **bfloat16**; AG mode **All2All, pull** (the fastest on this machine, see
  `E_CORRECTION.md` E.1); weight column-sharded as `(N/world_size, K)` per repo convention
  ([test_ag_kernel.py:99](../../test/python/ag_gemm/test_ag_kernel.py#L99)).
- M=4096, K=12288, world 8. Correctness: allclose against torch on all ranks.

## F.3 Results

All times in ms. 200 rounds each, 0 rounds discarded for clock, modal SM clock
1155-1200 MHz.

| N | ring mode | overlapped | GEMM only | **ECT** | ECT p10-p90 | ECT stdev | negative ECT |
| ---: | --- | ---: | ---: | ---: | --- | ---: | ---: |
| 4096 | all2all | 0.7363 | 0.2775 | **0.4588** | 0.4454-0.5079 | 0.0591 | 0/200 |
| 4096 | ring2d | 0.8243 | 0.2775 | **0.5473** | 0.5274-0.5734 | 0.1037 | 0/200 |
| 8192 | all2all | 0.6769 | 0.5642 | **0.1137** | 0.0973-0.1741 | 0.0804 | 0/200 |
| 8192 | ring2d | 0.7516 | 0.4905 | **0.2473** | 0.1833-0.2775 | 0.0554 | 0/200 |
| 49152 | all2all | 2.7469 | 2.5620 | **0.1638** | 0.1290-0.2150 | 0.0432 | 0/200 |
| 49152 | ring2d | 2.7720 | 2.5641 | **0.1679** | 0.1352-0.2191 | 0.0922 | 0/200 |

**Zero negative ECT across all 1200 rounds.** `gemm_only` stdev is 0.0012-0.0433 ms against
the roughly 0.15 ms run-to-run swing of the old method. The quantity is now stable enough to
quote.

## F.4 Overlap efficiency

Using Flux's own standalone AllGather time as the denominator (`E_CORRECTION.md` E.1:
All2All 0.468 ms, Ring2D 0.562 ms at this shape), and defining efficiency as the fraction
of the serial cost that overlap removes —
`(t_serial - t_measured) / (t_serial - t_perfect)` where `t_serial = gemm + comm` and
`t_perfect = max(gemm, comm)`:

| N | mode | gemm | comm | serial | perfect | measured | **efficiency** |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | all2all | 0.278 | 0.468 | 0.746 | 0.468 | 0.736 | **~2%** |
| 8192 | all2all | 0.564 | 0.468 | 1.032 | 0.564 | 0.677 | **76%** |
| 49152 | all2all | 2.562 | 0.468 | 3.030 | 2.562 | 2.747 | **60%** |

At N=4096 the GEMM is shorter than the communication and essentially no overlap occurs —
the measured total is within 1.3% of fully serial execution. Overlap only becomes effective
once the GEMM is long enough to hide the transfer.

The withdrawn figure was 72.1%. The corrected value at the paper shape is 60%, and the
number is now reproducible.

## F.5 Where the communication mode matters

| N | all2all | ring2d | difference |
| ---: | ---: | ---: | ---: |
| 4096 | 0.7363 | 0.8243 | **-10.7%** |
| 8192 | 0.6769 | 0.7516 | **-9.9%** |
| 49152 | 2.7469 | 2.7720 | **-0.9%** |

This refines `STATUS.md` 5.1. At the paper shape N=49152 the transport choice is worth 0.9%
end-to-end — communication really is hidden, as claimed. But at N=4096 and N=8192 it is
worth about 10%, because the GEMM no longer dominates. **Communication is on the critical
path at small N.**

That is a genuine qualification, so it is worth stating what it does *not* imply for the
dual-path question. Even at N=4096, where comm is fully exposed, host staging adds at most
5.3 GB/s per GPU against NVLink's 188-218 GB/s — under 3% more ingress bandwidth, worth
about 0.013 ms of the 0.4588 ms ECT, or 1.7% end-to-end in the best case. And
`D_CORRECTION.md` D.2 measures it as a net *loss* at every mixing ratio. The actionable
lever at small N is choosing All2All over Ring2D, which is free and worth 10%.

## F.6 Practical recommendation

Use **All2All with `use_read=True`** as the AG mode on this machine. It is the fastest
standalone (`E_CORRECTION.md` E.1), never worse end-to-end, and worth about 10% at the
smaller shapes. All Phase 0 measurements before this document used Ring2D.
