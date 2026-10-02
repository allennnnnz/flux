# Phase 0 D Correction: the NVLink baseline was 5.7x too slow

Date: 2026-09-23

This supersedes `CDE_REPORT.md` section D in full. Script: `scripts/dual_path_v3.py`,
replacing `scripts/dual_path_v2.py`. Raw data: `results/d_dual_path/dual_path_v3_all8.csv`.

## D.0 Why the old result was wrong

`CDE_REPORT.md` D reported pure-NVLink throughput of 306.923 GB/s aggregate over 8 GPUs —
**38.4 GB/s per GPU**. Flux's own AllGather moves 188.2 GB/s per GPU of ingress on the same
hardware in the same session (`E_CORRECTION.md` E.1), and a single-pair peer copy measures
**270.8 GB/s**. The D baseline was therefore about 5.7x below what this machine does.

Every gain figure in section D was computed against that baseline, so the +7.3% measured
gain and the +15.3% model prediction are both artifacts of dividing by a number that is
too small. They are withdrawn.

### Root cause

PyTorch executes a cross-device copy **on the source device's current stream**, not the
destination's (`aten/src/ATen/native/cuda/Copy.cu`). `dual_path_v2.py` wrapped each peer
copy in the *destination* device's stream and never set any stream on the source, so all 7
outgoing copies from each GPU serialized on that GPU's untouched default stream.

Measured directly, 8 GPUs all-to-all, 256 MiB per GPU:

| Stream placement | per-GPU ingress | aggregate |
| --- | ---: | ---: |
| destination (v2) | 38.1 GB/s | 304.9 GB/s |
| source (v3) | 103.7 GB/s | 829.6 GB/s |

### Secondary cause: the transfer is per-copy-overhead bound below ~1 GiB

Stream count turns out to be irrelevant — 1 stream and 7 streams per device measure
identically. Only the size of each individual peer copy matters:

| per GPU per iteration | per peer copy | per-GPU GB/s | aggregate GB/s |
| ---: | ---: | ---: | ---: |
| 64 MiB | 9.1 MiB | 34.7 | 277.8 |
| 256 MiB | 36.6 MiB | 129.6 | 1036.4 |
| 1024 MiB | 146.3 MiB | 210.9 | 1687.2 |

`dual_path_v3.py` therefore defaults to 1024 MiB. Anything smaller understates NVLink badly.

## D.1 Corrected NVLink baseline

8 GPUs, 1024 MiB per GPU per iteration, 20 copies per interval, 5 warmup intervals
discarded, 20 formal repeats:

| | per-GPU GB/s | aggregate GB/s |
| --- | ---: | ---: |
| v2 (defective) | 38.4 | 306.9 |
| **v3 (corrected)** | **217.66** | **1741.27** |
| Flux All2All, for reference | 188.2 | 1505.6 |

The target set for this fix was "at least match Flux All2All, ≈188 GB/s per GPU". Met:
217.66 GB/s per GPU, 5.7x the v2 figure.

### Nsight Systems verification

Profile of α=0 (`nsys profile -t cuda`, 3 copies x 3 intervals):

- 504 memory operations, all `[CUDA memcpy Peer-to-Peer]`. Nothing is being staged through
  host memory behind our back.
- All 8 devices use 7 distinct streams each, confirming the source-side placement took
  effect.
- Median per-copy bandwidth **216.5 GB/s** (153.39 MB in 708.4 µs), consistent with the
  217.66 GB/s wall-clock figure.
- **Peak concurrency is 8, not 56.** Each GPU executes its 7 outgoing copies serially on
  its copy engines while the 8 GPUs run concurrently with each other. This explains why
  stream count makes no difference: per-GPU egress is already saturated by one copy at a
  time. 8 x 216.5 = 1732 GB/s, matching the measured 1741 GB/s aggregate.

## D.2 Corrected alpha sweep

α = fraction of each GPU's payload diverted through the host-staging ring.

| α | per-GPU GB/s | aggregate GB/s | vs α=0 | staging per-GPU GB/s |
| ---: | ---: | ---: | ---: | ---: |
| 0.00 | **217.66** | 1741.27 | — | 0.00 |
| 0.01 | 210.83 | 1686.65 | **-3.14%** | 2.11 |
| 0.02 | 203.98 | 1631.83 | **-6.28%** | 4.08 |
| 0.03 | 176.34 | 1410.70 | **-18.98%** | 5.29 |
| 0.05 | 106.64 | 853.13 | **-51.01%** | 5.33 |
| 0.08 | 67.03 | 536.27 | **-69.20%** | 5.36 |
| 0.10 | 53.53 | 428.27 | **-75.40%** | 5.35 |

**Every value of α is a net loss, and the loss is monotonic.** There is no optimum. The
staging path saturates at about 5.3 GB/s per GPU — consistent with section C's 5.9 GB/s per
GPU — while each unit of payload diverted to it costs far more NVLink time than it buys.

## D.3 Mechanism

Profiling α=0.02 separates the two effects.

| | peer copy bandwidth | staging D2H | staging H2D |
| --- | ---: | ---: | ---: |
| α=0 | 216.5 GB/s | — | — |
| α=0.02 | 212.1 GB/s (-2.0%) | 10.9 GB/s | 11.8 GB/s |

The NVLink copies are barely disturbed — only 2% slower. The loss does **not** come from
staging stealing NVLink bandwidth.

It comes from the staging chain being a **serial D2H → H2D dependency**. A 21.47 MB chunk
takes 1970 µs down to host and 1821 µs back up, 3.79 ms in total, while the NVLink portion
of the same iteration moves 0.98 GiB in 4.96 ms. The staging chain is nearly as long as the
entire NVLink transfer while carrying 2% of the payload.

Model: `t(α) = max((1-α)·S/B_nv, α·S·(1/B_d2h + 1/B_h2d))`, with S = 1 GiB,
B_nv = 217.66, B_d2h = 10.9, B_h2d = 11.8 GB/s.

- Crossover at α = 0.0254, predicted best throughput 223.3 GB/s, i.e. **+2.6%** at most.
- At α = 0.05 the model predicts 109 GB/s; **measured 106.64 GB/s**. The model is sound.

So even a perfectly overlapped implementation could gain at most about 2.6% here, and the
measurement does not reach even that, because a double-buffer depth of 2 cannot hide a
3.79 ms serial chain behind a 4.96 ms window when both ends share the source and
destination GPUs' copy engines.

## D.4 Measurement caveat for anyone re-running this

CUPTI classifies **pinned-host → device copies as `Peer-to-Peer`**, not `Host-to-Device`.
At α=0.02 the profile shows 576 PtoP and 72 DtoH operations and zero HtoD, which reads as
if the staging return leg never ran. It did: 576 = 504 real peer copies (150,323,856 B
each) + 72 staging H2D (21,474,832 B each), and the byte totals reconcile exactly to
77,309,411,328. Separate the two by transfer size, not by the reported copy kind.

## D.5 Consequence

This removes the only evidence that pointed toward a dual-channel transport being useful on
this machine. With a correct NVLink baseline, host staging is a loss at every mixing ratio
tested, and the theoretical ceiling if it were implemented perfectly is +2.6%.

Combined with `STATUS.md` 5.1 — making Flux's communication 17% faster produced zero
end-to-end change, because comm is already fully hidden behind the GEMM — there is no
remaining argument for dual-channel scheduling on A100. The decision to leave Flux core
scheduling alone stands, now on measured rather than assumed grounds.

## D.6 The conclusion is robust to staging-ring topology

An obvious objection: the default ring 0→1→2→…→7→0 puts four of its eight hops on
same-PCIe-switch pairs (0→1, 2→3, 4→5, 6→7), and `BIDIR_CORRECTION.md` X.5 shows a
same-switch staging pair runs at 13.9 GB/s against 22.2 GB/s for a different-switch pair.
The sweep was therefore re-run with a switch-aware ring, 0→2→4→6→1→3→5→7→0, which has no
same-switch hop:

| α | naive ring, per-GPU GB/s | switch-aware ring, per-GPU GB/s |
| ---: | ---: | ---: |
| 0.00 | 217.66 | 217.27 |
| 0.02 | 203.98 (-6.28%) | 201.13 (-7.43%) |
| 0.05 | 106.64 (-51.01%) | 104.03 (-52.12%) |
| 0.08 | 67.03 (-69.20%) | 65.13 (-70.02%) |

Staging saturates at 5.20 GB/s per GPU with the switch-aware ring against 5.35 with the
naive one — no improvement. Every α remains a net loss.

The reason is the asymmetry in `BIDIR_CORRECTION.md` X.3. **In any staging ring every GPU
is simultaneously a producer and a consumer**, so every GPU is doing a D2H and an H2D at
once, which is precisely the contended regime where H2D collapses to 6.46 GB/s. Switch
assignment is a second-order effect on top of that. The 22.19 GB/s measured for a lone
GPU0→GPU2 pair is achievable only because no single GPU does both directions there; it is
structurally unreachable in a ring.

Measured ring staging of 5.20-5.35 GB/s per GPU is 80-83% of the 6.46 GB/s contended-H2D
ceiling, so the implementation is already close to what the hardware allows. There is no
remaining headroom to recover.
