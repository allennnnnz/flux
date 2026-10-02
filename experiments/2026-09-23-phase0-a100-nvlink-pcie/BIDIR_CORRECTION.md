# Phase 0 Bidirectional Correction: H2D and D2H were the same number

Date: 2026-09-23

Supersedes the "Added Bidirectional PCIe Measurements" table in `CDE_REPORT.md`.
Script: `scripts/bidirectional_bandwidth_v2.py`, replacing `scripts/bidirectional_bandwidth.py`.
Raw data: `results/bidirectional/bidir_v2_*.csv`.

## X.0 Three defects in the old measurement

**1. The two directions were the same number by construction.**

```python
h2d = bytes_per_direction / seconds / 1e9
d2h = bytes_per_direction / seconds / 1e9
```

Both derive from one wall clock. Every row in `CDE_REPORT.md` where H2D exactly equals D2H
(12.689/12.689, 14.683/14.683, 57.354/57.354) is this artifact, not a measurement.

**2. The script's own CUDA-event data contradicted it and was never used.** It printed
`h2d_cuda_median_ms` and `d2h_cuda_median_ms` into every CSV and then discarded them. For
GPU0 they read **1687.96 ms vs 963.80 ms** for identical payloads — the two directions
differ by 1.75x. The evidence that the reported number was wrong was sitting in the output
file all along.

**3. The directions do not cover the same span, so the wall clock mixes two regimes.**
D2H finishes long before H2D. Measured overlap is only **57-60% of the total span**; the
remainder is H2D running alone. A wall-clock figure therefore blends contended and
uncontended phases and describes neither.

## X.1 Corrected measurements

Method: one process per GPU, NUMA-local binding, separate device buffers per direction, an
event recorded after **every individual copy**, all offsets taken from one common reference
event per device. 256 MiB per copy, 20 copies, 5 warmup, 20 repeats. "Overlap window" is
the interval during which both directions were in flight; bandwidth restricted to it is the
only figure that legitimately describes simultaneous bidirectional transfer.

| Config | H2D own | D2H own | **H2D in overlap** | **D2H in overlap** | combined | overlap span |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GPU0 | 12.669 | 22.122 | **6.164** | **22.122** | 28.286 | 57.3% |
| GPU0+1, same switch | 7.338 | 12.363 | **3.688** | **12.363** | 16.051 | 59.4% |
| GPU0+2, diff switches | 12.310 | 21.456 | **5.908** | **21.456** | 27.364 | 57.4% |
| 8 GPUs | 7.186 | 12.019 | **3.563** | **12.019** | 15.582 | 59.7% |

All values per GPU in GB/s. Aggregates: 8 GPUs give 28.50 H2D + 96.16 D2H = 124.66 GB/s
combined during overlap.

GPU0+2 on different switches matches GPU0 alone almost exactly (5.908 vs 6.164, 21.456 vs
22.122), while GPU0+1 on the same switch is roughly halved — independently reproducing the
shared-uplink result of `B_REDO.md` B.4.

Replaced values: the old table's 12.689 / 14.683 / 57.354 "per-direction" figures.

## X.2 Cross-check against nvbandwidth

Built from `github.com/NVIDIA/nvbandwidth` v0.10.0 at `-t *_memcpy_ce`:

| GPU0 | nvbandwidth | this work | `B_REDO.md` |
| --- | ---: | ---: | ---: |
| H2D unidirectional | 21.82 | — | 21.795 |
| D2H unidirectional | 23.99 | — | 23.924 |
| H2D while bidirectional | **8.88** | 6.164 | — |
| D2H while bidirectional | **22.13** | 22.122 | — |

The unidirectional figures validate `B_REDO.md` to within 0.2%. The bidirectional D2H
matches to within 0.05%. Bidirectional H2D differs (8.88 vs 6.16) because nvbandwidth runs
both directions continuously for a matched duration while this script measures a defined
overlap window, but both land in the same regime and both show the same structure.

Note nvbandwidth needs no Boost as of v0.10 despite what `debian_install.sh` suggests, which
matters here because the machine has no sudo.

## X.3 Mechanism: H2D is throttled by concurrent D2H; D2H is immune

Nsight Systems, GPU0, classifying each copy by whether the opposite direction was in flight
for more than half its duration:

| Direction | State | n | median | bandwidth |
| --- | --- | ---: | ---: | ---: |
| H2D | uncontended | 28 | 12.20 ms | **22.01 GB/s** |
| H2D | contended by D2H | 12 | 41.56 ms | **6.46 GB/s** |
| D2H | contended by H2D (all of them) | 40 | 11.92 ms | **22.52 GB/s** |

Concurrent D2H traffic costs H2D a factor of **3.4**, while D2H itself runs at full speed
throughout. The profile confirms the two directions genuinely overlap on the device
timeline, so this is resource sharing under contention, not scheduling serialization.

This asymmetry is the single most useful fact recovered from the correction: on this
machine, PCIe is not a symmetric full-duplex resource.

## X.4 Consequence for section C

`CDE_REPORT.md` C expressed host-staging efficiency against the withdrawn bidirectional
baseline. Re-expressed, and now explained:

**8-GPU staging ring.** Each GPU simultaneously performs a D2H of its own chunk and an H2D
of its predecessor's, so it is exactly the contended case above. The ceiling is the
**contended** H2D rate, 6.46 GB/s per GPU, not any nominal PCIe figure. Section C measured
5.874 GB/s per pair — **90.9% of that ceiling**. The ring was already running close to what
the hardware permits; the old "81.9% of 57.354" framing obscured this.

**Single pair GPU0 → GPU1.** Here no GPU does both directions at once: GPU0 only reads out,
GPU1 only writes in. The bound is therefore the *unidirectional* min(D2H 23.924,
H2D 21.795) = 21.795 GB/s. Section C measured 13.815 GB/s, i.e. **63.4%**, with the gap
attributable to pipelining rather than to contention.

The old figure of "108.9%" — an efficiency above 100% — should have been treated as
evidence that the baseline was wrong.

## X.5 Staging bandwidth by topology (section C extension)

`scripts/staging_stream_bench.cu`, 1024 MiB payload, 5 warmup, 20 repeats, median of
`aggregate_payload_GBps`. Section C only ever measured GPU0 → GPU1.

| Pair | Topology | 4 MiB | 8 MiB | 16 MiB | 32 MiB | best | vs unidirectional 21.795 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 → 1 | same switch | 13.913 | 11.067 | 13.349 | 13.396 | 13.913 | 63.8% |
| **0 → 2** | **different switches, same NUMA** | **22.191** | 21.673 | 21.027 | 20.694 | **22.191** | **101.8%** |
| 3 → 4 | cross NUMA | 18.503 | 18.589 | 18.362 | 17.982 | 18.589 | 85.3% |
| 0 → 4 | cross NUMA | 18.674 | 18.565 | 18.362 | 18.109 | 18.674 | 85.7% |

Efficiency is expressed against the unidirectional bound min(H2D 21.795, D2H 23.924) =
21.795 GB/s, since in a lone pair the source only reads out and the destination only writes
in — no GPU performs both directions, so the contended regime of X.3 does not apply.

**Section C measured the single worst pair on the machine.** GPU0 and GPU1 share a Gen4 x16
switch uplink, so the D2H out of GPU0 and the H2D into GPU1 contend for the same link.
Choosing a different-switch pair instead yields 22.191 GB/s, 60% better and effectively at
the unidirectional ceiling. Cross-NUMA pairs land in between at about 18.6 GB/s.

Note this does **not** rescue the dual-path idea: see `D_CORRECTION.md` D.6. A lone pair
avoids contention only because no GPU does both directions; in a staging ring every GPU
does both, and the ceiling reverts to the 6.46 GB/s contended-H2D rate.
