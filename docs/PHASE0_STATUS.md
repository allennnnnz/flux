# Phase 0 Status

> **Frozen record — Phase 0 is closed (2026-09-23).** Read-only. Raw data, scripts and the
> per-section correction documents referenced below live in
> `experiments/2026-09-23-phase0-a100-nvlink-pcie/`; bare filenames such as `D_CORRECTION.md`
> refer to that folder. If a later audit overturns anything here, record it in
> `DECISIONS.md` and in the consuming workstream's STATUS withdrawal table — do not edit this
> file.

Date: 2026-09-23
Machine: `css-host-158`, 8x A100-SXM4-80GB

This is the single entry point for the experiment. It states what currently holds, what has
been withdrawn, and what is still open. Where a number appears here, the document that owns
it is named. **Read this before citing anything from `README.md`.**

---

## 0. Document map and reading order

| Document | Time | Status |
| --- | --- | --- |
| `README.md` | 12:25 | Valid. Summary and index; points here for detail. |
| `AUDIT.md` | 09:36 | Valid. Records which `README.md` claims failed audit and why. |
| `B_REDO.md` | 09:58 | Valid. Auditable replacement for all PCIe bandwidth numbers. |
| `CDE_REPORT.md` | 10:42 | Section C valid. **Sections D and E withdrawn** (annotated in place). |
| `E_CORRECTION.md` | 11:25 | Valid. Replaces section E. |
| `D_CORRECTION.md` | 11:52 | Valid. Replaces section D. |
| `BIDIR_CORRECTION.md` | 12:07 | Valid. Replaces the bidirectional table; extends section C. |
| `FLUX_BASELINE.md` | 12:20 | Valid. Replaces the AG+GEMM timings and all overlap efficiencies. |
| `STATUS.md` | this file | Current consolidated state. |

The distilled conclusions and design rules that later workstreams consume are in
`PHASE0_FINDINGS.md` (same folder). This file is the full record.

The experiment went through one initial pass and three correction rounds. Each round
invalidated part of the previous one, which is why the reading order matters.

---

## 1. Environment

- GPU: 8x NVIDIA A100-SXM4-80GB (GA100), NVSwitch, all GPU pairs report `NV12`.
- Driver 615.71.09; `nvidia-smi` reports CUDA user-mode stack 13.4.
- pixi environment; PyTorch 2.6.0+cu124; `nvidia.nvshmem` imports.
- NUMA: GPU0-3 on node 0 (CPU 0-31,64-95), GPU4-7 on node 1 (CPU 32-63,96-127).
- **No sudo.** This is why `lspci -vv` returns `Capabilities: <access denied>` and why
  sysfs was used for all link interrogation.
- Flux runs launch as
  `pixi run --manifest-path pixi.toml ./launch.sh <script> ...`
  (`launch.sh` alone fails: `torchrun: command not found`).

---

## 2. What currently holds

### 2.1 PCIe topology (B_REDO.md B.1)

Four Broadcom PEX880xx Gen4 switches. **GPUs share a switch uplink in pairs: 0/1, 2/3,
4/5, 6/7.** Every root port and switch upstream port reads 16.0 GT/s x16 (Gen4 x16) from
sysfs.

Internal switch link Gen/width beyond the first upstream port remains unconfirmable without
sudo.

### 2.2 PCIe bandwidth, unidirectional (B_REDO.md B.3-B.5)

Method: `scripts/global_bandwidth_v2.py`, 8 spawned processes, 1024 MiB/GPU/copy, 20 copies
per measured interval, 5 warmup intervals discarded, 20 formal repeats, wall-clock
aggregate, explicit `numa_run_on_node` + `numa_set_membind` before importing torch.

| Case | H2D GB/s | D2H GB/s |
| --- | ---: | ---: |
| Single GPU (GPU0) | 21.795 | 23.924 |
| 8 GPUs, NUMA local | 101.173 (12.647/GPU) | 105.560 (13.195/GPU) |
| 8 GPUs, NUMA remote | 82.415 (10.302/GPU) | 75.583 (9.448/GPU) |

**Shared-uplink control experiment** — this is the cleanest result in the whole experiment:

| GPUs | Topology | H2D GB/s | D2H GB/s |
| --- | --- | ---: | ---: |
| 0 | single | 21.795 | 23.924 |
| 0,1 | **same switch** | 25.467 | 26.600 |
| 0,2 | different switches, same NUMA | 42.903 | 46.638 |
| 0,2,4,6 | four different switches | 85.119 | 93.072 |
| 0-7 | all | 101.173 | 105.560 |

Two GPUs on the same switch reach roughly one Gen4 x16 uplink's worth of bandwidth; two on
different switches reach roughly twice a single GPU. The shared-upstream hypothesis is
confirmed.

### 2.3 PCIe bandwidth, bidirectional (BIDIR_CORRECTION.md)

`scripts/bidirectional_bandwidth_v2.py`. The v1 script reported H2D and D2H as the **same
number by construction** (both `bytes / wall_seconds`), which is why every old row had
H2D == D2H exactly. Its own unused CUDA-event columns already showed the directions
differing by 1.75x.

Per GPU, GB/s, restricted to the window in which both directions were actually in flight
(that window is only 57-60% of the total span):

| Config | H2D (overlap) | D2H (overlap) | combined |
| --- | ---: | ---: | ---: |
| GPU0 | 6.164 | 22.122 | 28.286 |
| GPU0+1, same switch | 3.688 | 12.363 | 16.051 |
| GPU0+2, diff switches | 5.908 | 21.456 | 27.364 |
| 8 GPUs | 3.563 | 12.019 | 15.582 |

**PCIe here is not a symmetric full-duplex resource.** Nsight Systems, classifying each
copy by whether the opposite direction was in flight: H2D runs at 22.01 GB/s uncontended
but **6.46 GB/s** when a D2H overlaps it — a factor of 3.4 — while D2H holds 22.52 GB/s
throughout. The two directions do genuinely overlap on the device timeline, so this is
resource contention, not serialization.

Cross-checked against `nvbandwidth` v0.10.0: unidirectional H2D 21.82 / D2H 23.99 GB/s
(validating `B_REDO.md` to within 0.2%), bidirectional D2H 22.13 (matching to 0.05%).

### 2.4 Host staging (CDE_REPORT.md C, extended in BIDIR_CORRECTION.md X.5)

`scripts/staging_stream_bench.cu`. Double-buffered producer/consumer using
`cuStreamWriteValue32` / `cuStreamWaitValue32`, host buffers via `numa_alloc_onnode()` then
`cudaHostRegisterMapped`.

Section C measured only GPU0 → GPU1, which is **the worst pair on the machine** — those two
share a Gen4 x16 switch uplink. By topology, best over chunk sizes, against the
unidirectional bound min(H2D 21.795, D2H 23.924) = 21.795 GB/s:

| Pair | Topology | best GB/s | efficiency |
| --- | --- | ---: | ---: |
| 0 → 1 | same switch | 13.913 | 63.8% |
| **0 → 2** | different switches | **22.191** | **101.8%** |
| 3 → 4 | cross NUMA | 18.589 | 85.3% |
| 0 → 4 | cross NUMA | 18.674 | 85.7% |

8-GPU ring: 46.988 GB/s aggregate, 5.874 per pair. Here every GPU is simultaneously a
producer and a consumer, so the ceiling is the *contended* H2D rate of 6.46 GB/s (2.3), and
5.874 is **90.9%** of it — the ring is already near the hardware limit. The old "81.9% of
57.354" and "108.9%" framings came from the withdrawn bidirectional baseline; an efficiency
above 100% should have been read as evidence the baseline was wrong.

16 MiB is the chunk sweet spot for the ring; 1 MiB collapses to 18.727 GB/s aggregate.

### 2.5 Dual path, NVLink + PCIe staging (D_CORRECTION.md)

`scripts/dual_path_v3.py`. The v2 script measured 38.4 GB/s per GPU of NVLink because
PyTorch runs a cross-device copy on the **source** device's current stream, and v2 set the
destination's — so each GPU's 7 outgoing copies serialized on its default stream. The
transfer is also per-copy-overhead bound below ~1 GiB per GPU per iteration; stream count
is irrelevant.

Corrected NVLink baseline, 8 GPUs, 1024 MiB/GPU, 20 copies, 5 warmup, 20 repeats:

| | per-GPU GB/s | aggregate GB/s |
| --- | ---: | ---: |
| v2 (defective) | 38.4 | 306.9 |
| **v3 (corrected)** | **217.66** | **1741.27** |
| Flux All2All, reference | 188.2 | 1505.6 |

Corrected alpha sweep — **every α is a net loss, monotonically**:

| α | per-GPU GB/s | vs α=0 | staging per-GPU GB/s |
| ---: | ---: | ---: | ---: |
| 0.00 | 217.66 | — | 0.00 |
| 0.01 | 210.83 | -3.14% | 2.11 |
| 0.02 | 203.98 | -6.28% | 4.08 |
| 0.03 | 176.34 | -18.98% | 5.29 |
| 0.05 | 106.64 | -51.01% | 5.33 |
| 0.10 | 53.53 | -75.40% | 5.35 |

Nsight Systems confirms: all operations are `[CUDA memcpy Peer-to-Peer]`, 7 streams in use
per device, median 216.5 GB/s per copy, and peak concurrency 8 — each GPU serializes its 7
outgoing copies on its copy engines while the 8 GPUs run concurrently.

Mechanism (profiled at α=0.02): peer copies slow by only 2% (216.5 → 212.1 GB/s), so
staging does **not** steal NVLink bandwidth. The loss is that the staging chain is a serial
D2H → H2D dependency — 21.47 MB costs 1970 µs down and 1821 µs back, 3.79 ms, against
4.96 ms for the 0.98 GiB NVLink portion of the same iteration. Model
`t(α) = max((1-α)S/B_nv, αS(1/B_d2h + 1/B_h2d))` caps the best possible gain at **+2.6%**
at α≈0.025, and predicts 109 GB/s at α=0.05 against 106.64 measured.

### 2.6 Flux AllGather, Flux's own comm path (E_CORRECTION.md E.1)

`scripts/flux_comm_baseline.py` drives `flux.AllGatherOp` — the same code
`AGKernel.forward()` uses internally. Every configuration verified bitwise-identical to
NCCL `all_gather` on all 8 ranks before timing. Reported value is rank-max median.

M=4096, K=12288, fp16, world 8:

| Path | rank-max median ms | algbw GB/s | vs NCCL |
| --- | ---: | ---: | ---: |
| NCCL all_gather | 0.569 | 176.8 | 1.00x |
| **Flux All2All, pull** | **0.468** | **214.9** | **1.22x** |
| Flux All2All, push | 0.471 | 213.8 | 1.21x |
| Flux Ring1D, pull | 0.496 | 202.8 | 1.15x |
| Flux Ring2D, push | 0.562 | 179.1 | 1.01x |
| Flux Ring1D, push | 0.565 | 178.2 | 1.01x |
| Flux Ring2D, push, CUDA core | 1.066 | 94.4 | 0.53x |

Confirmed across M = 1024 / 2048 / 4096 / 8192 / 16384.

- Flux's own AG beats NCCL by 13-22% in All2All mode at every size.
- **Ring2D — the mode every Phase 0 E run used — is among the slowest modes here.** It
  dispatches to `copy_ring_push_2d_pcie`, a PCIe-topology-aware schedule that buys nothing
  on an all-NV12 NVSwitch box.

### 2.7 Flux AG+GEMM end-to-end (FLUX_BASELINE.md)

`scripts/flux_ag_gemm_baseline_v2.py`. ECT is measured as a **per-round difference** of
interleaved overlapped and GEMM-only runs, 200 rounds, bfloat16, All2All pull, SM clock
logged. The GEMM-only reference is `AGKernel.gemm_only()` — the same AG-fused GEMM kernel
with the full input in place and all signals pre-set true, not a different operator.

M=4096, K=12288, world 8, times in ms:

| N | mode | overlapped | GEMM only | ECT | overlap efficiency |
| ---: | --- | ---: | ---: | ---: | ---: |
| 4096 | all2all | 0.7363 | 0.2775 | 0.4588 | ~2% |
| 4096 | ring2d | 0.8243 | 0.2775 | 0.5473 | — |
| 8192 | all2all | 0.6769 | 0.5642 | 0.1137 | 76% |
| 8192 | ring2d | 0.7516 | 0.4905 | 0.2473 | — |
| 49152 | all2all | 2.7469 | 2.5620 | 0.1638 | **60%** |
| 49152 | ring2d | 2.7720 | 2.5641 | 0.1679 | — |

**Zero negative ECT across all 1200 rounds**, against the old method which produced them.
The withdrawn overlap efficiency was 72.1%; the corrected, reproducible value at the paper
shape is **60%**.

Communication mode matters by shape: all2all beats ring2d by **10.7% at N=4096** and
**9.9% at N=8192**, but only **0.9% at N=49152**. Communication is on the critical path at
small N and hidden at large N. At N=4096 essentially no overlap occurs at all — the
measured total is within 1.3% of fully serial execution.

Correctness across Phase 0: AG+GEMM allclose and bitwise match vs torch; GEMM_RS allclose
but not bitwise (reduction ordering — expected).

---

## 3. What is withdrawn

Everything in this section was published in an earlier document and must not be cited.

| Withdrawn claim | Source | Why | Replacement |
| --- | --- | --- | --- |
| `NV12` proves PCIe switch topology | README | `NV12` describes NVLink/NVSwitch, not PCIe sharing | B_REDO B.1 |
| "a second node is not required" | README | Phase 0 never ran a dual-node experiment | none; question unanswered |
| Aggregate H2D 162.5 / D2H 172.8 GB/s | README | Global timing method and raw output not preserved | B_REDO B.3 |
| D2D 11.23 TB/s | README | Same | not re-measured |
| Single-GPU 21.6-22.6 GB/s | README | Could not establish whether single or aggregated | B_REDO B.5 |
| Dual path best α=0.02, +3.6% | README | Superseded by a far more rigorous method | CDE_REPORT D |
| Flux AG timing `total 0.000 ms` | README | Invalid output | CDE_REPORT E / E_CORRECTION |
| NCCL as "pure communication baseline" | CDE_REPORT E | NCCL and Flux do not share a comm path | E_CORRECTION E.1 |
| AG overlap efficiency 72.1% | CDE_REPORT E | Flux ECT divided by an **NCCL** time | E_CORRECTION E.3: a 63-83% range |
| GEMM_RS overlap efficiency 99.1% | CDE_REPORT E | Same denominator defect, plus misread numerator | E_CORRECTION E.4: not measurable here |
| "Flux no-overlap comm 0.906 ms" | CDE_REPORT E | **This row is NCCL too**, not Flux | E_CORRECTION E.0 |
| NVLink baseline 306.9 GB/s (38.4/GPU) | CDE_REPORT D | Peer copies placed on the destination stream; 5.7x too slow | D_CORRECTION D.1: 217.66/GPU |
| Dual path best α=0.1, +7.3% | CDE_REPORT D | Artifact of the above baseline | D_CORRECTION D.2: every α is a loss |
| Bidirectional H2D = D2H (12.689 etc.) | CDE_REPORT | Both were `bytes/wall_seconds`, the same expression | BIDIR_CORRECTION X.1 |
| Staging efficiency 108.9% / 81.9% | CDE_REPORT C | Expressed against the withdrawn bidirectional baseline | BIDIR_CORRECTION X.4-X.5 |
| AG+GEMM ECT from `total - gemm_only` | CDE_REPORT E | Two separately-averaged loops; produced negative ECT | FLUX_BASELINE F.3 |

The last row is the most consequential finding of the correction rounds: because
`perf_flux_no_overlap` calls `all_gather_into_tensor_with_fp8` (a wrapper over
`torch.distributed.all_gather_into_tensor`), **no number anywhere in the original Phase 0
measured Flux's own communication.**

---

## 4. Open items

Priority order set 2026-09-23. **All five items are complete.** Original text retained
below each item for the record.

1. ~~**Fix the NVLink path in `scripts/dual_path_v2.py`.**~~ **DONE** — see
   `D_CORRECTION.md` and 2.5. Root cause was destination-stream placement, not the
   per-copy sync originally suspected. Corrected baseline 217.66 GB/s/GPU (target was
   ≈188). Full α sweep re-run: every α is a net loss. Nsight Systems timeline verified.
   Original text below for the record.

   ~~Fix the NVLink path in `scripts/dual_path_v2.py`.~~ Target: per-GPU NVLink bandwidth
   at least matching Flux All2All (≈188 GB/s per GPU). Confirm peer access is enabled,
   confirm the 7 peer copies per GPU actually run concurrently on separate streams, and
   verify the timeline in Nsight Systems. Identified cause: `run_once()` calls
   `sync(devices)` on every invocation while being driven in a per-copy loop, so every
   1 GiB transfer drains all 8 GPUs and no cross-copy overlap is possible. After the fix,
   re-run α = 0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1, reporting **both per-GPU and 8-GPU
   aggregate** for every number.

2. ~~**Validate the bidirectional measurement (2.3).**~~ **DONE** — `BIDIR_CORRECTION.md`. v1 emitted one number for both directions; corrected, cross-checked against nvbandwidth, mechanism profiled. Original text:

   ~~Validate the bidirectional measurement (2.3).~~ Switch to independent CUDA-event
   timing per direction. Use Nsight Systems to confirm H2D and D2H genuinely overlap and to
   identify which copy engine each uses. Cross-check by building
   `github.com/NVIDIA/nvbandwidth` and running its bidirectional cases.

3. ~~**Extend section C.**~~ **DONE** — `BIDIR_CORRECTION.md` X.5. GPU0→GPU2 and cross-NUMA measured; efficiency re-expressed against the unidirectional bound. Section C had measured the worst pair on the machine. Original text:

   ~~Extend section C.~~ Add GPU0 → GPU2 (different switches) and GPU3 → GPU4 (cross-NUMA).
   Re-express staging efficiency against the **unidirectional** baseline, not the
   bidirectional one.

4. ~~**Rebuild the Flux baseline properly.**~~ **DONE** — `FLUX_BASELINE.md`. bf16, All2All pull, interleaved per-round ECT, 200 rounds, SM clock logged, N ∈ {4096, 8192, 49152}. Zero negative ECT in 1200 rounds. Original text:

   ~~Rebuild the Flux baseline properly.~~ Use All2All pull and **bfloat16**. Replace the
   GEMM-only reference with `AGKernel` run on data already fully in place and all signals
   pre-set to true. Interleave the overlap and GEMM-only variants within each round, repeat
   **≥200 rounds**, and take the median of the per-round difference rather than the
   difference of two separately-averaged runs. Log SM clock in the background and discard
   anomalous rounds. Add N = 4096 and N = 8192 alongside the paper's N = 49152.

5. ~~**Rewrite `README.md`**~~ **DONE** — rewritten as a summary plus index; `run_phase0_repro.sh` now reproduces only the currently-valid set and names inline the superseded scripts it deliberately skips. Original text:

   ~~Rewrite `README.md`~~ down to a summary plus a pointer to this file, and update
   `run_phase0_repro.sh` so it reproduces the currently-valid set rather than the withdrawn
   one.

### Explicitly not being pursued

- **The D-section α gap between 0.1 and 0.2.** Deferred until item 1 lands; the present
  curve shape is untrustworthy, so refining its grid would be premature.
- **The two repo test-script bugs** (`test_comm_ag.py --exp=ring2d` passing `iter` to a
  zero-parameter `_run()`; `test_comm_rs.py` referencing `WORLD_SIZE` at module scope) and
  the **unconfirmed switch-internal PCIe link spec** (blocked on sudo). Keep the record;
  take no action.

### Standing constraints discovered, no action required

- `use_cuda_core_ag=True` cannot run for fp16: `ag_a2a_mode` is only instantiated for INT8
  input with an FP32 scale (`all_gather_impls.cu:127`).
- GEMM_RS communication is not separable on sm80 single-node: the scatter lives in the GEMM
  epilogue writing to peers via `output_scatter_ptrs`, `forward_reduce_scatter` performs
  only the final local reduce, and `forward_barrier` is a no-op. Measuring it needs kernel
  instrumentation.
- The ECT method `total - gemm_only` is unusable as written: `gemm_only` swings
  2.528-2.877 ms across repeats and one run produced a negative ECT. The script also
  reports two inconsistent "GEMM alone" numbers (the printed `comm` field uses a separate
  `flux.GemmOnly` op, not `AGKernel.gemm_only`), disagreeing within one run (0.267 vs
  0.137 ms). Item 4's interleaved per-round-difference method is the replacement.

---

## 5. Where the decision stands

**Decision: unchanged. Do not modify Flux core scheduling on A100.**

The original `README.md` reached this conclusion from a withdrawn +3.6% figure. The
conclusion survives, but the earlier reasoning does not and the D-section evidence that
appeared to weaken it must not be used.

### 5.1 Primary evidence: a natural experiment already in this data

Sections 2.6 and 2.7 form a controlled comparison that needs no new measurement:

| | AG comm time | AG+GEMM end-to-end |
| --- | ---: | ---: |
| Ring2D | 0.562 ms | 2.700 ms |
| All2All pull | 0.468 ms | 2.709 ms |
| Change | **-17%** | **none** |

Making Flux's communication 17% faster produced **zero** end-to-end improvement. At this
shape the communication is already fully hidden behind the GEMM, so the critical path is
the GEMM, not the transport. A second transport channel adds bandwidth to a resource that
is not the bottleneck.

`FLUX_BASELINE.md` F.5 confirms this on the rebuilt 200-round baseline (bf16, All2All vs
Ring2D): **0.9%** apart at N=49152 — but **10.7%** apart at N=4096 and **9.9%** at N=8192,
where the GEMM no longer dominates. So the honest statement is narrower than "comm never
matters": communication is hidden at the paper shape and **on the critical path at small
N**.

That qualification does not rescue dual-channel. Even at N=4096, where comm is fully
exposed, host staging adds at most 5.3 GB/s per GPU against NVLink's 188-218 — under 3%
more ingress, worth ~0.013 ms of the 0.4588 ms ECT, or 1.7% end-to-end at best; and D.2
measures it as a net *loss* at every mixing ratio. The free lever at small N is choosing
All2All over Ring2D, worth 10%.

### 5.2 Secondary evidence: the headroom ceiling

Even granting the C-section staging numbers at face value:

- Per-GPU host staging (8-GPU ring): 46.988 / 8 = **5.9 GB/s per GPU**.
- Per-GPU NVLink ingress, from Flux's own All2All (2.6): 100,663,296 x 7/8 / 0.468448 ms =
  **188.2 GB/s per GPU**.

Ratio: 5.9 / 188.2 ≈ **3.1%**. Host staging can add at most about 3% of additional ingress
bandwidth on this machine, before accounting for any of it being wasted on a non-bottleneck
resource per 5.1.

### 5.3 The D-section result was an artifact — now corrected and resolved

`CDE_REPORT.md` D reported pure-NVLink throughput of 38.4 GB/s per GPU. The machine
actually does 217.66 GB/s per GPU (270.8 GB/s for a single peer pair). The cause was
destination-stream placement of peer copies; PyTorch runs cross-device copies on the
**source** device's stream, so all 7 outgoing copies per GPU serialized on its default
stream. D's +7.3% gain and +15.3% model prediction were both artifacts of dividing by a
baseline 5.7x too small.

**Re-measured with the corrected baseline (`D_CORRECTION.md`), every α is a net loss:**
-3.14% at α=0.01, -6.28% at α=0.02, -18.98% at α=0.03, -75.40% at α=0.10. Monotonic, no
optimum.

The mechanism is not bandwidth theft — profiling shows peer copies slow by only 2% when
staging is active. It is that the staging chain is a serial D2H → H2D dependency costing
3.79 ms for 21.47 MB, against 4.96 ms for the 0.98 GiB NVLink portion of the same
iteration. The best a perfect implementation could achieve is **+2.6%** at α≈0.025, and the
model that yields that number predicts 109 GB/s at α=0.05 against 106.64 measured.

This removes the only evidence that had pointed toward dual-channel transport being useful
here.

### 5.4 What would change the decision

Nothing in Phase 0 does. A dual-channel transport becomes interesting only where
communication is actually on the critical path **and** the second channel is not an order
of magnitude slower than the first. Small N satisfies the first condition on this machine
but not the second: PCIe staging is ~3% of NVLink here, and measured as a net loss.

The conditions that would change the answer are a machine without full NVLink, a PCIe-only
or heterogeneous accelerator, or any setting where the two paths are within a small factor
of each other. That was already `README.md`'s stated fallback rationale — backend
abstraction as a proxy for future PCIe-only devices — and it remains the defensible one.

---

## 6. Files

Scripts (`scripts/`):

| Script | Purpose | Status |
| --- | --- | --- |
| `global_bandwidth_v2.py` | 8-process PCIe H2D/D2H with NUMA binding | current (B) |
| `bidirectional_bandwidth_v2.py` | simultaneous H2D+D2H, per-direction events | **current** |
| `staging_stream_bench.cu` | double-buffered host staging (`--devices 0,1`, not `=`) | current (C) |
| `dual_path_v3.py` | NVLink + staging alpha sweep, source-stream placement | **current** (D) |
| `flux_comm_baseline.py` | Flux-native AllGather comm | current (E) |
| `flux_ag_gemm_baseline_v2.py` | AG+GEMM, interleaved per-round ECT | **current** (E) |
| `cross_numa_pair.py` | GPU3 D2H + GPU4 H2D through one buffer | current |
| `run_phase0_repro.sh` | reproduction driver, valid set only | **current** |
| `nccl_collective_baseline.py` | NCCL AG/RS | NCCL reference only, **not** a Flux baseline |
| `bidirectional_bandwidth.py` | v1 | **withdrawn** — one number for both directions |
| `dual_path_v2.py` | v2 | **withdrawn** — destination-stream placement, 5.7x slow |
| `dual_path_bench.py` | first pass | superseded twice |
| `global_bandwidth.py` | first pass | superseded by v2 |

Results (`results/`). Files from the correction rounds:

- `bidirectional/bidir_v2_*.csv` — per-direction bidirectional
- `c_staging/staging_v2_*.csv` — staging by topology
- `d_dual_path/dual_path_v3_all8.csv`, `dual_path_v3_switchaware_ring.csv`
- `e_flux/flux_ag_comm_M*.csv` — Flux-native AllGather
- `e_flux/ag_gemm_v2_N*.csv` — 200-round AG+GEMM baseline
- `e_flux/ag_gemm_ringmode_repeats.txt` — evidence the old ECT method was unstable

External tools used, built in the scratchpad (not checked in):
`github.com/NVIDIA/nvbandwidth` v0.10.0 (needs no Boost despite `debian_install.sh`),
and `nsys` from `/usr/local/bin`.
