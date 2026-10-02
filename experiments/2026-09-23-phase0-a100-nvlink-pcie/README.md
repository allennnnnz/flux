# Phase 0: A100 single-node NVLink / PCIe baseline

Date: 2026-09-23
Machine: `css-host-158`, 8x A100-SXM4-80GB, NVSwitch, no sudo.

> **Phase 0 is closed.** The consolidated record is [`docs/PHASE0_STATUS.md`](../../docs/PHASE0_STATUS.md);
> the distilled conclusions and design rules are [`docs/PHASE0_FINDINGS.md`](../../docs/PHASE0_FINDINGS.md).
> This folder is the raw archive. New work goes in `ws/` — see [`PROJECT.md`](../../PROJECT.md).

## Question

Is there a case for giving Flux a second, PCIe-host-staged transport channel alongside
NVLink on this machine?

## Answer

**No.** Do not modify Flux core scheduling on A100. Two independent lines of evidence:

1. **Communication is not the bottleneck at the shape that matters.** Making Flux's
   AllGather 17% faster (Ring2D → All2All) changes end-to-end AG+GEMM time by 0.9% at
   N=49152. At smaller N (4096, 8192) it is worth about 10%, so communication *is* on the
   critical path there — but see 2.
2. **The second channel is ~3% of the first, and measured as a net loss.** Host staging
   delivers 5.3 GB/s per GPU against NVLink's 217.66 GB/s per GPU. Sweeping the mix ratio
   α over 0.01-0.12 produces a loss at *every* value, monotonically, with or without a
   switch-aware staging ring.

A dual-channel transport becomes interesting where the two paths are within a small factor
of each other — a machine without full NVLink, or a PCIe-only or heterogeneous accelerator.
Building a backend abstraction against that future remains reasonable; changing the A100
NVLink path does not.

## What this experiment actually produced

The first pass of this work was largely wrong. It went through four correction rounds, and
the corrections are the substance:

| Finding | Where |
| --- | --- |
| GPUs share PCIe switch uplinks **in pairs** (0/1, 2/3, 4/5, 6/7); confirmed by a control experiment | `B_REDO.md` B.4 |
| PCIe is **not symmetric full-duplex**: concurrent D2H throttles H2D by 3.4x, while D2H is unaffected | `BIDIR_CORRECTION.md` X.3 |
| Flux's own AllGather beats NCCL by **13-22%** in All2All mode — invisible in the original report, which used NCCL as the "Flux" baseline | `E_CORRECTION.md` E.1 |
| **Ring2D**, the mode every original measurement used, is among the slowest here; All2All is worth ~10% end-to-end at small N | `E_CORRECTION.md` E.1, `FLUX_BASELINE.md` F.5 |
| PyTorch runs cross-device copies on the **source** device's stream — placing them on the destination's stream costs 5.7x | `D_CORRECTION.md` D.0 |
| Host staging in a ring is bounded by the **contended** H2D rate (6.46 GB/s), and already achieves 91% of it | `BIDIR_CORRECTION.md` X.4 |

## Headline numbers

All corrected. See `docs/PHASE0_STATUS.md` section 2 for method and section 3 for what these replace.

| Quantity | Value |
| --- | ---: |
| NVLink all-to-all, per GPU | 217.66 GB/s |
| Flux AllGather All2All (pull), per GPU ingress | 188.2 GB/s |
| PCIe H2D / D2H, single GPU, unidirectional | 21.795 / 23.924 GB/s |
| PCIe H2D / D2H, single GPU, **simultaneous** | 6.16 / 22.12 GB/s |
| Host staging, best pair (different switches) | 22.191 GB/s |
| Host staging, 8-GPU ring, per GPU | 5.87 GB/s |
| AG+GEMM overlap efficiency, N=49152, bf16 | 60% |

## Documents

| File | Contents |
| --- | --- |
| **[`../../docs/PHASE0_STATUS.md`](../../docs/PHASE0_STATUS.md)** | **Start here.** Full record: numbers, withdrawal list, decision. |
| [`../../docs/PHASE0_FINDINGS.md`](../../docs/PHASE0_FINDINGS.md) | Distilled conclusions and design rules for later work. |
| [`AUDIT.md`](AUDIT.md) | First audit of the original pass. |
| [`B_REDO.md`](B_REDO.md) | PCIe topology and unidirectional bandwidth, auditable. |
| [`BIDIR_CORRECTION.md`](BIDIR_CORRECTION.md) | Bidirectional PCIe; extends the staging section. |
| [`D_CORRECTION.md`](D_CORRECTION.md) | Dual-path; the corrected NVLink baseline. |
| [`E_CORRECTION.md`](E_CORRECTION.md) | Flux's own communication path, measured. |
| [`FLUX_BASELINE.md`](FLUX_BASELINE.md) | AG+GEMM with a reproducible ECT method. |
| [`CDE_REPORT.md`](CDE_REPORT.md) | Superseded. Sections D and E withdrawn in place. |

## Reproducing

```bash
bash experiments/2026-09-23-phase0-a100-nvlink-pcie/scripts/run_phase0_repro.sh
```

The script reproduces only the currently-valid set. It notes inline which superseded
scripts it deliberately does not run.

Flux benchmarks must be launched through pixi:

```bash
pixi run --manifest-path pixi.toml ./launch.sh <script> ...
```

Invoking `./launch.sh` directly fails with `torchrun: command not found`.
