# Development runs (2026-10-09) — NOT measurements

Copied from the session scratchpad because code comments and the JOURNAL cite them as the reason for a fix.
None ran under exclusive_guard_v2, most used 3-5 rounds, and several used code that was later changed.
Do not quote these numbers as results; the measurements are in ../ (I3 official run) and ../../i2_characterize_v2/.

| file | what it showed | fix it led to |
| --- | --- | --- |
| smoke_pipe_log.txt | first full pass works (rel err 4.3e-3), level 1 not faster | characterisation (device contention) |
| char_dry_log.txt | v1 section D without window restriction: same-GPU H2D ~19-21 GB/s (wrong: uncontended tail) | window-restricted D (CLAUDE.md 5.1.7); Phase 0 6.46 confirmed |
| char_dry2_*_log.txt | Python thread-pool DMA ~0.2 ms per copy; GOMP spin count effect | C DMA engine (csrc/hxbdma.c), GOMP_SPINCOUNT=1e6 |
| char_dma0_log.txt | DMA threads on NUMA 0 reduce contention only slightly | kept DMA on the device node; `dma_cores` option |
| i3_dry_log.txt | large-n slowdown, cpu compute 3-4x slower in pipeline | flush() before the gate; control cores off the compute siblings |
| i3_dry2/3_log.txt | device copies ~2x slower than v1 section A (cache-hot) | characterisation v2 (GPU-staged data, shared engine) |
| i3_dry4/5_log.txt | level-0 cpu compute 28.7 ms vs 7.7 alone; clock filter dropped idle-GPU rounds | grow-only scratch (no per-shape realloc); max-clock filter |
