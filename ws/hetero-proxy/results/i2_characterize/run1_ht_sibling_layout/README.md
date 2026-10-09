# run1 (2026-10-09 07:47Z, guard CLEAN) — superseded, kept for audit

Backend layout at the time: device control threads (dispatcher, queue threads) on the hyper-thread siblings of the
compute cores. A waiting (spinning) queue thread halves its sibling compute core's throughput and the statically
scheduled OpenMP team waits for its slowest thread. Consequences in this run:

- section C ("compute while copies run") shows 1.6-3.6x slower compute; that factor mixes in the HT-sibling
  interference of the copy queues' waiting threads, it is not (only) memory-bandwidth contention;
- the I3 dry run then saw compute 3-4x slower inside the pipeline than section B alone.

Sections A (device DMA), B (compute alone, copy queues idle), D (GPU copies, window-restricted) and E are not
affected by the layout as far as known, but the whole run is superseded by the rerun in the parent directory
(control threads on 2 dedicated physical cores; compute siblings idle). Changes between the runs:
hxb/cpu_backend.py layout + flush(); hxb/csrc/hxbdma.c sfence moved into the worker threads.
