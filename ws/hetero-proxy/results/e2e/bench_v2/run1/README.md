# bench_v2 run1 (2026-10-14 09:31-09:54Z, guard CLEAN, 20 rounds) — valid measurement, two open questions

Code: scripts/e2e/qwen_tp_v2.py @ 341b8b5 (configurations A, Bfinal, Bmid8, Bmid1).
1. Bmid1 prefill adds 87 ms (8x1024) / 151 ms (4x4096) — far above the pre-registered +2..+6%. Suspected cause: the
   CPU backend's rmsnorm op allocates several full-size temporaries (pow, cast) per call -> page faults for 80-160 MiB
   inputs; i.e. the CPU implementation, not the PCIe channel. To be checked.
2. Bmid* decode is FASTER than A (b1 -4.2%): A is one CUDA graph per step, Bmid two graphs (split at layer 33). The
   graph split itself may change the step time; a control "A split at the same place, GPU norm, no hop" is needed.
Also: this engine's decode is slow in absolute terms (A b32 42.9 ms/token vs fusion-dispatch G4 best ~11.5 ms), so
percentages understate the hop; report absolute added ms as well.
