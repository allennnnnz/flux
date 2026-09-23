# Phase 0 Report Audit

Date: 2026-09-23

The original aggregate H2D/D2H values are withdrawn because the global timing
method and raw output were not preserved.

## A. PCIe topology

The old report answered the PCIe-switch question with `nvidia-smi topo -m` and
`NV12`. That was wrong: `NV12` describes GPU-to-GPU NVLink/NVSwitch paths, not
PCIe switch sharing. **判定：不成立。**

The old report also said that a second node was not required. Phase 0 did not
ask that question and no dual-node experiment was performed. That statement is
withdrawn. **判定：不成立。**

Relevant `lspci -tv` tree fragments:

```text
09:02.0-[0a-23]
  ... 0b:00.0 ... 10:00.0 NVIDIA GA100 (GPU0)
  ... 0b:04.0 ... 16:00.0 NVIDIA GA100 (GPU1)
42:02.0-[43-65]
  ... 44:00.0 ... 49:00.0 NVIDIA GA100 (GPU2)
  ... 44:00.0 ... 4d:00.0 NVIDIA GA100 (GPU3)
83:02.0-[84-9c]
  ... 85:00.0 ... 8a:00.0 NVIDIA GA100 (GPU4)
  ... 85:04.0 ... 8f:00.0 NVIDIA GA100 (GPU5)
bf:02.0-[c0-d8]
  ... c1:00.0 ... c6:00.0 NVIDIA GA100 (GPU6)
  ... c1:00.0 ... ca:00.0 NVIDIA GA100 (GPU7)
```

Sysfs reported, for all eight GPU endpoints, `current_link_speed=16.0 GT/s
PCIe`, `current_link_width=16`, `max_link_speed=16.0 GT/s PCIe`, and
`max_link_width=16`. Thus the GPU endpoint links are PCIe Gen4 x16. `lspci`
identifies the relevant Broadcom devices as `PEX880xx PCIe Gen 4 Switch`, but
`lspci -vv` returned `Capabilities: <access denied>` for GPU and bridge
entries. The Gen/lane width of each internal switch upstream link is therefore
**無法確認**. **判定：需重測。**

## B. Eight-GPU host-transfer bandwidth

The old report did not preserve whether eight tests overlapped, so its 162.5
and 172.8 GB/s values are not auditable. **判定（舊方法）：無法確認。**

The new `scripts/global_bandwidth.py` uses eight independent Python
`multiprocessing.spawn` processes, one per GPU, a barrier before each copy, and
a second barrier after each CUDA stream synchronizes. It computes:

```text
aggregate_GBps = (8 * bytes_per_process) / global_wall_seconds / 1e9
per_gpu_GBps   = aggregate_GBps / 8
```

The old 32 MiB test was a single size, not a shmoo; the new script accepts
multiple sizes. The old aggregate command did not preserve per-process NUMA
binding. **判定（舊方法）：不成立。**

All rows below used 8 processes, GPU0-7, 64 MiB per GPU, and 5 repeats. The
aggregate is total payload divided by one global wall-clock interval; it is not
a sum of process-reported rates.

| Binding | Direction | Wall seconds, repeats 0-4 | Aggregate GB/s, repeats 0-4 | Derived per-GPU GB/s |
| --- | --- | --- | --- | --- |
| default | H2D | 0.009905, 0.007106, 0.006694, 0.006652, 0.010621 | 54.202, 75.556, 80.203, 80.709, 50.548 | 6.775, 9.444, 10.025, 10.089, 6.318 |
| default | D2H | 0.007129, 0.007237, 0.007082, 0.007069, 0.006993 | 75.306, 74.184, 75.813, 75.950, 76.767 | 9.413, 9.273, 9.477, 9.494, 9.596 |
| NUMA 0 | H2D | 0.026089, 0.019349, 0.007082, 0.007158, 0.006970 | 20.578, 27.747, 75.805, 75.007, 77.028 | 2.572, 3.468, 9.476, 9.376, 9.628 |
| NUMA 1 | H2D | 0.006754, 0.006627, 0.006822, 0.006823, 0.006714 | 79.495, 81.018, 78.701, 78.680, 79.960 | 9.937, 10.127, 9.838, 9.835, 9.995 |
| NUMA 0 | D2H | 0.007131, 0.009540, 0.009134, 0.015138, 0.015103 | 75.286, 56.276, 58.775, 35.466, 35.548 | 9.411, 7.034, 7.347, 4.433, 4.444 |
| NUMA 1 | D2H | 0.013715, 0.013058, 0.008439, 0.007930, 0.007791 | 39.144, 41.115, 63.619, 67.705, 68.907 | 4.893, 5.139, 7.952, 8.463, 8.613 |

The derived per-GPU values are arithmetic averages, not independent per-GPU
measurements. The result is variable and needs more controlled repetition and
placement before using a single bandwidth number. **判定：需重測。**

## B.5 Single-GPU H2D claim

The old report gives 21.6-22.6 GB/s, but did not record the original
`bandwidthTest` command, exact mode/size, repeat count, or raw output. I cannot
establish whether it was one GPU at a time or an aggregation. **判定：無法確認。**
