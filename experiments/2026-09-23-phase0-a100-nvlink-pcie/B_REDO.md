# Phase 0 B Redo

Date: 2026-09-23

This replaces the previous unauditable Phase 0 B numbers.

## B.1 PCIe switch upstream links

Tool: direct sysfs reads from
`/sys/bus/pci/devices/<BDF>/{current_link_speed,current_link_width,max_link_speed,max_link_width}`.

Root ports:

| BDF | current speed | current width | max speed | max width |
| --- | --- | --- | --- | --- |
| 0000:09:02.0 | 16.0 GT/s PCIe | x16 | 16.0 GT/s PCIe | x16 |
| 0000:42:02.0 | 16.0 GT/s PCIe | x16 | 16.0 GT/s PCIe | x16 |
| 0000:83:02.0 | 16.0 GT/s PCIe | x16 | 16.0 GT/s PCIe | x16 |
| 0000:bf:02.0 | 16.0 GT/s PCIe | x16 | 16.0 GT/s PCIe | x16 |

Switch upstream ports checked from the `lspci -tv` topology:

| BDF | current speed | current width | max speed | max width |
| --- | --- | --- | --- | --- |
| 0000:0a:00.0 | 16.0 GT/s PCIe | x16 | 16.0 GT/s PCIe | x16 |
| 0000:43:00.0 | 16.0 GT/s PCIe | x16 | 16.0 GT/s PCIe | x16 |
| 0000:84:00.0 | 16.0 GT/s PCIe | x16 | 16.0 GT/s PCIe | x16 |
| 0000:c0:00.0 | 16.0 GT/s PCIe | x16 | 16.0 GT/s PCIe | x16 |

Verdict: each root port and first switch upstream link is running at PCIe Gen4 x16.

## B.2 Method

Script: `scripts/global_bandwidth_v2.py`.

All benchmark rows below used:

- Direction: H2D or D2H as listed.
- Size: 1024 MiB per GPU per copy.
- Copies inside one measured interval: 20.
- Warmup: 5 measured-size iterations, discarded.
- Formal repeats: 20.
- Wall-time bandwidth: `num_gpus * 1024 MiB * 20 / wall_seconds`.
- CUDA event timing: each process records CUDA event elapsed time around its own 20 copies.
- Start skew: max minus min of worker `time.perf_counter()` start timestamps after the global barrier.
- NUMA local: GPU0-3 bound to NUMA 0, GPU4-7 bound to NUMA 1.
- NUMA remote: GPU0-3 bound to NUMA 1, GPU4-7 bound to NUMA 0.

The script sets CPU affinity from `/sys/devices/system/node/node*/cpulist` and uses `libnuma`
`numa_run_on_node()` plus `numa_set_membind()` before importing torch and allocating pinned host memory.

Raw outputs are in `results/b_redo/*.csv`.

## B.3 Local vs remote NUMA, 8 GPUs

| Direction | NUMA mode | Aggregate median GB/s | Aggregate stdev GB/s | Per-GPU median GB/s | Per-GPU stdev GB/s | Max start skew us | Median CUDA ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| H2D | local | 101.173 | 0.009 | 12.647 | 0.001 | 135.720 | 1695.417 |
| H2D | remote | 82.415 | 0.174 | 10.302 | 0.022 | 122.037 | 2030.672 |
| D2H | local | 105.560 | 0.064 | 13.195 | 0.008 | 92.945 | 1614.350 |
| D2H | remote | 75.583 | 0.030 | 9.448 | 0.004 | 99.689 | 2258.084 |

Verdict: local NUMA binding is materially faster than remote NUMA binding for 8-GPU PCIe traffic.

## B.4 Shared-upstream control experiment

All rows use local NUMA binding.

### H2D

| GPUs | Topology | Aggregate median GB/s | Aggregate stdev GB/s | Per-GPU median GB/s | Max start skew us | Median CUDA ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | Single GPU | 21.795 | 0.012 | 21.795 | 0.000 | 985.152 |
| 0,1 | Same switch | 25.467 | 0.001 | 12.734 | 53.532 | 1686.073 |
| 0,2 | Different switches, same NUMA | 42.903 | 0.027 | 21.452 | 39.997 | 1000.272 |
| 0,2,4,6 | Four different switches | 85.119 | 0.100 | 21.280 | 57.749 | 1004.649 |
| 0-7 | All GPUs | 101.173 | 0.009 | 12.647 | 135.720 | 1695.417 |

### D2H

| GPUs | Topology | Aggregate median GB/s | Aggregate stdev GB/s | Per-GPU median GB/s | Max start skew us | Median CUDA ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | Single GPU | 23.924 | 0.046 | 23.924 | 0.000 | 897.481 |
| 0,1 | Same switch | 26.600 | 0.002 | 13.300 | 28.678 | 1614.508 |
| 0,2 | Different switches, same NUMA | 46.638 | 0.006 | 23.319 | 37.195 | 916.732 |
| 0,2,4,6 | Four different switches | 93.072 | 0.007 | 23.268 | 61.736 | 919.234 |
| 0-7 | All GPUs | 105.560 | 0.064 | 13.195 | 92.945 | 1614.350 |

Verdict: the decisive control supports the shared-upstream hypothesis. GPU0+GPU1
on the same switch is close to one Gen4 x16 uplink worth of aggregate bandwidth,
while GPU0+GPU2 on different switches is close to twice the single-GPU bandwidth.

## B.5 Replacement single-GPU bandwidth

Tool: `scripts/global_bandwidth_v2.py`, local NUMA binding, GPU0 only.

Parameters: 1024 MiB per copy, 20 copies per measured interval, 5 warmup intervals discarded,
20 formal repeats, wall-time aggregate bandwidth. Since this is one GPU, aggregate equals per-GPU.

| Direction | Median GB/s | Stdev GB/s | Median CUDA ms |
| --- | ---: | ---: | ---: |
| H2D | 21.795 | 0.012 | 985.152 |
| D2H | 23.924 | 0.046 | 897.481 |

Verdict: the previous B.5 numbers are replaced by these auditable single-GPU results.

## Implication for later D modeling

For 8 GPUs using PCIe concurrently with local NUMA binding, use these per-GPU medians:

- H2D: 12.647 GB/s per GPU.
- D2H: 13.195 GB/s per GPU.

Do not use the old unauditable 21.6-22.6 GB/s single-GPU value for the 8-GPU concurrent model.
