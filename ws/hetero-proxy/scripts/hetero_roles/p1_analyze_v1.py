################################################################################
# hetero-proxy hetero_roles P1 (2026-10-15): which surveyed role can the CPU "device" actually contribute to?
# Reads results/hetero_roles/p1/{p1_cpu.json, p1_gpu.json} (+ fixed anchors below, each with its source) and prints:
#   T1 attention offload: the share of decode attention the device can take next to the 8-GPU domain at equal step
#      time (bandwidth-balanced: f* = BW_dev / (BW_dev + 8 BW_gpu)) -> the throughput gain bound
#   T2 MoE expert offload: per expert call, compute where the weights live (device) vs swap the weights to the GPU
#      over PCIe and compute there; + tokens per expert at decode batch sizes (n = B * top_k / n_experts)
#   T3 KV-cache tier (estimate from measured parts, no end-to-end run yet): restore a reused prefix's KV from host
#      memory vs recompute its prefill
#   T4 P/D with the device doing decode (from the E2E measurements)
# Usage: python3 p1_analyze_v1.py results/hetero_roles/p1
################################################################################
import json
import os
import sys

d = sys.argv[1]
cpu = json.load(open(os.path.join(d, "p1_cpu.json")))
gpu = json.load(open(os.path.join(d, "p1_gpu.json")))
out = []


def say(s=""):
    print(s)
    out.append(s)


say("T1 attention offload (Lamina-style), Qwen2.5-32B layer shapes, KV read rate")
for k in ("T1_L4096_B16", "T1_L32768_B1", "T1_L32768_B16"):
    c, g = cpu[k]["kv_GBps"], gpu[k]["kv_GBps"]
    f = c / (c + 8 * g)
    say(f"  {k[3:]:<12} device {c:6.1f} GB/s  one A100 {g:7.1f} GB/s  -> device share next to 8 GPUs {100 * f:4.2f}%"
        f"  (throughput gain bound ~{100 * f / (1 - f):4.2f}%)")
bw = cpu["R_read_GBps_nt26"]
g = gpu["T1_L32768_B16"]["kv_GBps"]
say(f"  even at the device's full read bandwidth ({bw:.0f} GB/s): share {100 * bw / (bw + 8 * g):4.2f}%")
say("")
say("T2 MoE expert offload (Fiddler / KTransformers-style): one expert call, weights NOT on the GPU")
topk = {"qwen3-30b-a3b": (8, 128), "deepseek-v2-lite": (6, 64), "mixtral-8x7b": (2, 8)}
for name, (k_, e_) in topk.items():
    h2d = gpu[f"T2_{name}_h2d"]["ms"]
    for n in (1, 4, 16, 64):
        dev = cpu[f"T2_{name}_n{n}"]["ms"]
        swap = h2d + gpu[f"T2_{name}_gpu_n{n}"]["ms"]
        say(f"  {name:<17} n={n:<3} compute on device {dev:7.3f} ms | swap to GPU + compute {swap:7.3f} ms -> "
            f"{'device' if dev < swap else 'swap  '} faster x{max(dev, swap) / min(dev, swap):4.1f}")
    say(f"    decode batch B -> tokens per expert n = B x {k_} / {e_}: B=16 -> {16 * k_ / e_:.1f}, "
        f"B=64 -> {64 * k_ / e_:.1f}, B=256 -> {256 * k_ / e_:.1f}")
say("  (activations crossing PCIe per call: n x hidden x 2 bytes each way, KB-scale; hop latency ~0.05-0.3 ms "
    "measured in E2E B, not included above)")
say("")
say("T3 KV-cache tier (estimate): Qwen2.5-32B TP8, KV per GPU per token = 64 layers x 1 KV head x 128 x 2 x 2 B = 32 KiB")
h2d_8gpu = 10.0  # GB/s per GPU with all 8 GPUs copying at once (results/e2e/c_diag, b_h2d_all8, 2026-10-14)
prefill_8k = 397.7  # ms, 8 x 1024 tokens, engine v5 config A (results/e2e/bench_v5, 2026-10-14)
for P in (2048, 8192, 32768):
    nb = 32 * 1024 * P
    say(f"  prefix {P:>6} tokens: restore {nb / 1e6:7.1f} MB per GPU at {h2d_8gpu:.0f} GB/s = {nb / h2d_8gpu / 1e6:6.1f} ms"
        + (f"  vs recompute ~{prefill_8k:.0f} ms (8 x 1024 measured) -> ~x{prefill_8k / (nb / h2d_8gpu / 1e6):.0f}"
           if P == 8192 else ""))
say("")
say("T4 P/D with the device doing decode: one Qwen2.5-32B layer on the CPU device, decode B=1 13 ms, B=32 109 ms")
say("  (results/e2e/cpu_layer) -> 64 layers ~0.8 s / token at B=1 vs the 8-GPU domain 14.8 ms (bench_v5):")
say(f"  the device adds ~{100 * 14.8 / 830:.1f}% decode capacity at B=1 and less at larger batches")
open(os.path.join(d, "analysis.txt"), "w").write("\n".join(out) + "\n")
