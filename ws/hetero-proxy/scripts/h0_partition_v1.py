################################################################################
# hetero-proxy H0 (2026-10-09): what role can ONE PCIe-only chip play next to an 8x A100 NVLink TP domain?
# Analytic model (no GPU). Every number is an estimate [推論]; inputs and assumptions are printed.
# Inputs
#   t_gpu(M)  per-layer time of the 8x A100 TP8 domain, Qwen2.5-32B, best measured policy (G4 block oracle,
#             ws/fusion-dispatch/results/g4_block_oracle/, 4 blocks incl. attention, decode = CUDA graph)
#   PCIe      22.2 GB/s best one-way staging (PHASE0_FINDINGS 2.5); 6.46 GB/s H2D when the same GPU also sends
#             (2.4); per-transfer fixed cost ALPHA (assumption until hetero-proxy 1b measures it)
#   s         chip compute speed relative to ONE A100; a chip running an unsharded layer needs ~8 x t_gpu / s
#             (TP8 per-rank work x 8; counts the domain's comm time as compute, i.e. pessimistic for the chip)
# Roles
#   TP   chip is an extra tensor-parallel member (any share): per layer it must receive the layer input twice
#        (QKV, gate_up) and return its partial sums twice (O, down) -> 4 x M x H x 2 bytes over PCIe
#   PP   chip is a pipeline stage holding k layers: per pass it receives and returns one M x H activation;
#        k balanced so both stages take equal time -> throughput gain s / 8, decode latency per token grows
#   ATT  chip runs decode attention with the KV cache (Lamina-style): per layer q, k, v in and attention out
#   DRAFT chip runs a speculative-decoding draft model: token ids / logits only (negligible PCIe)
# Usage: python3 h0_partition_v1.py   (writes results/h0_partition/h0_partition.csv and h0_partition_log.txt)
################################################################################
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "results", "h0_partition")
H, LAYERS, NKV, HD = 5120, 64, 8, 128
B = 2
PCIE, PCIE_BIDIR, ALPHA = 22.2e9, 6.46e9, 20e-6
T_GPU = {32: 0.7168, 128: 1.1064, 256: 1.7603, 384: 2.2344, 512: 2.5702,  # decode, ms per 4 blocks (auto_g4 choice)
         1024: 3.6275, 2048: 6.2121, 4096: 11.3562}                       # prefill
SPEEDS = [1 / 30, 1 / 8, 1 / 4, 1 / 2, 1.0]
MS = [32, 256, 1024, 4096]


def main():
    os.makedirs(OUT, exist_ok=True)
    log = open(os.path.join(OUT, "h0_partition_log.txt"), "w")
    rows = []

    def say(s=""):
        print(s)
        log.write(s + "\n")
    say("H0 inputs: Qwen2.5-32B (H=5120, 64 layers), 8x A100 TP8 domain, PCIe 22.2 GB/s one-way, 6.46 GB/s H2D under")
    say(f"           bidirectional load, ALPHA={ALPHA * 1e6:.0f} us per transfer (assumption), chip speed s = x one A100\n")
    say("[TP] chip as an extra TP member: PCIe time per layer vs the domain's whole layer time")
    for M in MS:
        t = T_GPU[M] / 4
        nb = 4 * M * H * B
        best, worst = nb / PCIE * 1e3 + 4 * ALPHA * 1e3, nb / PCIE_BIDIR * 1e3 + 4 * ALPHA * 1e3
        say(f"  M={M:<5} PCIe {nb / 1e6:6.1f} MB/layer -> {best:6.3f} .. {worst:6.3f} ms   domain layer {t:6.3f} ms   "
            f"PCIe/layer = {best / t:4.1f}x .. {worst / t:4.1f}x  (any chip speed)")
        rows.append(["TP", M, "", f"{best:.4f}", f"{worst:.4f}", f"{t:.4f}", "", "", ""])
    say("  -> the chip's path alone is longer than the domain's layer: TP membership slows every layer, whatever s is.\n")

    say("[PP] chip as a pipeline stage (balanced k layers); throughput gain = s/8; PCIe per pass vs stage time")
    for s in SPEEDS:
        k = LAYERS / (1 + 8 / s)
        for M in MS:
            t = T_GPU[M] / 4
            stage = (LAYERS - k) * t
            tx = 2 * M * H * B / PCIE * 1e3 + 2 * ALPHA * 1e3
            gain = min(LAYERS * t / max(stage, tx), LAYERS * t / stage) - 1
            lat = (stage + k * 8 * t / s + tx) / (LAYERS * t)
            if M in (32, 4096):
                say(f"  s={s:5.3f} M={M:<5} chip layers {k:5.1f}  stage {stage:8.2f} ms  boundary {tx:6.3f} ms "
                    f"({tx / stage * 100:5.2f}% of stage)  throughput +{gain * 100:4.1f}%  single-pass latency x{lat:4.2f}")
            rows.append(["PP", M, f"{s:.4f}", f"{tx:.4f}", "", f"{stage:.4f}", f"{k:.2f}", f"{gain:.4f}", f"{lat:.3f}"])
    say("  -> PCIe is a small fraction of a stage; gain is capped by the chip's compute (s/8: +12.5% for an A100-class chip,")
    say("     +3.1% at s=1/4, +0.4% for a CPU); the price is longer single-pass (decode) latency unless requests are pipelined.\n")

    say("[ATT] chip runs decode attention (q, k, v in; output back) per layer")
    for M in (32, 256):
        t = T_GPU[M] / 4
        nb = M * (H + 2 * NKV * HD + H) * B
        tx = nb / PCIE * 1e3 + 2 * ALPHA * 1e3
        say(f"  M={M:<5} PCIe {nb / 1e6:5.2f} MB/layer -> {tx:6.3f} ms  vs domain layer {t:6.3f} ms ({tx / t * 100:5.1f}%)")
        rows.append(["ATT", M, "", f"{tx:.4f}", "", f"{t:.4f}", "", "", ""])
    say("  -> a large share of every layer; only workable with cross-layer / two-batch pipelining (Lamina); the gain is mainly")
    say("     KV-cache capacity, not compute.\n")
    say("[DRAFT] speculative-decoding draft on the chip: PCIe carries token ids / logits only -> link is not the issue, and")
    say("        there is no large transfer to overlap; the gain comes from running the draft in parallel with the target.\n")
    say("Conclusion [推論]: first real role = pipeline stage (PCIe-feasible, gain s/8). Flux-style chunk overlap applies to the")
    say("stage-boundary transfers (next stage's first GEMM starts on arriving row chunks). TP membership is excluded.")
    with open(os.path.join(OUT, "h0_partition.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["role", "M", "chip_speed", "pcie_ms_best", "pcie_ms_worst", "domain_ms", "pp_chip_layers",
                    "pp_throughput_gain", "pp_latency_ratio"])
        w.writerows(rows)
    log.close()


if __name__ == "__main__":
    main()
