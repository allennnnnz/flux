################################################################################
# hetero-proxy hetero_roles P1 (2026-10-15): the GPU side of the preliminary role tests (one A100, GPU0).
#   T1  decode attention with the engine's split-K kernel (e2e/fused_kernels.decode_attn_split), the same layer shapes
#       as the CPU test (8 KV heads x 5 q heads, D 128), contexts 4k / 32k, batch 1 / 16; per-call time inside a
#       CUDA graph -> KV GB/s
#   T2  per MoE expert shape: (a) H2D of the expert's weights from registered pinned memory (what "swap the expert to
#       the GPU" costs), (b) the expert computed on the GPU with resident weights (torch bf16), n = 1 / 4 / 16 / 64
# Usage: CUDA_MODULE_LOADING=EAGER pixi run --manifest-path pixi.toml python p1_roles_gpu_v1.py --out <dir>
################################################################################
import argparse
import json
import math
import os
import statistics
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "e2e"))
from hxb import cudamem as cm  # noqa: E402
from fused_kernels import decode_attn_split  # noqa: E402

EXPERTS = {"qwen3-30b-a3b": (2048, 768), "deepseek-v2-lite": (2048, 1408), "mixtral-8x7b": (4096, 14336)}


def graph_time(fn, n=20):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
        fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n):
            fn()
    g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(5):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) / n)
    return statistics.median(ts)


def event_time(fn, reps=20):
    fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b))
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.cuda.set_device(0)
    dev = torch.device("cuda:0")
    res = {}
    log = open(os.path.join(args.out, "p1_gpu_log.txt"), "w")

    def say(s):
        print(s, flush=True)
        log.write(s + "\n")
    NKV, G, D = 8, 5, 128
    for Lc in (4096, 32768):
        for B in (1, 16):
            q = torch.randn(B, NKV * G, D, device=dev).to(torch.bfloat16)
            kc = torch.randn(B, NKV, Lc, D, device=dev).to(torch.bfloat16)
            vc = torch.randn_like(kc)
            pos = torch.full((B,), Lc - 1, dtype=torch.long, device=dev)
            scr = tuple(torch.empty(n, device=dev) for n in (B * NKV * 8 * 16, B * NKV * 8 * 16, B * NKV * 8 * 16 * D))
            ms = graph_time(lambda: decode_attn_split(q, kc, vc, pos, 1 / math.sqrt(D), scratch=scr))
            kv = 2 * B * NKV * Lc * D * 2
            res[f"T1_L{Lc}_B{B}"] = dict(ms_per_layer=ms, kv_GBps=kv / (ms / 1e3) / 1e9)
            say(f"[T1-GPU] decode attention L={Lc:<6} B={B:<3}: {ms:8.4f} ms per layer, KV read "
                f"{kv / (ms / 1e3) / 1e9:7.1f} GB/s")
            del kc, vc
    for name, (H, I) in EXPERTS.items():
        nb = 3 * H * I * 2
        host = torch.empty(nb // 2, dtype=torch.bfloat16).share_memory_()
        cm.register(host.data_ptr(), nb)
        devw = torch.empty(nb // 2, dtype=torch.bfloat16, device=dev)
        st = torch.cuda.current_stream()
        ms = event_time(lambda: cm.memcpy(st, devw.data_ptr(), host.data_ptr(), nb))
        res[f"T2_{name}_h2d"] = dict(ms=ms, GBps=nb / (ms / 1e3) / 1e9)
        say(f"[T2-GPU] expert {name:<17} weights {nb / 1e6:6.1f} MB H2D: {ms:8.3f} ms ({nb / (ms / 1e3) / 1e9:5.1f} GB/s)")
        Wg = torch.randn(I, H, device=dev).to(torch.bfloat16)
        Wu = torch.randn(I, H, device=dev).to(torch.bfloat16)
        Wd = torch.randn(H, I, device=dev).to(torch.bfloat16)
        for n in (1, 4, 16, 64):
            x = torch.randn(n, H, device=dev).to(torch.bfloat16)
            ms = graph_time(lambda: F.linear(F.silu(F.linear(x, Wg)) * F.linear(x, Wu), Wd))
            res[f"T2_{name}_gpu_n{n}"] = dict(ms=ms)
            say(f"[T2-GPU] expert {name:<17} n={n:<3} resident compute: {ms:8.4f} ms")
        cm.unregister(host.data_ptr())
        del Wg, Wu, Wd, devw, host
    with open(os.path.join(args.out, "p1_gpu.json"), "w") as f:
        json.dump(res, f, indent=1)
    log.close()


if __name__ == "__main__":
    main()
