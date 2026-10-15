################################################################################
# hetero-proxy hetero_roles P1 (2026-10-15): preliminary test of the surveyed heterogeneous roles - what the CPU
# "device" (NUMA node 1) can contribute per role, before building any of them.
#   R   peak read bandwidth (fp32 sum, 2 GiB), 26 and 32 threads: the anchor for memory-bound roles
#   T1  decode attention (Lamina-style offload): Qwen2.5-32B layer shapes (8 KV heads x 5 q heads, D 128), KV in
#       bf16 in device memory; contexts 4k / 32k, batch 1 / 16; per-layer time -> KV GB/s
#   T2  one MoE expert computed where it lives (Fiddler / KTransformers): SwiGLU expert shapes of Qwen3-30B-A3B
#       (2048 x 768), DeepSeek-V2-Lite (2048 x 1408), Mixtral-8x7B (4096 x 14336); n = 1 / 4 / 16 / 64 tokens
# C / OpenMP kernels in hxb/csrc/hxbdma.c (bf16 weights / KV, fp32 math). Correctness vs torch fp32 on small cases.
# Usage: numactl --physcpubind=32-63 --membind=1 pixi run --manifest-path pixi.toml python p1_roles_cpu_v1.py --out <dir>
################################################################################
import argparse
import ctypes
import json
import math
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from hxb import native  # noqa: E402

L = native.lib()
EXPERTS = {"qwen3-30b-a3b": (2048, 768), "deepseek-v2-lite": (2048, 1408), "mixtral-8x7b": (4096, 14336)}


def timeit(fn, reps):
    fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def attn(q, K, V, lens, out, scratch, nt, scale):
    B, NKV, G, D = q.shape
    L.hxb_attn_decode(q.data_ptr(), K.data_ptr(), V.data_ptr(), lens.data_ptr(), out.data_ptr(), B, NKV, G, D,
                      K.shape[2], scale, nt, scratch.data_ptr())


def expert(x, Wg, Wu, Wd, out, act, nt):
    L.hxb_moe_expert(x.data_ptr(), Wg.data_ptr(), Wu.data_ptr(), Wd.data_ptr(), out.data_ptr(), act.data_ptr(),
                     x.shape[0], Wg.shape[1], Wg.shape[0], nt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    res = {"affinity": sorted(os.sched_getaffinity(0))}
    log = open(os.path.join(args.out, "p1_cpu_log.txt"), "w")

    def say(s):
        print(s, flush=True)
        log.write(s + "\n")
    g = torch.Generator().manual_seed(0)

    # R
    a = torch.rand(512 * 2 ** 20, generator=g)  # 2 GiB fp32
    for nt in (26, 32):
        t = timeit(lambda: L.hxb_read_sum(a.data_ptr(), a.numel(), nt), args.reps)
        res[f"R_read_GBps_nt{nt}"] = a.numel() * 4 / t / 1e9
        say(f"[R ] read bandwidth, {nt} threads: {res[f'R_read_GBps_nt{nt}']:7.1f} GB/s")
    del a

    # T1 correctness (small) then timing
    NKV, G, D = 8, 5, 128
    scale = 1 / math.sqrt(D)
    q = torch.randn(2, NKV, G, D, generator=g)
    K = torch.randn(2, NKV, 300, D, generator=g).to(torch.bfloat16)
    V = torch.randn(2, NKV, 300, D, generator=g).to(torch.bfloat16)
    lens = torch.tensor([300, 177], dtype=torch.int32)
    out = torch.empty_like(q)
    scr = torch.empty(32 * G * 300)
    attn(q, K, V, lens, out, scr, 26, scale)
    err = 0.0
    for b in range(2):
        n = int(lens[b])
        s = torch.einsum("hgd,hld->hgl", q[b], K[b, :, :n].float()) * scale
        ref = torch.einsum("hgl,hld->hgd", s.softmax(-1), V[b, :, :n].float())
        err = max(err, (out[b] - ref).abs().max().item())
    say(f"[T1] attention kernel vs torch fp32: max |diff| {err:.2e}")
    res["T1_maxdiff"] = err
    for Lc in (4096, 32768):
        for B in (1, 16):
            q = torch.randn(B, NKV, G, D, generator=g)
            K = torch.randn(B, NKV, Lc, D, generator=g).to(torch.bfloat16)
            V = torch.randn(B, NKV, Lc, D, generator=g).to(torch.bfloat16)
            lens = torch.full((B,), Lc, dtype=torch.int32)
            out = torch.empty_like(q)
            scr = torch.empty(32 * G * Lc)
            t = timeit(lambda: attn(q, K, V, lens, out, scr, 26, scale), args.reps)
            kv = 2 * B * NKV * Lc * D * 2
            res[f"T1_L{Lc}_B{B}"] = dict(ms_per_layer=t * 1e3, kv_GBps=kv / t / 1e9)
            say(f"[T1] decode attention L={Lc:<6} B={B:<3}: {t * 1e3:8.2f} ms per layer, KV read {kv / t / 1e9:6.1f} GB/s")
            del K, V

    # T2 correctness (small) then timing
    Hs, Is = 256, 96
    Wg = (torch.randn(Is, Hs, generator=g) * 0.05).to(torch.bfloat16)
    Wu = (torch.randn(Is, Hs, generator=g) * 0.05).to(torch.bfloat16)
    Wd = (torch.randn(Hs, Is, generator=g) * 0.05).to(torch.bfloat16)
    x = torch.randn(5, Hs, generator=g)
    out, act = torch.empty(5, Hs), torch.empty(5, Is)
    expert(x, Wg, Wu, Wd, out, act, 26)
    ref = (torch.nn.functional.silu(x @ Wg.float().t()) * (x @ Wu.float().t())) @ Wd.float().t()
    err = (out - ref).abs().max().item() / ref.abs().max().item()
    say(f"[T2] expert kernel vs torch fp32: max rel diff {err:.2e}")
    res["T2_maxreldiff"] = err
    # Each call uses the NEXT expert of a pool >= 1.5 GB (cold weights, as in real routing: consecutive calls hit
    # different experts). v1 reused one expert: it stayed in the 48 MB L3 and read "221 GB/s" > the DRAM anchor.
    for name, (H, I) in EXPERTS.items():
        wbytes = 3 * H * I * 2
        E = max(4, int(1.5e9 // wbytes) + 1)
        pool = [tuple((torch.randn(*shp, generator=g) * 0.02).to(torch.bfloat16) for shp in ((I, H), (I, H), (H, I)))
                for _ in range(E)]
        for n in (1, 4, 16, 64):
            x = torch.randn(n, H, generator=g)
            out, act = torch.empty(n, H), torch.empty(n, I)
            k = [0]

            def call():
                Wg, Wu, Wd = pool[k[0] % E]
                k[0] += 1
                expert(x, Wg, Wu, Wd, out, act, 26)
            ts = []
            for _ in range(2 * E):
                t0 = time.perf_counter()
                call()
                ts.append(time.perf_counter() - t0)
            t = statistics.median(ts[E:])
            res[f"T2_{name}_n{n}"] = dict(ms=t * 1e3, weight_MB=wbytes / 1e6, weight_GBps=wbytes / t / 1e9, pool=E)
            say(f"[T2] expert {name:<17} ({wbytes / 1e6:6.1f} MB, pool {E:3d}) n={n:<3}: {t * 1e3:8.3f} ms  "
                f"(weights read at {wbytes / t / 1e9:6.1f} GB/s)")
        del pool
    with open(os.path.join(args.out, "p1_cpu.json"), "w") as f:
        json.dump(res, f, indent=1)
    log.close()


if __name__ == "__main__":
    main()
