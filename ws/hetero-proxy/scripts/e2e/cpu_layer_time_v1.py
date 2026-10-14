################################################################################
# hetero-proxy E2E (2026-10-14): how long does ONE real Qwen2.5-32B decoder layer take on the CPU "chip"?
# Checks the inference in STATUS section 0 (whole layers on the CPU would be slower than the 8-GPU domain running the
# whole model: prefill 4096 tokens ~2 s per layer vs ~0.18 s, decode B=32 ~10-20 ms per layer vs ~11.5 ms).
# Real layer-0 weights, fp32 (no AVX512_BF16 / AMX on the Xeon 8358), the same layer math as qwen_tp_v1.py at
# world size 1. Run pinned to NUMA node 1 as the CPU backend's compute cores (numactl, 26 threads).
#   prefill: B=1, T in {1024, 4096}, causal attention; decode: B in {1, 32}, one token, KV cache of 512 tokens.
# Usage: numactl --cpunodebind=1 --membind=1 pixi run --manifest-path pixi.toml ~/.venvs/hetero-proxy-hf/bin/python \
#            ws/hetero-proxy/scripts/e2e/cpu_layer_time_v1.py --out <dir> [--threads 26]
################################################################################
import argparse
import json
import math
import os
import statistics
import sys
import time

import torch
import torch.nn.functional as F
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qwen_tp_v1 import Cfg, model_dir, rmsnorm, rotate_half  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=26)
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    d = model_dir()
    c = Cfg(d)
    idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]

    def get(n):
        with safe_open(os.path.join(d, idx[n]), framework="pt", device="cpu") as f:
            return f.get_tensor(n).float().contiguous()
    p = "model.layers.0."
    w = dict(ln1=get(p + "input_layernorm.weight"), ln2=get(p + "post_attention_layernorm.weight"),
             wqkv=torch.cat([get(p + f"self_attn.{k}_proj.weight") for k in "qkv"]),
             bqkv=torch.cat([get(p + f"self_attn.{k}_proj.bias") for k in "qkv"]),
             wo=get(p + "self_attn.o_proj.weight"),
             wgu=torch.cat([get(p + "mlp.gate_proj.weight"), get(p + "mlp.up_proj.weight")]),
             wd=get(p + "mlp.down_proj.weight"))
    hd, nq, nkv = c.HD, c.NQ, c.NKV
    inv = 1.0 / (c.theta ** (torch.arange(0, hd, 2).float() / hd))
    emb = torch.cat([torch.outer(torch.arange(8192).float(), inv)] * 2, -1)
    cos, sin = emb.cos(), emb.sin()

    def layer(x, kc, vc, pos0):
        B, T, _ = x.shape
        h = rmsnorm(x, w["ln1"], c.eps)
        q, k, v = F.linear(h, w["wqkv"], w["bqkv"]).split([nq * hd, nkv * hd, nkv * hd], -1)
        q = q.view(B, T, nq, hd).transpose(1, 2)
        k = k.view(B, T, nkv, hd).transpose(1, 2)
        v = v.view(B, T, nkv, hd).transpose(1, 2)
        cs, sn = cos[pos0:pos0 + T], sin[pos0:pos0 + T]
        q, k = q * cs + rotate_half(q) * sn, k * cs + rotate_half(k) * sn
        kc[:, :, pos0:pos0 + T], vc[:, :, pos0:pos0 + T] = k, v
        a = F.scaled_dot_product_attention(q, kc[:, :, :pos0 + T], vc[:, :, :pos0 + T], is_causal=T > 1,
                                           enable_gqa=True, scale=1 / math.sqrt(hd))
        x = x + F.linear(a.transpose(1, 2).reshape(B, T, nq * hd), w["wo"])
        g, u = F.linear(rmsnorm(x, w["ln2"], c.eps), w["wgu"]).split([c.FFN, c.FFN], -1)
        return x + F.linear(F.silu(g) * u, w["wd"])

    res = {}
    cases = [("prefill_1x1024", 1, 1024, 0), ("prefill_1x4096", 1, 4096, 0), ("decode_b1", 1, 1, 512),
             ("decode_b32", 32, 1, 512)]
    for name, B, T, pos0 in cases:
        x = torch.randn(B, T, c.H) * 0.1
        kc = torch.randn(B, nkv, pos0 + T, hd) * 0.1
        vc = torch.randn(B, nkv, pos0 + T, hd) * 0.1
        layer(x, kc, vc, pos0)  # warm-up
        ts = []
        for _ in range(args.reps):
            t0 = time.perf_counter()
            layer(x, kc, vc, pos0)
            ts.append((time.perf_counter() - t0) * 1e3)
        res[name] = dict(median_ms=statistics.median(ts), all_ms=ts)
        print(f"[cpu-layer] {name:<15} {statistics.median(ts):10.2f} ms per layer (median of {args.reps})", flush=True)
    with open(os.path.join(args.out, "cpu_layer_time.json"), "w") as f:
        json.dump(dict(threads=args.threads, affinity=sorted(os.sched_getaffinity(0)), results=res), f, indent=1)


if __name__ == "__main__":
    main()
