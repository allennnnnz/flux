################################################################################
# hetero-proxy E2E (2026-10-14): fused Triton kernels (fused_kernels.py) against the v3 torch path they replace.
# Shapes of Qwen2.5-32B at TP8 (H 5120, 5 q heads + 1 kv head per rank, hd 128, ffn 3456 per rank). Reports the
# fraction of bf16 outputs that differ and the max |diff|, then per-op time (CUDA events, median of 50).
# Usage: pixi run --manifest-path pixi.toml ~/.venvs/hetero-proxy-hf/bin/python test_fused_kernels_v1.py
################################################################################
import math
import os
import statistics
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fused_kernels import add_rmsnorm, decode_attn, rope_cache, silu_mul  # noqa: E402
from qwen_tp_v1 import rmsnorm, rotate_half  # noqa: E402


def cmp(name, a, b):
    d = (a.float() - b.float()).abs()
    nd = int((a != b).sum())
    print(f"[{name}] differing {nd}/{a.numel()} ({100 * nd / a.numel():.4f}%), max |diff| {d.max().item():.4g}", flush=True)


def t(fn, n=50):
    fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) * 1e3)
    return statistics.median(ts)


def main():
    torch.manual_seed(0)
    dev = "cuda:0"
    H, nq, nkv, hd, ffn, L = 5120, 5, 1, 128, 3456, 640
    for B in (1, 32, 2048):
        x = (torch.randn(B, H, device=dev) * 2).to(torch.bfloat16)
        r = (torch.randn(B, H, device=dev)).to(torch.bfloat16)
        w = (torch.rand(H, device=dev) * 2).to(torch.bfloat16)
        ref_x = x + r
        ref_h = rmsnorm(ref_x, w, 1e-6)
        fx, fh = add_rmsnorm(x, w, 1e-6, r=r)
        cmp(f"add_rmsnorm B={B} residual", fx, ref_x)
        cmp(f"add_rmsnorm B={B} norm", fh, ref_h)
        print(f"   time torch {t(lambda: rmsnorm(x + r, w, 1e-6)):8.1f} us   fused {t(lambda: add_rmsnorm(x, w, 1e-6, r=r)):8.1f} us")
        gu = torch.randn(B, 2 * ffn, device=dev).to(torch.bfloat16)
        g, u = gu.split([ffn, ffn], -1)
        cmp(f"silu_mul B={B}", silu_mul(gu), F.silu(g) * u)
        print(f"   time torch {t(lambda: F.silu(g) * u):8.1f} us   fused {t(lambda: silu_mul(gu)):8.1f} us")
    # rope + cache + attention (decode), B = 1 / 32 / 64
    inv = 1.0 / (1e6 ** (torch.arange(0, hd, 2, device=dev).float() / hd))
    emb = torch.cat([torch.outer(torch.arange(L, device=dev).float(), inv)] * 2, -1)
    cos, sin = emb.cos().to(torch.bfloat16), emb.sin().to(torch.bfloat16)
    for B in (1, 32, 64):
        qkv = torch.randn(B, (nq + 2 * nkv) * hd, device=dev).to(torch.bfloat16)
        kc = torch.randn(B, nkv, L, hd, device=dev).to(torch.bfloat16)
        vc = torch.randn(B, nkv, L, hd, device=dev).to(torch.bfloat16)
        pos = torch.randint(100, 600, (B,), device=dev)
        # torch path (v3)
        kc2, vc2 = kc.clone(), vc.clone()
        q, k, v = qkv.split([nq * hd, nkv * hd, nkv * hd], -1)
        q = q.view(B, 1, nq, hd).transpose(1, 2)
        k = k.view(B, 1, nkv, hd).transpose(1, 2)
        v = v.view(B, 1, nkv, hd).transpose(1, 2)
        c, s = cos[pos][:, None, None, :], sin[pos][:, None, None, :]
        q = q * c + rotate_half(q) * s
        k = k * c + rotate_half(k) * s
        bi = torch.arange(B, device=dev)
        kc2[bi, :, pos] = k[:, :, 0]
        vc2[bi, :, pos] = v[:, :, 0]
        mask = (torch.arange(L, device=dev)[None, :] <= pos[:, None])[:, None, None, :]
        ref_a = F.scaled_dot_product_attention(q, kc2, vc2, attn_mask=mask, enable_gqa=True,
                                               scale=1 / math.sqrt(hd)).transpose(1, 2).reshape(B, nq * hd)
        kc3, vc3 = kc.clone(), vc.clone()
        qf = rope_cache(qkv, cos, sin, pos, kc3, vc3, nq, nkv, hd)
        cmp(f"rope q B={B}", qf, q[:, :, 0])
        cmp(f"rope k->cache B={B}", kc3, kc2)
        cmp(f"v->cache B={B}", vc3, vc2)
        fa = decode_attn(qf, kc3, vc3, pos, 1 / math.sqrt(hd))
        cmp(f"decode_attn B={B}", fa, ref_a)
        print(f"   time torch rope+cache+sdpa {t(lambda: F.scaled_dot_product_attention(q, kc2, vc2, attn_mask=mask, enable_gqa=True)):8.1f} us (sdpa only)"
              f"   fused rope {t(lambda: rope_cache(qkv, cos, sin, pos, kc3, vc3, nq, nkv, hd)):6.1f} us  attn {t(lambda: decode_attn(qf, kc3, vc3, pos, 1 / math.sqrt(hd))):6.1f} us")


if __name__ == "__main__":
    main()
