"""Slot-indexed variants of the E2E decode kernels (e2e/fused_kernels.py) for the K2 multi-user engine (2026-10-16).

In e2e/fused_kernels.py row b of a batch reads and writes KV cache row b (the batch is the first B cache rows). With
many users and a fixed number of cache slots, batch row b belongs to the conversation in slot slot[b]: these kernels
load slot[b] from device memory and use it as the cache row. Everything else (rounding points, split-K softmax) is
the original code; with slot = arange(B) they compute exactly what the originals compute (tested in
test_k2_kernels_v1.py). CUDA-graph safe: positions and slots are read from device memory.
  rope_cache_slot          qkv [R, (nq+2nkv)*hd]: RoPE at pos[r]; q -> q_out [R, nq, hd]; k, v -> cache[slot[r], :, pos[r]]
                           (also used for prefill: R = the new tokens of one sequence, pos = P + arange, slot = s)
  decode_attn_split_slot   q [B, nq, hd] over cache[slot[b], kvh, 0..pos[b]]
"""
import os
import sys

import torch
import triton
import triton.language as tl

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "e2e"))
from fused_kernels import _attn_merge  # noqa: E402


@triton.jit
def _rope_cache_slot(qkv_ptr, cos_ptr, sin_ptr, pos_ptr, slot_ptr, qo_ptr, kc_ptr, vc_ptr, s_qkv, NQ: tl.constexpr,
                     NKV: tl.constexpr, HD: tl.constexpr, s_kb, s_kh, s_kl):
    b = tl.program_id(0)
    hh = tl.program_id(1)
    half: tl.constexpr = HD // 2
    p = tl.load(pos_ptr + b)
    sb = tl.load(slot_ptr + b)
    i = tl.arange(0, half)
    c1 = tl.load(cos_ptr + p * HD + i).to(tl.float32)
    c2 = tl.load(cos_ptr + p * HD + half + i).to(tl.float32)
    s1 = tl.load(sin_ptr + p * HD + i).to(tl.float32)
    s2 = tl.load(sin_ptr + p * HD + half + i).to(tl.float32)
    base = qkv_ptr + b * s_qkv + hh * HD
    a = tl.load(base + i).to(tl.float32)
    z = tl.load(base + half + i).to(tl.float32)
    o1 = ((a * c1).to(tl.bfloat16).to(tl.float32) + (-z * s1).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
    o2 = ((z * c2).to(tl.bfloat16).to(tl.float32) + (a * s2).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
    if hh < NQ:
        qo = qo_ptr + (b * NQ + hh) * HD
        tl.store(qo + i, o1)
        tl.store(qo + half + i, o2)
    else:
        kvh = hh - NQ
        kd = kc_ptr + sb * s_kb + kvh * s_kh + p * s_kl
        tl.store(kd + i, o1)
        tl.store(kd + half + i, o2)
        vsrc = qkv_ptr + b * s_qkv + (NQ + NKV + kvh) * HD
        vd = vc_ptr + sb * s_kb + kvh * s_kh + p * s_kl
        tl.store(vd + tl.arange(0, HD), tl.load(vsrc + tl.arange(0, HD)))


def rope_cache_slot(qkv, cos, sin, pos, slot, kc, vc, nq, nkv, hd, q_out=None):
    """qkv [R, (nq+2nkv)*hd]; pos, slot [R] int64; kc/vc [n_slots, nkv, L, hd] (one layer, all slots)."""
    R = qkv.shape[0]
    q_out = torch.empty(R, nq, hd, dtype=qkv.dtype, device=qkv.device) if q_out is None else q_out
    _rope_cache_slot[(R, nq + nkv)](qkv, cos, sin, pos, slot, q_out, kc, vc, qkv.stride(0), NQ=nq, NKV=nkv, HD=hd,
                                    s_kb=kc.stride(0), s_kh=kc.stride(1), s_kl=kc.stride(2), num_warps=1)
    return q_out


@triton.jit
def _attn_split_slot(q_ptr, kc_ptr, vc_ptr, pos_ptr, slot_ptr, m_ptr, l_ptr, a_ptr, scale, NQ: tl.constexpr,
                     NKV: tl.constexpr, HD: tl.constexpr, GQ: tl.constexpr, s_kb, s_kh, s_kl, BLOCK_N: tl.constexpr,
                     NSPLIT: tl.constexpr):
    bk = tl.program_id(0)
    sp = tl.program_id(1)
    b = bk // NKV
    kvh = bk % NKV
    G: tl.constexpr = NQ // NKV
    sb = tl.load(slot_ptr + b)
    n_keys = tl.load(pos_ptr + b) + 1
    per = tl.cdiv(tl.cdiv(n_keys, NSPLIT), BLOCK_N) * BLOCK_N
    start = sp * per
    end = tl.minimum(start + per, n_keys)
    g = tl.arange(0, GQ)
    gm = g < G
    d = tl.arange(0, HD)
    q = tl.load(q_ptr + ((b * NQ + kvh * G + g) * HD)[:, None] + d[None, :], mask=gm[:, None], other=0.0)
    m_i = tl.zeros([GQ], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([GQ], dtype=tl.float32)
    acc = tl.zeros([GQ, HD], dtype=tl.float32)
    kbase = kc_ptr + sb * s_kb + kvh * s_kh
    vbase = vc_ptr + sb * s_kb + kvh * s_kh
    for n0 in range(start, end, BLOCK_N):
        n = n0 + tl.arange(0, BLOCK_N)
        valid = n < end
        k = tl.load(kbase + n[:, None] * s_kl + d[None, :], mask=valid[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k)) * scale
        s = tl.where(valid[None, :], s, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        e = tl.exp(s - m_new[:, None])
        v = tl.load(vbase + n[:, None] * s_kl + d[None, :], mask=valid[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(e.to(tl.bfloat16), v)
        l_i = l_i * alpha + tl.sum(e, axis=1)
        m_i = m_new
    o = (bk * NSPLIT + sp) * GQ + g
    tl.store(m_ptr + o, m_i)
    tl.store(l_ptr + o, l_i)
    tl.store(a_ptr + o[:, None] * HD + d[None, :], acc)


def decode_attn_split_slot(q, kc, vc, pos, slot, scale, out=None, nsplit=8, block_n=64, scratch=None):
    """q [B, nq, hd]; kc/vc [n_slots, nkv, L, hd]; row b attends cache[slot[b], kvh, 0..pos[b]]. Returns [B, nq*hd]."""
    B, nq, hd = q.shape
    nkv = kc.shape[1]
    GQ = max(16, triton.next_power_of_2(nq // nkv))
    if scratch is None:
        scratch = (torch.empty(B * nkv * nsplit * GQ, dtype=torch.float32, device=q.device),
                   torch.empty(B * nkv * nsplit * GQ, dtype=torch.float32, device=q.device),
                   torch.empty(B * nkv * nsplit * GQ * hd, dtype=torch.float32, device=q.device))
    m, l, a = scratch
    out = torch.empty(B, nq * hd, dtype=q.dtype, device=q.device) if out is None else out
    _attn_split_slot[(B * nkv, nsplit)](q, kc, vc, pos, slot, m, l, a, scale, NQ=nq, NKV=nkv, HD=hd, GQ=GQ,
                                        s_kb=kc.stride(0), s_kh=kc.stride(1), s_kl=kc.stride(2), BLOCK_N=block_n,
                                        NSPLIT=nsplit, num_warps=4)
    _attn_merge[(B * nkv,)](m, l, a, out, NQ=nq, NKV=nkv, HD=hd, GQ=GQ, NSPLIT=nsplit, num_warps=4)
    return out
