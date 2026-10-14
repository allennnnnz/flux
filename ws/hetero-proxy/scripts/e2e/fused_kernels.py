"""Fused Triton kernels for the E2E engine v4 (2026-10-14).

nsys of qwen_tp_v3 decode (results/e2e/nsys_decode/analysis.txt): ~3850 kernels per step, half of GPU0's time in
small elementwise kernels (fp32 RMSNorm chains, RoPE chunk / cat / mul, masked SDPA math path, silu * up). These
kernels replace them; every one keeps HF Qwen2's bf16 rounding points (results may still differ by an ulp where the
fp32 reduction order differs, e.g. the RMSNorm sum).
  add_rmsnorm   x' = bf16(x + r); h = bf16(bf16(x' * rsqrt(mean(x'^2) + eps)) * w)      (r optional)
  rope_cache    decode: RoPE on q and k of qkv [B, (nq+2nkv)*hd] at per-row positions; q -> q_out [B, nq, hd];
                k, v -> KV cache [B, nkv, L, hd] at pos[b]       (HF: bf16(bf16(q*cos) + bf16(rot(q)*sin)))
  decode_attn   q [B, nq, hd] over cache[:, kvh, 0..pos[b]] (GQA), fp32 online softmax, bf16 out [B, nq*hd]
  silu_mul      out = bf16(bf16(silu(g)) * u) for gu = [g | u] along the last dim
All are CUDA-graph safe: positions are read from device memory inside the kernels.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _add_rmsnorm(x_ptr, r_ptr, w_ptr, xo_ptr, h_ptr, H, eps, HAS_R: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    m = offs < H
    x = tl.load(x_ptr + row * H + offs, mask=m, other=0.0).to(tl.float32)
    if HAS_R:
        r = tl.load(r_ptr + row * H + offs, mask=m, other=0.0).to(tl.float32)
        x = (x + r).to(tl.bfloat16).to(tl.float32)
        tl.store(xo_ptr + row * H + offs, x.to(tl.bfloat16), mask=m)
    var = tl.sum(x * x, axis=0) / H
    inv = tl.math.rsqrt(var + eps)
    xn = (x * inv).to(tl.bfloat16).to(tl.float32)
    w = tl.load(w_ptr + offs, mask=m, other=0.0).to(tl.float32)
    tl.store(h_ptr + row * H + offs, (xn * w).to(tl.bfloat16), mask=m)


def add_rmsnorm(x, w, eps, r=None, x_out=None, h_out=None):
    """x, r: [rows, H] bf16 contiguous. Returns (x' (= x + r, or x itself when r is None), h)."""
    rows, H = x.shape[0] if x.dim() == 2 else x.numel() // x.shape[-1], x.shape[-1]
    h_out = torch.empty_like(x) if h_out is None else h_out
    if r is not None:
        x_out = torch.empty_like(x) if x_out is None else x_out
    _add_rmsnorm[(rows,)](x, r if r is not None else x, w, x_out if r is not None else x, h_out, H, eps,
                          HAS_R=r is not None, BLOCK=triton.next_power_of_2(H), num_warps=8)
    return (x_out if r is not None else x), h_out


@triton.jit
def _rope_cache(qkv_ptr, cos_ptr, sin_ptr, pos_ptr, qo_ptr, kc_ptr, vc_ptr, s_qkv, NQ: tl.constexpr,
                NKV: tl.constexpr, HD: tl.constexpr, s_kb, s_kh, s_kl):
    b = tl.program_id(0)
    hh = tl.program_id(1)  # 0..NQ-1 q heads, NQ..NQ+NKV-1 kv heads
    half: tl.constexpr = HD // 2
    p = tl.load(pos_ptr + b)
    i = tl.arange(0, half)
    c1 = tl.load(cos_ptr + p * HD + i).to(tl.float32)
    c2 = tl.load(cos_ptr + p * HD + half + i).to(tl.float32)
    s1 = tl.load(sin_ptr + p * HD + i).to(tl.float32)
    s2 = tl.load(sin_ptr + p * HD + half + i).to(tl.float32)
    base = qkv_ptr + b * s_qkv + hh * HD
    a = tl.load(base + i).to(tl.float32)          # first half
    z = tl.load(base + half + i).to(tl.float32)   # second half
    # rotate_half(x) = cat(-x2, x1);  out = bf16(bf16(x * cos) + bf16(rot * sin))
    o1 = ((a * c1).to(tl.bfloat16).to(tl.float32) + (-z * s1).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
    o2 = ((z * c2).to(tl.bfloat16).to(tl.float32) + (a * s2).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
    if hh < NQ:
        qo = qo_ptr + (b * NQ + hh) * HD
        tl.store(qo + i, o1)
        tl.store(qo + half + i, o2)
    else:
        kvh = hh - NQ
        kd = kc_ptr + b * s_kb + kvh * s_kh + p * s_kl
        tl.store(kd + i, o1)
        tl.store(kd + half + i, o2)
        vsrc = qkv_ptr + b * s_qkv + (NQ + NKV + kvh) * HD
        vd = vc_ptr + b * s_kb + kvh * s_kh + p * s_kl
        tl.store(vd + tl.arange(0, HD), tl.load(vsrc + tl.arange(0, HD)))


def rope_cache(qkv, cos, sin, pos, kc, vc, nq, nkv, hd, q_out=None):
    """qkv [B, (nq+2nkv)*hd] bf16; cos/sin [L, hd] bf16; pos [B] int64; kc/vc [B, nkv, L, hd] views (one layer)."""
    B = qkv.shape[0]
    q_out = torch.empty(B, nq, hd, dtype=qkv.dtype, device=qkv.device) if q_out is None else q_out
    _rope_cache[(B, nq + nkv)](qkv, cos, sin, pos, q_out, kc, vc, qkv.stride(0), NQ=nq, NKV=nkv, HD=hd,
                               s_kb=kc.stride(0), s_kh=kc.stride(1), s_kl=kc.stride(2), num_warps=1)
    return q_out


@triton.jit
def _decode_attn(q_ptr, kc_ptr, vc_ptr, pos_ptr, o_ptr, scale, NQ: tl.constexpr, NKV: tl.constexpr,
                 HD: tl.constexpr, s_kb, s_kh, s_kl, BLOCK_N: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    kvh = h // (NQ // NKV)
    p = tl.load(pos_ptr + b)
    d = tl.arange(0, HD)
    q = tl.load(q_ptr + (b * NQ + h) * HD + d).to(tl.float32)
    m_i = tl.zeros([], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([], dtype=tl.float32)
    acc = tl.zeros([HD], dtype=tl.float32)
    kbase = kc_ptr + b * s_kb + kvh * s_kh
    vbase = vc_ptr + b * s_kb + kvh * s_kh
    for n0 in range(0, p + 1, BLOCK_N):
        n = n0 + tl.arange(0, BLOCK_N)
        valid = n <= p
        k = tl.load(kbase + n[:, None] * s_kl + d[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
        s = tl.sum(k * q[None, :], axis=1) * scale
        s = tl.where(valid, s, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=0))
        alpha = tl.exp(m_i - m_new)
        e = tl.exp(s - m_new)
        v = tl.load(vbase + n[:, None] * s_kl + d[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
        acc = acc * alpha + tl.sum(e[:, None] * v, axis=0)
        l_i = l_i * alpha + tl.sum(e, axis=0)
        m_i = m_new
    tl.store(o_ptr + (b * NQ + h) * HD + d, (acc / l_i).to(tl.bfloat16))


def decode_attn(q, kc, vc, pos, scale, out=None, block_n=64):
    """q [B, nq, hd]; kc/vc [B, nkv, L, hd] (one layer); pos [B]: attend keys 0..pos[b]. Returns [B, nq*hd]."""
    B, nq, hd = q.shape
    nkv = kc.shape[1]
    out = torch.empty(B, nq * hd, dtype=q.dtype, device=q.device) if out is None else out
    _decode_attn[(B, nq)](q, kc, vc, pos, out, scale, NQ=nq, NKV=nkv, HD=hd, s_kb=kc.stride(0), s_kh=kc.stride(1),
                          s_kl=kc.stride(2), BLOCK_N=block_n, num_warps=4)
    return out


@triton.jit
def _silu_mul(gu_ptr, o_ptr, F, n_elem, BLOCK: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = idx < n_elem
    row = idx // F
    col = idx % F
    g = tl.load(gu_ptr + row * 2 * F + col, mask=m, other=0.0).to(tl.float32)
    u = tl.load(gu_ptr + row * 2 * F + F + col, mask=m, other=0.0).to(tl.float32)
    s = (g / (1.0 + tl.exp(-g))).to(tl.bfloat16).to(tl.float32)
    tl.store(o_ptr + idx, (s * u).to(tl.bfloat16), mask=m)


def silu_mul(gu, out=None):
    """gu [..., 2F] bf16 -> [..., F] = bf16(bf16(silu(g)) * u)."""
    F = gu.shape[-1] // 2
    rows = gu.numel() // gu.shape[-1]
    out = torch.empty(*gu.shape[:-1], F, dtype=gu.dtype, device=gu.device) if out is None else out
    n = rows * F
    _silu_mul[(triton.cdiv(n, 1024),)](gu, out, F, n, BLOCK=1024, num_warps=4)
    return out


# ---- split-K decode attention (flash-decoding), 2026-10-14 ---------------------------------------------------------
# nsys of v4 decode: _decode_attn took 23% of the step (~62 us per layer): grid (B, nq) gave 5 programs at B=1 and no
# tensor cores. Here the q heads sharing a kv head are one tile (padded to 16 rows, tl.dot), the keys are split
# across NSPLIT programs, and a second kernel merges the partial softmaxes. P is cast to bf16 for the PV product,
# as flash attention does.
@triton.jit
def _attn_split(q_ptr, kc_ptr, vc_ptr, pos_ptr, m_ptr, l_ptr, a_ptr, scale, NQ: tl.constexpr, NKV: tl.constexpr,
                HD: tl.constexpr, GQ: tl.constexpr, s_kb, s_kh, s_kl, BLOCK_N: tl.constexpr, NSPLIT: tl.constexpr):
    bk = tl.program_id(0)
    sp = tl.program_id(1)
    b = bk // NKV
    kvh = bk % NKV
    G: tl.constexpr = NQ // NKV
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
    kbase = kc_ptr + b * s_kb + kvh * s_kh
    vbase = vc_ptr + b * s_kb + kvh * s_kh
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


@triton.jit
def _attn_merge(m_ptr, l_ptr, a_ptr, o_ptr, NQ: tl.constexpr, NKV: tl.constexpr, HD: tl.constexpr,
                GQ: tl.constexpr, NSPLIT: tl.constexpr):
    bk = tl.program_id(0)
    b = bk // NKV
    kvh = bk % NKV
    G: tl.constexpr = NQ // NKV
    g = tl.arange(0, GQ)
    d = tl.arange(0, HD)
    M = tl.zeros([GQ], dtype=tl.float32) - float("inf")
    for sp in range(NSPLIT):
        M = tl.maximum(M, tl.load(m_ptr + (bk * NSPLIT + sp) * GQ + g))
    Lsum = tl.zeros([GQ], dtype=tl.float32)
    acc = tl.zeros([GQ, HD], dtype=tl.float32)
    for sp in range(NSPLIT):
        o = (bk * NSPLIT + sp) * GQ + g
        w = tl.exp(tl.load(m_ptr + o) - M)  # empty split: m = -inf -> w = 0
        Lsum += tl.load(l_ptr + o) * w
        acc += tl.load(a_ptr + o[:, None] * HD + d[None, :]) * w[:, None]
    out = acc / Lsum[:, None]
    tl.store(o_ptr + ((b * NQ + kvh * G + g) * HD)[:, None] + d[None, :], out.to(tl.bfloat16), mask=(g < G)[:, None])


def decode_attn_split(q, kc, vc, pos, scale, out=None, nsplit=8, block_n=64, scratch=None):
    """Same contract as decode_attn: q [B, nq, hd]; kc/vc [B, nkv, L, hd]; keys 0..pos[b]. Returns [B, nq*hd]."""
    B, nq, hd = q.shape
    nkv = kc.shape[1]
    GQ = max(16, triton.next_power_of_2(nq // nkv))
    if scratch is None:
        scratch = (torch.empty(B * nkv * nsplit * GQ, dtype=torch.float32, device=q.device),
                   torch.empty(B * nkv * nsplit * GQ, dtype=torch.float32, device=q.device),
                   torch.empty(B * nkv * nsplit * GQ * hd, dtype=torch.float32, device=q.device))
    m, l, a = scratch
    out = torch.empty(B, nq * hd, dtype=q.dtype, device=q.device) if out is None else out
    _attn_split[(B * nkv, nsplit)](q, kc, vc, pos, m, l, a, scale, NQ=nq, NKV=nkv, HD=hd, GQ=GQ, s_kb=kc.stride(0),
                                   s_kh=kc.stride(1), s_kl=kc.stride(2), BLOCK_N=block_n, NSPLIT=nsplit, num_warps=4)
    _attn_merge[(B * nkv,)](m, l, a, out, NQ=nq, NKV=nkv, HD=hd, GQ=GQ, NSPLIT=nsplit, num_warps=4)
    return out
