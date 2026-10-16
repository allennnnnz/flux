################################################################################
# hetero-proxy K2 (2026-10-16): the slot-indexed kernels (k2_kernels.py) against the originals (e2e/fused_kernels.py).
#   1. slot = arange(B): bit-identical outputs and cache writes (same code apart from the row -> slot lookup)
#   2. permuted slots: row b's outputs equal the original's on a cache whose row b is slot[b]'s row
#   3. prefill use of rope_cache_slot: R new tokens of one sequence (pos = P + arange, slot = s) write the same
#      cache entries as R single-row calls
# Usage (one GPU): pixi run --manifest-path pixi.toml python test_k2_kernels_v1.py
################################################################################
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "e2e"))
from fused_kernels import decode_attn_split, rope_cache  # noqa: E402
from k2_kernels import decode_attn_split_slot, rope_cache_slot  # noqa: E402

torch.manual_seed(0)
dev = torch.device("cuda:0")
nq, nkv, hd, L, S, B = 5, 1, 128, 1024, 9, 6
cos = torch.randn(L, hd, device=dev).to(torch.bfloat16)
sin = torch.randn(L, hd, device=dev).to(torch.bfloat16)
qkv = torch.randn(B, (nq + 2 * nkv) * hd, device=dev).to(torch.bfloat16)
pos = torch.tensor([3, 700, 64, 1000, 129, 0], device=dev)
scale = 1 / math.sqrt(hd)
ok = True

# 1. identity slots
kc0 = torch.randn(S, nkv, L, hd, device=dev).to(torch.bfloat16)
vc0 = torch.randn_like(kc0)
ka, va, kb, vb = kc0.clone(), vc0.clone(), kc0.clone(), vc0.clone()
q1 = rope_cache(qkv, cos, sin, pos, ka[:B], va[:B], nq, nkv, hd)
q2 = rope_cache_slot(qkv, cos, sin, pos, torch.arange(B, device=dev), kb, vb, nq, nkv, hd)
o1 = decode_attn_split(q1, ka[:B], va[:B], pos, scale)
o2 = decode_attn_split_slot(q2, kb, vb, pos, torch.arange(B, device=dev), scale)
r1 = torch.equal(q1, q2) and torch.equal(ka, kb) and torch.equal(va, vb) and torch.equal(o1, o2)
print(f"[1] identity slots bit-identical (q, cache, attention): {r1}")
ok &= r1

# 2. permuted slots
slot = torch.tensor([7, 2, 5, 0, 8, 3], device=dev)
kc, vc = kc0.clone(), vc0.clone()
qs = rope_cache_slot(qkv, cos, sin, pos, slot, kc, vc, nq, nkv, hd)
os_ = decode_attn_split_slot(qs, kc, vc, pos, slot, scale)
kr, vr = kc0[slot].clone(), vc0[slot].clone()  # the original's view: row b = slot[b]'s cache row
qr = rope_cache(qkv, cos, sin, pos, kr, vr, nq, nkv, hd)
orf = decode_attn_split(qr, kr, vr, pos, scale)
untouched = [s for s in range(S) if s not in slot.tolist()]
r2 = torch.equal(os_, orf) and torch.equal(kc[slot], kr) and torch.equal(vc[slot], vr) and \
    torch.equal(kc[untouched], kc0[untouched])
print(f"[2] permuted slots match the original on the gathered cache, other slots untouched: {r2}")
ok &= r2

# 3. prefill: R tokens of one sequence in slot 4 at positions P..P+R-1
R, P, s = 37, 500, 4
qkvp = torch.randn(R, (nq + 2 * nkv) * hd, device=dev).to(torch.bfloat16)
kc, vc = kc0.clone(), vc0.clone()
qp = rope_cache_slot(qkvp, cos, sin, P + torch.arange(R, device=dev), torch.full((R,), s, device=dev), kc, vc, nq,
                     nkv, hd)
kc2, vc2 = kc0.clone(), vc0.clone()
qq = []
for r in range(R):
    qq.append(rope_cache_slot(qkvp[r:r + 1], cos, sin, torch.tensor([P + r], device=dev),
                              torch.tensor([s], device=dev), kc2, vc2, nq, nkv, hd))
r3 = torch.equal(qp, torch.cat(qq)) and torch.equal(kc, kc2) and torch.equal(vc, vc2)
print(f"[3] prefill rows = single-row calls (q and cache): {r3}")
ok &= r3
print("ALL PASS" if ok else "FAIL")
