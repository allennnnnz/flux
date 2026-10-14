################################################################################
# hetero-proxy E2E (2026-10-14): split-K decode attention vs the first fused decode_attn and an fp32 SDPA reference;
# per-call time measured inside a CUDA graph (64 calls per graph). Run from the repo root.
################################################################################
import math, sys, torch, torch.nn.functional as F
sys.path.insert(0, "ws/hetero-proxy/scripts/e2e")
from fused_kernels import decode_attn, decode_attn_split
dev = "cuda:0"; nq, nkv, hd, L = 5, 1, 128, 640
def graph_time(fn, n=64):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s): fn(); fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n): fn()
    g.replay(); torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True); a.record()
    for _ in range(5): g.replay()
    b.record(); b.synchronize(); return a.elapsed_time(b) / 5 / n * 1e3
for B in (1, 32, 64):
    q = torch.randn(B, nq, hd, device=dev).to(torch.bfloat16)
    kc = torch.randn(B, nkv, L, hd, device=dev).to(torch.bfloat16); vc = torch.randn_like(kc)
    pos = torch.randint(500, 600, (B,), device=dev)
    mask = (torch.arange(L, device=dev)[None, :] <= pos[:, None])[:, None, None, :]
    ref = F.scaled_dot_product_attention(q.float()[:, :, None], kc.float(), vc.float(), attn_mask=mask, enable_gqa=True, scale=1/math.sqrt(hd))[:, :, 0].reshape(B, -1)
    o1 = decode_attn(q, kc, vc, pos, 1/math.sqrt(hd)); o2 = decode_attn_split(q, kc, vc, pos, 1/math.sqrt(hd))
    sc = (torch.empty(B*nkv*8*16, device=dev), torch.empty(B*nkv*8*16, device=dev), torch.empty(B*nkv*8*16*hd, device=dev))
    print(f"B={B}: max|old-fp32ref| {(o1.float()-ref).abs().max().item():.3g}  max|split-fp32ref| {(o2.float()-ref).abs().max().item():.3g}  "
          f"graph time old {graph_time(lambda: decode_attn(q, kc, vc, pos, 0.088)):.1f} us  split {graph_time(lambda: decode_attn_split(q, kc, vc, pos, 0.088, scratch=sc)):.1f} us", flush=True)
