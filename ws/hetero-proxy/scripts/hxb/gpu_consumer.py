"""Chunk-signaled GEMM on the GPU: C = A @ B where A arrives in row chunks (Flux's per-tile wait, in Triton).

Each program computes one BM x BN tile; before loading A it spins (acquire, system scope, as Flux's
SystemBarrier::wait_eq) until flags[chunk] >= target, chunk = first row of the tile // chunk_rows. Programs are
ordered chunk-major (GROUP_M = tiles per chunk), so the resident programs are those of the earliest chunks and
waiting never blocks a chunk that has already arrived. The flags are written by Inbound with stream memory
operations, never by a kernel. WAIT=False gives the same kernel without waiting: the GEMM-only reference
(the analogue of Flux's AGKernel.gemm_only).
"""
import torch
import triton
import triton.language as tl


# do_not_specialize: Triton specialises integer arguments equal to 1 or divisible by 16; a new `target` value would
# compile and load a new module mid-run, and module loading synchronises the context -> deadlock against streams
# parked on a WaitValue (found 2026-10-09: pass 1, target=2, hung in _init_handles).
@triton.jit(do_not_specialize=["target"])
def _chunk_gemm(a_ptr, b_ptr, c_ptr, flag_ptr, M, N, K, s_am, s_ak, s_bk, s_bn, s_cm, s_cn, chunk_rows, target,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr, WAIT: tl.constexpr):
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BM)
    num_n = tl.cdiv(N, BN)
    per_group = GROUP_M * num_n
    first_m = (pid // per_group) * GROUP_M
    gsize = tl.minimum(num_m - first_m, GROUP_M)
    pid_m = first_m + (pid % per_group) % gsize
    pid_n = (pid % per_group) // gsize
    if WAIT:
        fp = flag_ptr + (pid_m * BM) // chunk_rows
        f = tl.atomic_add(fp, 0, sem="acquire", scope="sys")
        while f < target:
            f = tl.atomic_add(fp, 0, sem="acquire", scope="sys")
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    a_ptrs = a_ptr + offs_m[:, None] * s_am + offs_k[None, :] * s_ak
    b_ptrs = b_ptr + offs_k[:, None] * s_bk + offs_n[None, :] * s_bn
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K - k * BK), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k * BK) & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BK * s_ak
        b_ptrs += BK * s_bk
    c_ptrs = c_ptr + offs_m[:, None] * s_cm + offs_n[None, :] * s_cn
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# 2026-10-09 probe, A100 bf16 4096x5120x5120: this config 0.912 ms (236 TFLOPS), torch.mm (cuBLAS) 0.919 ms;
# with WAIT and flags already set +2%.
CONFIG = dict(BM=128, BN=256, BK=32, num_warps=8, num_stages=3)


def chunk_gemm(a, b, c, flags=None, chunk_rows=None, target=1, cfg=None):
    """c = a @ b; if flags is given, the tiles of row chunk i wait for flags[i] >= target."""
    cfg = dict(CONFIG, **(cfg or {}))
    M, K = a.shape
    N = b.shape[1]
    wait = flags is not None
    cr = chunk_rows or M
    if wait:
        assert cr % cfg["BM"] == 0, "chunk_rows must be a multiple of BM (a tile never spans two chunks)"
    group = max(1, min(cr // cfg["BM"], 8)) if wait else 8
    grid = (triton.cdiv(M, cfg["BM"]) * triton.cdiv(N, cfg["BN"]),)
    _chunk_gemm[grid](a, b, c, flags if wait else a, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                      c.stride(0), c.stride(1), cr, target, BM=cfg["BM"], BN=cfg["BN"], BK=cfg["BK"], GROUP_M=group,
                      WAIT=wait, num_warps=cfg["num_warps"], num_stages=cfg["num_stages"])
    return c
