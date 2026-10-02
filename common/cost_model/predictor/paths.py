"""Predicted time of every op-level path (arm) for one layer and one M.

AG side (column-parallel layer, each rank holds weight (n, K), n = N / W):
  A fused AGKernel   B NCCL all_gather + cuBLAS   C Flux AllGather + cuBLAS
  D Flux AllGather + Flux gemm_only (serial)
RS side (row-parallel layer, each rank holds weight (N, k), k = K / W):
  A Flux GemmRS      B cuBLAS + NCCL reduce_scatter
Arm names match ws/fusion-dispatch results (t_fused, t_B, t_C, t_D).
"""
from .overlap import fused_ag_time, fused_rs_time

AG_ARMS = ("A", "B", "C", "D")
RS_ARMS = ("A", "B")
FLUX_GEMM_ARMS = {"ag": {"A", "D"}, "rs": {"A"}}  # arms that run a Flux CUTLASS GEMM


def predict_ag(prof, cfgs, M, n, K, clock=None):
    W = prof.W
    nb = M * K * 2
    t_nccl = prof.comm.nccl("ag", nb, clock)
    t_fag = prof.comm.flux_ag(nb, W)
    t_cub = prof.gemm["cublas"].time(M, n, K)
    cfg = cfgs.get("ag", M, n, K)
    t_fg = prof.gemm["flux_ag"].time(M, n, K, cfg["tile"])
    t_a = fused_ag_time(M, n, K, W, cfg, t_fag, t_fg, prof.fused_ag)
    arms = {"A": t_a, "B": t_nccl + t_cub, "C": t_fag + t_cub, "D": t_fag + t_fg}
    comps = {"nccl": t_nccl, "fluxag": t_fag, "cublas": t_cub, "fluxgemm": t_fg}
    return arms, comps, cfg


def predict_rs(prof, cfgs, M, N, k, clock=None):
    W = prof.W
    t_rs = prof.comm.nccl("rs", M * N * 2, clock)
    t_cub = prof.gemm["cublas"].time(M, N, k)
    cfg = cfgs.get("rs", M, N, k)
    t_g = prof.gemm["flux_rs"].time(M, N, k, cfg["tile"])
    t_a = fused_rs_time(M, N, W, t_g, prof.fused_rs["alpha_rs"], prof.fused_rs["beta_scat"])
    arms = {"A": t_a, "B": t_cub + t_rs}
    comps = {"nccl_rs": t_rs, "cublas": t_cub, "fluxrs_gemm": t_g}
    return arms, comps, cfg
