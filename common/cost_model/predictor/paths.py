"""Predicted time of every op-level path (arm) for one layer and one M.

AG side (column-parallel layer, each rank holds weight (n, K), n = N / W):
  A fused AGKernel   B NCCL all_gather + cuBLAS   C Flux AllGather + cuBLAS
  D Flux AllGather + Flux gemm_only (serial)
RS side (row-parallel layer, each rank holds weight (N, k), k = K / W):
  A Flux GemmRS      B cuBLAS + NCCL reduce_scatter
Arm names match ws/fusion-dispatch results (t_fused, t_B, t_C, t_D).

meas (optional, G4 / D-009): measured standalone single-GPU GEMM times in ms that replace the GEMM
model, because cuBLAS and Flux configs have cliffs no GEMM model predicts (G3):
  AG: {"cublas": torch.mm (M x n x K), "fluxgemm": AGKernel.gemm_only}
  RS: {"cublas": torch.mm (M x N x k), "fluxgemm_only": flux.GemmOnly}
Communication and overlap always come from the model.
GemmRS (RS arm A), first available of:
  fused_rs["meas"]    GEMM = measured flux.GemmOnly, alpha / beta fitted against it (G4)
  fused_rs["shared"]  GEMM = the Flux gemm_only family model, alpha / beta fitted (G4, model only)
  fused_rs            joint 6-parameter fit (G1-G3; overfits, see G2 report section 3)
"""
from .overlap import fused_ag_time, fused_rs_time

AG_ARMS = ("A", "B", "C", "D")
RS_ARMS = ("A", "B")
FLUX_GEMM_ARMS = {"ag": {"A", "D"}, "rs": {"A"}}  # arms that run a Flux CUTLASS GEMM


def predict_ag(prof, cfgs, M, n, K, clock=None, meas=None):
    meas = meas or {}
    W = prof.W
    nb = M * K * 2
    t_nccl = prof.comm.nccl("ag", nb, clock)
    t_fag = prof.comm.flux_ag(nb, W)
    cfg = cfgs.get("ag", M, n, K)
    t_cub = meas.get("cublas") or prof.gemm["cublas"].time(M, n, K)
    t_fg = meas.get("fluxgemm") or prof.gemm["flux_ag"].time(M, n, K, cfg["tile"])
    t_a = fused_ag_time(M, n, K, W, cfg, t_fag, t_fg, prof.fused_ag)
    arms = {"A": t_a, "B": t_nccl + t_cub, "C": t_fag + t_cub, "D": t_fag + t_fg}
    comps = {"nccl": t_nccl, "fluxag": t_fag, "cublas": t_cub, "fluxgemm": t_fg}
    return arms, comps, cfg


def predict_rs(prof, cfgs, M, N, k, clock=None, meas=None, legacy=False):
    """legacy=True reproduces G1-G3 (joint-fit GemmRS model) even if newer parameters exist."""
    meas = meas or {}
    W = prof.W
    t_rs = prof.comm.nccl("rs", M * N * 2, clock)
    t_cub = meas.get("cublas") or prof.gemm["cublas"].time(M, N, k)
    cfg = cfgs.get("rs", M, N, k)
    frs = prof.fused_rs
    if not legacy and meas.get("fluxgemm_only") and "meas" in frs:
        t_g, prm = meas["fluxgemm_only"], frs["meas"]
    elif not legacy and "shared" in frs:
        t_g, prm = prof.gemm["flux_ag"].time(M, N, k, cfg["tile"]), frs["shared"]
    else:
        t_g, prm = prof.gemm["flux_rs"].time(M, N, k, cfg["tile"]), frs
    t_a = fused_rs_time(M, N, W, t_g, prm["alpha_rs"], prm["beta_scat"])
    arms = {"A": t_a, "B": t_cub + t_rs}
    comps = {"nccl_rs": t_rs, "cublas": t_cub, "fluxrs_gemm": t_g}
    return arms, comps, cfg
