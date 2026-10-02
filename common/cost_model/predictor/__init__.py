"""Analytical cost model for Flux on/off and comm-path decisions (fusion-dispatch phase 3, D-008).

time(path) = communication + computation - overlap, each term a small physical model:
  curves.py      piecewise alpha-beta curves (bytes -> ms)
  comm.py        NCCL collectives; Flux AllGather = alpha_sync + W * copy(bytes / W)
  gemm.py        roofline + tile padding, one parameter set per kernel family
  flux_config.py which tuned / fallback CUTLASS config Flux runs for a shape
  overlap.py     fused AG+GEMM = event simulation of the stream-K tile schedule; GemmRS model
  paths.py       predicted time of each op-level path
  profile.py     fitted parameters + provenance (JSON); fitting from component samples
Pure Python (no numpy). Design: ws/fusion-dispatch/reports/20261002_plan_general_dispatcher.md.
"""
from .flux_config import FluxConfigs
from .paths import AG_ARMS, FLUX_GEMM_ARMS, RS_ARMS, predict_ag, predict_rs
from .profile import HardwareProfile, fit_profile

__all__ = ["FluxConfigs", "HardwareProfile", "fit_profile", "predict_ag", "predict_rs", "AG_ARMS",
           "RS_ARMS", "FLUX_GEMM_ARMS"]
