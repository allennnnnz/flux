"""Hardware profile: every fitted parameter of the predictor, with provenance, saved as JSON
(common/cost_model/hw_profiles/<host>_tp<W>_<state>.json, or a workstream results dir).

Fitting takes generic component samples, so the same code fits from op-map component columns (G1)
or from calibration microbenchmarks (G2).
"""
import json
import math
import statistics
import time

from .comm import CommModel
from .fitting import msle, nelder_mead
from .gemm import B_PEAK, GemmFamily
from .overlap import fused_ag_time, fused_rs_time


class HardwareProfile:
    def __init__(self, W, comm, gemm, fused_ag, fused_rs, meta=None):
        self.W, self.comm, self.gemm = W, comm, gemm
        self.fused_ag = fused_ag   # {"t_k0": ms, "local_bw": B/ms, "d_tail": ms, "kappa": -}
        self.fused_rs = fused_rs   # {"alpha_rs": ms, "beta_scat": bytes/ms}
        self.meta = meta or {}

    def to_dict(self):
        return {"meta": self.meta, "W": self.W, "comm": self.comm.to_dict(),
                "gemm": {k: v.to_dict() for k, v in self.gemm.items()},
                "fused_ag": self.fused_ag, "fused_rs": self.fused_rs}

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=1)

    @classmethod
    def load(cls, path):
        d = json.load(open(path))
        return cls(d["W"], CommModel.from_dict(d["comm"]),
                   {k: GemmFamily.from_dict(v) for k, v in d["gemm"].items()},
                   d["fused_ag"], d["fused_rs"], d.get("meta"))


NSYS_KERNEL_START = {"t_k0": 0.0317, "local_bw": 838e6,
                     "source": "results/v8_nsys/parse_all.txt + results/e2_nsys/parse_G-FC1.txt: "
                               "A_fused GEMM start after first copy, linear in local shard bytes"}


def fit_fused_ag(samples, W, kernel_start=NSYS_KERNEL_START, kappa_grid=None):
    """samples: [(M, n, K, cfg, t_flux_ag, t_gemm, t_A)], all measured.
    d_tail: closed form from shapes where one tile row spans every shard (no overlap possible,
    so t_A = t_flux_ag + t_gemm - d_tail). kappa: 1-D grid on the remaining samples (M >= 1024)."""
    single = [s for s in samples if -(-s[0] // s[3]["tile"][0]) == 1]
    d_tail = statistics.median(s[4] + s[5] - s[6] for s in single) if single else 0.014
    multi = [s for s in samples if -(-s[0] // s[3]["tile"][0]) > 1 and s[0] >= 1024] or samples
    prm = {"t_k0": kernel_start["t_k0"], "local_bw": kernel_start["local_bw"], "d_tail": d_tail}
    best = None
    for kappa in kappa_grid or [0.025 * i for i in range(41)]:
        prm["kappa"] = kappa
        pred = [fused_ag_time(M, n, K, W, cfg, fa, g, prm) for M, n, K, cfg, fa, g, _ in multi]
        # median |log error|: robust to config cliffs where the fused kernel is slower than the
        # serial path for reasons the schedule model does not contain (e.g. G-FC1 M=3072)
        v = statistics.median(abs(math.log(p / s[6])) for p, s in zip(pred, multi))
        if best is None or v < best[1]:
            best = (kappa, v)
    prm["kappa"] = best[0]
    prm.update({"n_single": len(single), "n_multi": len(multi),
                "kernel_start_source": kernel_start["source"]})
    return prm


def fit_fused_rs(samples, W):
    """samples: [(M, N, k, cfg, t_A)]. Fits the GemmRS GEMM family jointly with alpha_rs and the
    scatter bandwidth (GemmRS communication cannot be isolated on sm80: CLAUDE.md trap table)."""
    def unpack(x):
        return (math.exp(x[0]), B_PEAK / (1 + math.exp(-x[1])), 1 / (1 + math.exp(-x[2])),
                1 + 29 / (1 + math.exp(-x[3])), math.exp(x[4]), math.exp(x[5]))

    def model(x):
        t0, bw, eta, p, a, b = unpack(x)
        fam = GemmFamily(t0, bw, eta, p, name="flux_rs")
        return fam, a, b

    def loss(x):
        fam, a, b = model(x)
        return msle([fused_rs_time(M, N, W, fam.time(M, N, k, cfg["tile"]), a, b)
                     for M, N, k, cfg, _ in samples], [s[4] for s in samples])
    x0 = [math.log(0.02), 1.0, math.log(0.8 / 0.2), -2.5, math.log(0.03), math.log(3e8)]
    x, v = nelder_mead(loss, x0, step=0.5, iters=6000)
    x, v = nelder_mead(loss, x, step=0.2, iters=6000)
    fam, a, b = model(x)
    return fam, {"alpha_rs": a, "beta_scat": b, "fit_msle": v}


def fit_profile(W, nccl_points, flux_ag_points, flux_alpha_sync, cublas_samples, flux_ag_samples,
                fused_ag_samples, fused_rs_samples, meta=None):
    comm = CommModel.fit(nccl_points, flux_ag_points, flux_alpha_sync, W)
    cub, _ = GemmFamily.fit(cublas_samples, "cublas")
    fag, _ = GemmFamily.fit(flux_ag_samples, "flux_ag", tile_from_sample=True)
    fused_ag = fit_fused_ag(fused_ag_samples, W)
    frs, fused_rs = fit_fused_rs(fused_rs_samples, W)
    meta = dict(meta or {})
    meta.setdefault("date", time.strftime("%Y-%m-%d"))
    return HardwareProfile(W, comm, {"cublas": cub, "flux_ag": fag, "flux_rs": frs}, fused_ag,
                           fused_rs, meta)
