"""GEMM time model, one parameter set per kernel family (cuBLAS, Flux AG gemm_only, Flux GemmRS).

t = t0 + (T_mem^p + T_comp^p)^(1/p)
  T_mem  = 2 * (n*K + M*K + M*n) / bw              (small M: reading the weight dominates)
  T_comp = 2 * ceil(M/tm)*tm * ceil(n/tn)*tn * K / (P_peak * eta)   (tile padding: M=520 costs
           like 640 with tm=128, which is the measured 512 -> 520 jump in E1)
  p      soft maximum (p -> inf is the plain roofline max)
P_peak = 312 TFLOP/s (A100 dense BF16). eta and bw absorb the clock / power state, so a profile
belongs to one clock state (plan: calibrate in the deployment clock state).
cuBLAS picks its own kernels: (tm, tn) is chosen from a small grid at fit time. Flux uses the
tile of the config it will actually run (flux_config.py).
"""
import math

from .fitting import msle, nelder_mead

P_PEAK = 312e12  # FLOP/s, A100 BF16 dense, 1410 MHz
B_PEAK = 2.039e12  # B/s, A100-SXM4-80GB HBM2e; the fit keeps bw below it (physical bound)
TILE_GRID = [(64, 64), (64, 128), (128, 64), (128, 128), (128, 256), (256, 128)]


class GemmFamily:
    def __init__(self, t0, bw, eta, p, tm=128, tn=128, name=""):
        self.t0, self.bw, self.eta, self.p, self.tm, self.tn, self.name = t0, bw, eta, p, tm, tn, name

    def time(self, M, n, K, tile=None):
        tm, tn = (tile[0], tile[1]) if tile else (self.tm, self.tn)
        t_mem = 2.0 * (n * K + M * K + M * n) / self.bw * 1e3
        t_comp = 2.0 * (-(-M // tm) * tm) * (-(-n // tn) * tn) * K / (P_PEAK * self.eta) * 1e3
        hi, lo = (t_mem, t_comp) if t_mem > t_comp else (t_comp, t_mem)
        return self.t0 + hi * (1.0 + (lo / hi) ** self.p) ** (1.0 / self.p)

    @classmethod
    def fit(cls, samples, name="", tile_from_sample=False):
        """samples: list of (M, n, K, tile_or_None, t_ms). Minimises mean squared log error."""
        def unpack(x):
            return (math.exp(x[0]), B_PEAK / (1 + math.exp(-x[1])), 1 / (1 + math.exp(-x[2])),
                    1 + 29 / (1 + math.exp(-x[3])))

        best = None
        grid = [None] if tile_from_sample else TILE_GRID
        for tg in grid:
            def loss(x, tg=tg):
                t0, bw, eta, p = unpack(x)
                fam = cls(t0, bw, eta, p, *(tg or (128, 128)))
                return msle([fam.time(M, n, K, tile if tile_from_sample else None)
                             for M, n, K, tile, _ in samples], [s[4] for s in samples])
            x0 = [math.log(0.02), 1.0, math.log(0.8 / 0.2), -2.5]
            x, v = nelder_mead(loss, x0, step=0.5, iters=3000)
            if best is None or v < best[1]:
                best = (x, v, tg)
        t0, bw, eta, p = unpack(best[0])
        tm, tn = best[2] or (128, 128)
        return cls(t0, bw, eta, p, tm, tn, name), best[1]

    def to_dict(self):
        return {"t0_ms": self.t0, "bw_Bps": self.bw, "eta": self.eta, "p": self.p,
                "tile_mn": [self.tm, self.tn], "name": self.name}

    @classmethod
    def from_dict(cls, d):
        return cls(d["t0_ms"], d["bw_Bps"], d["eta"], d["p"], d["tile_mn"][0], d["tile_mn"][1],
                   d.get("name", ""))
