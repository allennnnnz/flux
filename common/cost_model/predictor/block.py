"""Layout model: vLLM default (TP + all-reduce) vs sequence parallel (SP) for one transformer block.

Per block (attention and SiLU*up cost the same in both layouts and cancel):
  TP+AR  4 cuBLAS GEMMs on all M rows (QKV, O, gate_up, down) + 2 vLLM all-reduces of M x H
         + 2 RMSNorm + 2 residual adds on M rows
  SP     the 4 chosen op paths (AG+QKV, O+RS, AG+gate_up, down+RS; each already includes its GEMM)
         + 2 RMSNorm + 2 residual adds on M / W rows
delta = T(TP+AR) - T(SP) > 0 means SP is faster. Everything is in ms per block.
The all-reduce curve and the elementwise costs come from calibrate_block_v1.py (vLLM venv):
  BlockProfile.ar(bytes)            piecewise alpha-beta curve of vLLM's tensor_model_parallel_all_reduce
  BlockProfile.rmsnorm(rows, H) / add(rows, H)   t0 + bytes / bw fitted on (M, 6144)
"""
import json

from .curves import Curve


class BlockProfile:
    def __init__(self, W, ar_curve, norm, add, meta=None):
        self.W, self.ar_curve, self.norm_p, self.add_p, self.meta = W, ar_curve, norm, add, meta or {}

    def ar(self, nbytes):
        return self.ar_curve(nbytes)

    def rmsnorm(self, rows, H):
        return self.norm_p[0] + rows * H * 2 * 2 / self.norm_p[1]  # read + write

    def add(self, rows, H):
        return self.add_p[0] + rows * H * 2 * 3 / self.add_p[1]    # 2 reads + 1 write

    @staticmethod
    def _linfit(pts):
        """t = t0 + bytes / bw by least squares over (bytes, ms) points."""
        n = len(pts)
        mx = sum(b for b, _ in pts) / n
        my = sum(t for _, t in pts) / n
        slope = sum((b - mx) * (t - my) for b, t in pts) / sum((b - mx) ** 2 for b, _ in pts)
        return (max(0.0, my - slope * mx), 1.0 / slope)

    @classmethod
    def fit(cls, W, ar_pts, norm_pts, add_pts, meta=None):
        """ar_pts [(bytes, ms)]; norm_pts / add_pts [(rows, H, ms)]."""
        norm = cls._linfit([(r * H * 2 * 2, t) for r, H, t in norm_pts])
        add = cls._linfit([(r * H * 2 * 3, t) for r, H, t in add_pts])
        return cls(W, Curve.fit(ar_pts), norm, add, meta)

    def to_dict(self):
        return {"W": self.W, "ar": self.ar_curve.to_dict(), "rmsnorm_t0_ms_bw_Bpms": list(self.norm_p),
                "add_t0_ms_bw_Bpms": list(self.add_p), "meta": self.meta}

    def save(self, path):
        json.dump(self.to_dict(), open(path, "w"), indent=1)

    @classmethod
    def load(cls, path):
        d = json.load(open(path))
        return cls(d["W"], Curve.from_dict(d["ar"]), tuple(d["rmsnorm_t0_ms_bw_Bpms"]), tuple(d["add_t0_ms_bw_Bpms"]),
                   d.get("meta"))


def layout_times(bp, M, H, cublas_tp, sp_paths):
    """cublas_tp: the 4 TP GEMM times on M rows (ms); sp_paths: the 4 chosen SP op path times (ms).
    Returns (t_tp_ar, t_sp, delta) per block, attention / SiLU excluded (identical in both)."""
    W = bp.W
    t_tp = sum(cublas_tp) + 2 * bp.ar(M * H * 2) + 2 * bp.rmsnorm(M, H) + 2 * bp.add(M, H)
    t_sp = sum(sp_paths) + 2 * bp.rmsnorm(M // W, H) + 2 * bp.add(M // W, H)
    return t_tp, t_sp, t_tp - t_sp
