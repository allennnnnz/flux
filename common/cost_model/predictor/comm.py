"""Communication models.

NCCL all_gather / reduce_scatter / all_reduce: one piecewise alpha-beta curve per primitive, as a
  function of the full tensor size in bytes (AG output, RS input, AR buffer). Measured times depend
  on bytes only, not on the layer: same bytes from GPT-3 and Llama layers agree within 2-3%
  (results/final_ag_points.csv, c_nccl_ag).
Flux AllGather (copy engine, All2All pull): t = alpha_sync + W * copy(bytes / W).
  W copies run back to back on one stream (local + W-1 remote, src/coll/ths_op/all_gather_op.cc:
  553-576); alpha_sync is the two barriers + memset. Keeping W explicit is what lets the model move
  to another TP size; at W=8 alpha_sync is taken from the F2 microbenchmark (flux_default minus the
  same copies without synchronisation, results/f2_ag_latency_v1/).
Optional SM-clock scaling for SM-driven collectives (NCCL): t * (f_ref / f) ** gamma; copy-engine
  transfers are not scaled (V6c). gamma defaults to 0 because the E1 data cannot separate clock from
  protocol (see report).
"""
from .curves import Curve


class CommModel:
    def __init__(self, nccl, flux_copy, flux_alpha_sync, gamma=0.0, f_ref=1410):
        self.nccl_curves = nccl          # {"ag": Curve, "rs": Curve, "ar": Curve}
        self.flux_copy = flux_copy       # Curve over per-copy bytes
        self.flux_alpha_sync = flux_alpha_sync
        self.gamma, self.f_ref = gamma, f_ref

    def nccl(self, prim, nbytes, clock=None):
        t = self.nccl_curves[prim](nbytes)
        if clock and self.gamma:
            t *= (self.f_ref / clock) ** self.gamma
        return t

    def flux_ag(self, nbytes, W):
        return self.flux_alpha_sync + W * self.flux_copy(nbytes / W)

    @classmethod
    def fit(cls, nccl_points, flux_points, flux_alpha_sync, W):
        """nccl_points: {prim: [(bytes, ms)]}; flux_points: [(total bytes, ms)] of Flux AllGather."""
        nccl = {p: Curve.fit(v) for p, v in nccl_points.items() if v}
        copy = Curve.fit([(b / W, (t - flux_alpha_sync) / W) for b, t in flux_points])
        return cls(nccl, copy, flux_alpha_sync)

    def to_dict(self):
        return {"nccl": {p: c.to_dict() for p, c in self.nccl_curves.items()},
                "flux_copy_per_shard": self.flux_copy.to_dict(),
                "flux_alpha_sync_ms": self.flux_alpha_sync, "nccl_clock_gamma": self.gamma,
                "f_ref_mhz": self.f_ref}

    @classmethod
    def from_dict(cls, d):
        return cls({p: Curve.from_dict(c) for p, c in d["nccl"].items()},
                   Curve.from_dict(d["flux_copy_per_shard"]), d["flux_alpha_sync_ms"],
                   d.get("nccl_clock_gamma", 0.0), d.get("f_ref_mhz", 1410))
