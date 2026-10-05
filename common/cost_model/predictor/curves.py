"""Piecewise alpha-beta curves: time as a function of bytes, fitted from measured points.

Between knots the curve is linear in bytes, i.e. each segment is t = alpha_i + bytes / beta_i
(NCCL switches LL / LL128 / Simple with size, so one alpha-beta pair does not fit; see
ws/fusion-dispatch/reports/20261002_related_work.md section 4). Below the first knot the curve is
flat (latency floor); above the last knot it extends the last segment (bandwidth regime).
Pure Python on purpose: it must run in the pixi env, the vLLM venv and plain python3.
"""
import bisect
import statistics


class Curve:
    def __init__(self, xs, ts):
        assert len(xs) == len(ts) and xs, "empty curve"
        self.xs = [float(x) for x in xs]
        self.ts = [float(t) for t in ts]

    def __call__(self, x):
        xs, ts = self.xs, self.ts
        if x <= xs[0] or len(xs) == 1:
            return ts[0]
        if x >= xs[-1]:
            slope = max(0.0, (ts[-1] - ts[-2]) / (xs[-1] - xs[-2]))
            return ts[-1] + slope * (x - xs[-1])
        i = bisect.bisect_right(xs, x) - 1
        f = (x - xs[i]) / (xs[i + 1] - xs[i])
        return ts[i] + f * (ts[i + 1] - ts[i])

    @classmethod
    def fit(cls, points, rel_merge=0.10, monotone=True):
        """points: iterable of (bytes, ms). Points whose sizes are within rel_merge of a group's
        first size are merged (geometric-mean size, median time); then pool-adjacent-violators
        makes the curve non-decreasing (a larger message never takes less time).
        monotone=False (E4a2, 2026-10-05) skips that step: across css-host-158 + 159 NCCL all-reduce is
        NOT monotone in size (TP4 2+2: 1.5 MiB 0.750 ms, 3 MiB 0.404 ms; ws/fusion-dispatch/reports/
        20261005_e4a_slow_link_layout.md section 3), and pooling smeared that cliff onto its neighbours."""
        pts = sorted((float(x), float(t)) for x, t in points if x > 0 and t is not None)
        assert pts, "no points to fit"
        groups = [[pts[0]]]
        for x, t in pts[1:]:
            if x <= groups[-1][0][0] * (1 + rel_merge):
                groups[-1].append((x, t))
            else:
                groups.append([(x, t)])
        xs, ts, ws = [], [], []
        for g in groups:
            gx = [p[0] for p in g]
            prod = 1.0
            for v in gx:
                prod *= v ** (1.0 / len(gx))
            xs.append(prod)
            ts.append(statistics.median(p[1] for p in g))
            ws.append(len(g))
        if monotone:
            ts = _pav(ts, ws)
        return cls(xs, ts)

    def to_dict(self):
        return {"x_bytes": [round(x, 1) for x in self.xs], "t_ms": [round(t, 6) for t in self.ts]}

    @classmethod
    def from_dict(cls, d):
        return cls(d["x_bytes"], d["t_ms"])


def _pav(ys, ws):
    """Weighted isotonic (non-decreasing) regression, pool-adjacent-violators."""
    blocks = []  # [value, weight, count]
    for y, w in zip(ys, ws):
        blocks.append([y, w, 1])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            v2, w2, c2 = blocks.pop()
            v1, w1, c1 = blocks.pop()
            blocks.append([(v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2, c1 + c2])
    out = []
    for v, _, c in blocks:
        out.extend([v] * c)
    return out
