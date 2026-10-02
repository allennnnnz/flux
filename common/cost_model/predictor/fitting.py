"""Small pure-Python optimizer (Nelder-Mead) for fitting a handful of model parameters."""
import math


def nelder_mead(f, x0, step=0.2, iters=2000, tol=1e-10):
    n = len(x0)
    pts = [list(x0)]
    for i in range(n):
        p = list(x0)
        p[i] += step if p[i] == 0 else step * abs(p[i]) + 1e-3
        pts.append(p)
    vals = [f(p) for p in pts]
    for _ in range(iters):
        order = sorted(range(n + 1), key=lambda k: vals[k])
        pts = [pts[k] for k in order]
        vals = [vals[k] for k in order]
        if abs(vals[-1] - vals[0]) < tol:
            break
        cen = [sum(p[i] for p in pts[:-1]) / n for i in range(n)]
        refl = [cen[i] + (cen[i] - pts[-1][i]) for i in range(n)]
        fr = f(refl)
        if fr < vals[0]:
            exp = [cen[i] + 2 * (cen[i] - pts[-1][i]) for i in range(n)]
            fe = f(exp)
            pts[-1], vals[-1] = (exp, fe) if fe < fr else (refl, fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = refl, fr
        else:
            con = [cen[i] + 0.5 * (pts[-1][i] - cen[i]) for i in range(n)]
            fc = f(con)
            if fc < vals[-1]:
                pts[-1], vals[-1] = con, fc
            else:
                for k in range(1, n + 1):
                    pts[k] = [pts[0][i] + 0.5 * (pts[k][i] - pts[0][i]) for i in range(n)]
                    vals[k] = f(pts[k])
    best = min(range(n + 1), key=lambda k: vals[k])
    return pts[best], vals[best]


def msle(pred, meas):
    """mean squared log error (relative error, symmetric for small and large M)."""
    s = 0.0
    for p, m in zip(pred, meas):
        if p <= 0:
            return 1e9
        s += math.log(p / m) ** 2
    return s / max(1, len(meas))
