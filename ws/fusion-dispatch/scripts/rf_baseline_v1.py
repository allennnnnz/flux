################################################################################
# fusion-dispatch G1: machine-learning baseline (plan section 4, "Vidur-style random forest").
# Pure-Python random-forest regressor (CART, bootstrap, feature subsampling): no sklearn on this
# machine. One forest per (side, arm) predicts log(time) from shape features; the policy picks the
# arm with the smallest prediction. Deterministic (seeded).
################################################################################
import math
import random


class _Tree:
    def __init__(self, max_depth, min_leaf, mtry, rng):
        self.max_depth, self.min_leaf, self.mtry, self.rng = max_depth, min_leaf, mtry, rng

    def fit(self, X, y):
        self.root = self._build(list(range(len(y))), X, y, 0)
        return self

    def _build(self, idx, X, y, depth):
        vals = [y[i] for i in idx]
        mean = sum(vals) / len(vals)
        if depth >= self.max_depth or len(idx) < 2 * self.min_leaf:
            return mean
        best = None
        feats = self.rng.sample(range(len(X[0])), min(self.mtry, len(X[0])))
        for f in feats:
            order = sorted(idx, key=lambda i: X[i][f])
            ys = [y[i] for i in order]
            tot, tot2, n = sum(ys), sum(v * v for v in ys), len(ys)
            ls = ls2 = 0.0
            for j in range(n - 1):
                ls += ys[j]
                ls2 += ys[j] * ys[j]
                nl = j + 1
                if nl < self.min_leaf or n - nl < self.min_leaf:
                    continue
                if X[order[j]][f] == X[order[j + 1]][f]:
                    continue
                sse = (ls2 - ls * ls / nl) + ((tot2 - ls2) - (tot - ls) ** 2 / (n - nl))
                if best is None or sse < best[0]:
                    best = (sse, f, (X[order[j]][f] + X[order[j + 1]][f]) / 2, order[:nl], order[nl:])
        if best is None:
            return mean
        _, f, thr, left, right = best
        return (f, thr, self._build(left, X, y, depth + 1), self._build(right, X, y, depth + 1))

    def predict(self, x):
        node = self.root
        while isinstance(node, tuple):
            f, thr, left, right = node
            node = left if x[f] <= thr else right
        return node


class RandomForest:
    def __init__(self, n_trees=100, max_depth=10, min_leaf=2, mtry=None, seed=20261002):
        self.n_trees, self.max_depth, self.min_leaf, self.mtry, self.seed = n_trees, max_depth, min_leaf, mtry, seed

    def fit(self, X, y):
        rng = random.Random(self.seed)
        mtry = self.mtry or max(1, int(math.ceil(len(X[0]) / 2)))
        self.trees = []
        for _ in range(self.n_trees):
            boot = [rng.randrange(len(y)) for _ in range(len(y))]
            t = _Tree(self.max_depth, self.min_leaf, mtry, random.Random(rng.random()))
            self.trees.append(t.fit([X[i] for i in boot], [y[i] for i in boot]))
        return self

    def predict(self, x):
        return sum(t.predict(x) for t in self.trees) / len(self.trees)


def features(row):
    """Shape features of an op-level point (both sides use GEMM (M, cols, inner) and comm bytes)."""
    if row["side"] == "ag":
        M, c, inner, comm = row["M"], row["n"], row["K"], row["M"] * row["K"] * 2
    else:
        M, c, inner, comm = row["M"], row["N"], row["k"], row["M"] * row["N"] * 2
    l2 = math.log2
    return [l2(M), l2(c), l2(inner), l2(comm), l2(2.0 * M * c * inner), l2(2.0 * c * inner)]


def fit_forests(train_rows, arms_by_side):
    out = {}
    for side, arms in arms_by_side.items():
        rows = [r for r in train_rows if r["side"] == side]
        if not rows:
            continue
        X = [features(r) for r in rows]
        for a in arms:
            y = [math.log(r["arms"][a]) for r in rows]
            out[(side, a)] = RandomForest().fit(X, y)
    return out


def predict_forests(forests, row, arms):
    x = features(row)
    return {a: math.exp(forests[(row["side"], a)].predict(x)) for a in arms}
