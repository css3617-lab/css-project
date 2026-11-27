"""QoS-aware service composition model.

A problem has n abstract tasks, m candidates per task and K QoS attributes.
Each attribute has
  kind  : 'time' (response time), 'additive' (cost), 'prob' (availability/reliability),
          'bottleneck' (throughput)
  sign  : -1 lower is better, +1 higher is better
The workflow is either sequential (default) or a structured tree of
('task', i) | ('seq', [..]) | ('par', [..]) | ('xor', [..], probs) | ('loop', child, k).

Aggregation rules (Zeng et al., 2004 style):
             seq        par        xor (prob. p)    loop k
  time       sum        max        sum p*x          k*x
  additive   sum        sum        sum p*x          k*x
  prob       prod       prod       sum p*x          x**k
  bottleneck min        min        sum p*x          x
Normalization bounds are obtained by aggregating the per-task minima / maxima through the same
workflow, which is valid because every rule is monotone.
"""
import numpy as np
from . import normal as norm


class Attr:
    def __init__(self, name, values, kind, sign):
        self.name, self.v, self.kind, self.sign = name, np.asarray(values, float), kind, sign


# ------------------------------------------------------------------ workflow aggregation
def _agg_seq(vals, kind, axis=1):
    if kind in ("time", "additive"):
        return vals.sum(axis)
    if kind == "prob":
        return vals.prod(axis)
    return vals.min(axis)


def aggregate_tree(node, V, kind):
    """V: (P, n) per-task values of one attribute. Returns (P,) aggregated value."""
    t = node[0]
    if t == "task":
        return V[:, node[1]]
    if t == "loop":
        x = aggregate_tree(node[1], V, kind)
        k = node[2]
        if kind in ("time", "additive"):
            return k * x
        if kind == "prob":
            return x ** k
        return x
    ch = np.stack([aggregate_tree(c, V, kind) for c in node[1]], 1)
    if t == "seq":
        return _agg_seq(ch, kind)
    if t == "par":
        if kind == "time":
            return ch.max(1)
        if kind == "additive":
            return ch.sum(1)
        if kind == "prob":
            return ch.prod(1)
        return ch.min(1)
    if t == "xor":
        p = np.asarray(node[2])
        return (ch * p[None]).sum(1)
    raise ValueError(t)


def random_workflow(n, rng, p_par=0.3, p_xor=0.15, p_loop=0.1):
    """Random structured workflow over tasks 0..n-1 (series-parallel with choices and loops)."""
    def build(tasks):
        if len(tasks) == 1:
            node = ("task", tasks[0])
            if rng.random() < p_loop:
                node = ("loop", node, int(rng.integers(2, 4)))
            return node
        r = rng.random()
        k = int(rng.integers(2, min(4, len(tasks)) + 1))
        cuts = np.sort(rng.choice(np.arange(1, len(tasks)), k - 1, replace=False))
        parts = np.split(np.asarray(tasks), cuts)
        kids = [build(list(p)) for p in parts]
        if r < p_par:
            return ("par", kids)
        if r < p_par + p_xor:
            pr = rng.dirichlet(np.ones(len(kids)) * 2)
            return ("xor", kids, pr.tolist())
        return ("seq", kids)
    return ("seq", [build(list(range(n)))]) if n > 0 else None


# ------------------------------------------------------------------ problem
class Problem:
    """Generic QoS composition problem with global constraints on the 'time' and 'additive' attributes."""

    def __init__(self, attrs, weights, workflow=None, tight_time=0.35, tight_cost=0.50,
                 lam_c=2.0, lam_d=1.0):
        self.attrs = attrs
        self.n, self.m = attrs[0].v.shape
        self.w = np.asarray(weights, float) / np.sum(weights)
        self.workflow = workflow
        self.alive = np.ones((self.n, self.m), bool)
        self.lam_c, self.lam_d = lam_c, lam_d
        self.evals = 0
        self._bounds()
        self.cons = []   # (attr index, upper bound)
        for k, a in enumerate(attrs):
            if a.kind == "time" and tight_time is not None:
                self.cons.append((k, self.lo[k] + tight_time * (self.hi[k] - self.lo[k])))
            if a.kind == "additive" and tight_cost is not None:
                self.cons.append((k, self.lo[k] + tight_cost * (self.hi[k] - self.lo[k])))

    # aggregated value of attribute k for compositions given per-task values V (P, n)
    def _agg(self, k, V):
        if self.workflow is None:
            return _agg_seq(V, self.attrs[k].kind)
        return aggregate_tree(self.workflow, V, self.attrs[k].kind)

    def _bounds(self):
        self.lo, self.hi = [], []
        for k, a in enumerate(self.attrs):
            mn = a.v.min(1)[None]
            mx = a.v.max(1)[None]
            a1, a2 = self._agg(k, mn)[0], self._agg(k, mx)[0]
            self.lo.append(min(a1, a2)); self.hi.append(max(a1, a2))
        self.lo, self.hi = np.array(self.lo), np.array(self.hi)

    def local_score(self):
        s = np.zeros((self.n, self.m))
        for k, a in enumerate(self.attrs):
            lo, hi = a.v.min(1, keepdims=True), a.v.max(1, keepdims=True)
            x = (a.v - lo) / np.maximum(hi - lo, 1e-12)
            s += self.w[k] * (x if a.sign > 0 else 1 - x)
        return np.where(self.alive, s, 0.0)

    def aggregate(self, X):
        X = np.atleast_2d(X)
        idx = np.arange(self.n)
        Q = np.stack([self._agg(k, a.v[idx, X]) for k, a in enumerate(self.attrs)], 1)
        dead = (~self.alive[idx, X]).any(1)
        return Q, dead

    def utility(self, X):
        Q, dead = self.aggregate(X)
        rng_ = np.maximum(self.hi - self.lo, 1e-12)
        N = (Q - self.lo) / rng_
        sg = np.array([a.sign for a in self.attrs])
        N = np.where(sg > 0, N, 1 - N)
        u = N @ self.w
        viol = np.zeros(len(u))
        for k, ub in self.cons:
            viol += np.maximum(0, Q[:, k] - ub) / rng_[k]
        return u, viol, dead

    def fitness(self, X):
        u, viol, dead = self.utility(X)
        self.evals += np.atleast_2d(X).shape[0]
        return u - self.lam_c * viol - self.lam_d * dead

    def describe(self, X):
        u, v, d = self.utility(X)
        return float(u[0]), bool(v[0] == 0 and not d[0])


# ------------------------------------------------------------------ instance generators
W4 = (0.30, 0.20, 0.25, 0.25)


def synthetic(n, m, seed, corr=0.5, workflow=None, tight_time=0.35, tight_cost=0.50, weights=W4):
    """Correlated synthetic QoS (latent quality drives all attributes)."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, m))
    drv = lambda: corr * z + np.sqrt(1 - corr ** 2) * rng.standard_normal((n, m))
    rt = np.clip(600 - 200 * drv(), 100, 1500)
    cost = np.clip(1.5 + 1.0 * drv(), 0.10, 5.0)
    av = 0.85 + (0.999 - 0.85) * norm.cdf(drv())
    rel = 0.70 + (0.99 - 0.70) * norm.cdf(drv())
    attrs = [Attr("rt", rt, "time", -1), Attr("cost", cost, "additive", -1),
             Attr("avail", av, "prob", +1), Attr("rel", rel, "prob", +1)]
    return Problem(attrs, weights, workflow, tight_time, tight_cost)


W5 = (0.25, 0.15, 0.20, 0.20, 0.20)


class WSDream1:
    """WS-DREAM dataset #1 (339 users x 5825 services, real RT and throughput)."""

    def __init__(self, folder):
        import os
        self.rt = np.loadtxt(os.path.join(folder, "rtMatrix.txt"))
        self.tp = np.loadtxt(os.path.join(folder, "tpMatrix.txt"))
        valid = self.rt >= 0
        self.avail = valid.mean(0)                                    # observed success rate per service
        ok = (self.rt >= 0) & (self.rt <= 5.0)
        self.rel = ok.sum(0) / np.maximum(valid.sum(0), 1)            # on-time rate (<= 5 s) per service

    def instance(self, n, m, seed, workflow=None, tight_time=0.35, tight_cost=0.50, weights=W5):
        rng = np.random.default_rng(seed)
        while True:
            u = int(rng.integers(0, self.rt.shape[0]))
            good = np.where((self.rt[u] > 0) & (self.tp[u] > 0) & (self.avail > 0.5) & (self.rel > 0.2))[0]
            if len(good) >= n * m:
                break
        S = rng.choice(good, n * m, replace=False).reshape(n, m)
        rt = self.rt[u][S]
        tp = self.tp[u][S]
        av = np.clip(self.avail[S], 0.5, 0.9999)
        rel = np.clip(self.rel[S], 0.2, 0.9999)
        # WS-DREAM has no prices: cost is synthetic and grows with observed quality (documented in the paper)
        q = (norm.ppf(np.clip(av, 1e-4, 1 - 1e-4)) + np.log(tp) / 3) / 2
        q = (q - q.mean()) / (q.std() + 1e-12)
        cost = np.clip(1.5 + 0.5 * q + 0.5 * rng.standard_normal((n, m)), 0.10, 5.0)
        attrs = [Attr("rt", rt, "time", -1), Attr("tp", tp, "bottleneck", +1), Attr("avail", av, "prob", +1),
                 Attr("rel", rel, "prob", +1), Attr("cost", cost, "additive", -1)]
        p = Problem(attrs, weights, workflow, tight_time, tight_cost)
        p.user, p.services = u, S
        return p
