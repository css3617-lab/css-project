"""Optimizers for QoS-aware service composition under a common fitness-evaluation budget.

Every optimizer receives a Problem, a budget (number of fitness evaluations), a numpy Generator and a
logging step, and returns a Tracker with the best composition and the best-so-far curve.
Continuous-encoding methods (PSO, DE, GWO, WOA and the PSO phase of IPAA) share the same
heuristic-ordered decoding, so differences come from the search dynamics, not from the encoding.
"""
import numpy as np


class Tracker:
    def __init__(self, prob, budget, step):
        self.p, self.budget, self.step = prob, budget, step
        self.start = prob.evals
        self.best, self.bx = -np.inf, None
        self.curve = []
        self.next = step

    def used(self):
        return self.p.evals - self.start

    def update(self, X, f):
        i = int(np.argmax(f))
        if f[i] > self.best:
            self.best, self.bx = float(f[i]), np.array(X[i]).copy()
        while self.used() >= self.next and self.next <= self.budget:
            self.curve.append(self.best)
            self.next += self.step

    def done(self):
        return self.used() >= self.budget

    def frac(self):
        return min(self.used() / self.budget, 1.0)


def sorted_order(prob):
    return np.argsort(-prob.local_score(), axis=1)


def decode(prob, order, P):
    j = np.minimum((P * prob.m).astype(int), prob.m - 1)
    return order[np.arange(prob.n), j]


# ------------------------------------------------------------------ baselines
def random_search(prob, budget, rng, step, pop=40, **kw):
    T = Tracker(prob, budget, step)
    while not T.done():
        X = rng.integers(0, prob.m, (pop, prob.n))
        T.update(X, prob.fitness(X))
    return T


def ga(prob, budget, rng, step, pop=40, pc=0.9, k=3, **kw):
    T = Tracker(prob, budget, step)
    n, m = prob.n, prob.m
    pm = 1.0 / n
    X = rng.integers(0, m, (pop, n)); f = prob.fitness(X); T.update(X, f)
    while not T.done():
        def tour():
            c = rng.integers(0, pop, (pop, k))
            return c[np.arange(pop), np.argmax(f[c], 1)]
        A, B = X[tour()], X[tour()]
        mask = rng.random((pop, n)) < 0.5
        cross = rng.random(pop) < pc
        C = np.where(mask & cross[:, None], B, A)
        mut = rng.random((pop, n)) < pm
        C = np.where(mut, rng.integers(0, m, (pop, n)), C)
        fc = prob.fitness(C); T.update(C, fc)
        e = int(np.argmax(f))
        C[0], fc[0] = X[e], f[e]
        X, f = C, fc
    return T


def de(prob, budget, rng, step, pop=40, F=0.5, CR=0.9, **kw):
    """DE/rand/1/bin on the heuristic-ordered continuous encoding."""
    T = Tracker(prob, budget, step)
    order = sorted_order(prob)
    n = prob.n
    P = rng.random((pop, n))
    f = prob.fitness(decode(prob, order, P)); T.update(decode(prob, order, P), f)
    while not T.done():
        r = np.array([rng.choice(pop, 3, replace=False) for _ in range(pop)])
        V = P[r[:, 0]] + F * (P[r[:, 1]] - P[r[:, 2]])
        V = np.mod(V, 1.0)
        cross = rng.random((pop, n)) < CR
        cross[np.arange(pop), rng.integers(0, n, pop)] = True
        U = np.where(cross, V, P)
        XU = decode(prob, order, U)
        fu = prob.fitness(XU); T.update(XU, fu)
        better = fu >= f
        P[better], f[better] = U[better], fu[better]
    return T


def gwo(prob, budget, rng, step, pop=40, **kw):
    """Grey wolf optimizer on the heuristic-ordered continuous encoding."""
    T = Tracker(prob, budget, step)
    order = sorted_order(prob)
    n = prob.n
    P = rng.random((pop, n))
    while not T.done():
        X = decode(prob, order, P)
        f = prob.fitness(X); T.update(X, f)
        lead = P[np.argsort(-f)[:3]]
        a = 2.0 * (1 - T.frac())
        new = np.zeros_like(P)
        for L in lead:
            A = 2 * a * rng.random((pop, n)) - a
            C = 2 * rng.random((pop, n))
            new += L[None] - A * np.abs(C * L[None] - P)
        P = np.clip(new / 3.0, 0, 0.999999)
    return T


def woa(prob, budget, rng, step, pop=40, b=1.0, **kw):
    """Whale optimization algorithm on the heuristic-ordered continuous encoding."""
    T = Tracker(prob, budget, step)
    order = sorted_order(prob)
    n = prob.n
    P = rng.random((pop, n))
    best, bf = None, -np.inf
    while not T.done():
        X = decode(prob, order, P)
        f = prob.fitness(X); T.update(X, f)
        i = int(np.argmax(f))
        if f[i] > bf:
            bf, best = f[i], P[i].copy()
        a = 2.0 * (1 - T.frac())
        A = 2 * a * rng.random((pop, 1)) - a
        C = 2 * rng.random((pop, 1))
        p = rng.random((pop, 1))
        l = rng.uniform(-1, 1, (pop, 1))
        rnd = P[rng.integers(0, pop, pop)]
        enc_best = best[None] - A * np.abs(C * best[None] - P)
        enc_rand = rnd - A * np.abs(C * rnd - P)
        spiral = np.abs(best[None] - P) * np.exp(b * l) * np.cos(2 * np.pi * l) + best[None]
        P = np.where(p < 0.5, np.where(np.abs(A) < 1, enc_best, enc_rand), spiral)
        P = np.clip(P, 0, 0.999999)
    return T


def sa(prob, budget, rng, step, pop=40, t_end=1e-4, **kw):
    """Population of independent simulated-annealing chains (vectorized), single-task moves."""
    T = Tracker(prob, budget, step)
    n, m = prob.n, prob.m
    X = rng.integers(0, m, (pop, n)); f = prob.fitness(X); T.update(X, f)
    # initial temperature: mean absolute change of a random move ~ accepted with prob 0.5
    Y = X.copy(); Y[np.arange(pop), rng.integers(0, n, pop)] = rng.integers(0, m, pop)
    d = np.abs(prob.fitness(Y) - f); T.update(Y, f)
    t0 = max(np.mean(d) / np.log(2), 1e-6)
    while not T.done():
        temp = t0 * (t_end / t0) ** T.frac()
        Y = X.copy()
        Y[np.arange(pop), rng.integers(0, n, pop)] = rng.integers(0, m, pop)
        fy = prob.fitness(Y); T.update(Y, fy)
        acc = (fy >= f) | (rng.random(pop) < np.exp(np.minimum((fy - f) / temp, 0.0)))
        X[acc], f[acc] = Y[acc], fy[acc]
    return T


# ------------------------------------------------------------------ PSO
class PSOState:
    def __init__(self, prob, rng, pop, order):
        self.order = order
        self.P = rng.random((pop, prob.n))
        self.V = (rng.random((pop, prob.n)) - 0.5) * 0.2
        self.pb = self.P.copy(); self.pbf = np.full(pop, -np.inf)
        self.gb = None; self.gbf = -np.inf


def _pso_iter(prob, S, rng, T, w, c1=1.49445, c2=1.49445, vmax=0.25):
    X = decode(prob, S.order, S.P)
    f = prob.fitness(X); T.update(X, f)
    imp = f > S.pbf
    S.pb[imp], S.pbf[imp] = S.P[imp], f[imp]
    g = int(np.argmax(S.pbf))
    if S.pbf[g] > S.gbf:
        S.gb, S.gbf = S.pb[g].copy(), S.pbf[g]
    r1, r2 = rng.random(S.P.shape), rng.random(S.P.shape)
    S.V = np.clip(w * S.V + c1 * r1 * (S.pb - S.P) + c2 * r2 * (S.gb - S.P), -vmax, vmax)
    S.P = np.clip(S.P + S.V, 0, 0.999999)


def pso(prob, budget, rng, step, pop=40, **kw):
    T = Tracker(prob, budget, step)
    S = PSOState(prob, rng, pop, sorted_order(prob))
    while not T.done():
        _pso_iter(prob, S, rng, T, 0.9 - 0.5 * T.frac())
    return T


# ------------------------------------------------------------------ MMAS
class ACOState:
    def __init__(self, prob, alpha=1.0, beta=2.5, rho=0.2, tau0=None, p_best=0.05):
        self.eta = np.maximum(prob.local_score(), 1e-3)
        self.alpha, self.beta, self.rho = alpha, beta, rho
        self.tau = np.ones((prob.n, prob.m)) if tau0 is None else tau0.copy()
        p = p_best ** (1.0 / prob.n)
        self.tmax = 1.0
        self.tmin = self.tmax * (1 - p) / ((prob.m - 1) * p)


def _construct(prob, A, rng, ants):
    W = (A.tau ** A.alpha) * (A.eta ** A.beta)
    W = np.where(prob.alive, W, 0) + 1e-12
    cdf = np.cumsum(W / W.sum(1, keepdims=True), 1)
    u = rng.random((ants, prob.n))
    X = np.empty((ants, prob.n), int)
    for i in range(prob.n):
        X[:, i] = np.searchsorted(cdf[i], u[:, i])
    return np.minimum(X, prob.m - 1)


def _deposit(prob, A, xbest):
    A.tau *= (1 - A.rho)
    A.tau[np.arange(prob.n), xbest] += A.rho * A.tmax
    np.clip(A.tau, A.tmin, A.tmax, out=A.tau)


def aco(prob, budget, rng, step, pop=40, alpha=1.0, beta=2.5, rho=0.2, **kw):
    T = Tracker(prob, budget, step)
    A = ACOState(prob, alpha, beta, rho)
    while not T.done():
        X = _construct(prob, A, rng, pop)
        f = prob.fitness(X); T.update(X, f)
        _deposit(prob, A, X[int(np.argmax(f))] if rng.random() < 0.5 else T.bx)
    return T


def pso_aco_fixed(prob, budget, rng, step, pop=40, **kw):
    """Sequential relay hybrid: PSO for 50% of the budget, then MMAS from uniform pheromone + gbest."""
    T = Tracker(prob, budget, step)
    S = PSOState(prob, rng, pop, sorted_order(prob))
    while T.used() < 0.5 * budget:
        _pso_iter(prob, S, rng, T, 0.9 - 0.5 * T.frac())
    A = ACOState(prob)
    _deposit(prob, A, T.bx)
    while not T.done():
        X = _construct(prob, A, rng, pop)
        f = prob.fitness(X); T.update(X, f)
        _deposit(prob, A, X[int(np.argmax(f))] if rng.random() < 0.5 else T.bx)
    return T


# ------------------------------------------------------------------ IPAA (proposed)
def ipaa(prob, budget, rng, step, pop=40, window=8, eps=1e-3, div_min=0.02, elite=10,
         seeding=True, adaptive=True, tau_init=None, info=None, seed_base=0.3, seed_gain=0.2,
         min_pso=0.05, max_pso=0.6, alpha=1.0, beta=2.5, rho=0.2, xover=0.5, skip_pso=False, **kw):
    T = Tracker(prob, budget, step)
    if skip_pso and tau_init is None:
        tau_init = np.ones((prob.n, prob.m))
    if tau_init is None:
        S = PSOState(prob, rng, pop, sorted_order(prob))
        hist = []
        while not T.done():
            _pso_iter(prob, S, rng, T, 0.9 - 0.5 * T.frac())
            hist.append(T.best)
            if adaptive:
                stag = len(hist) > window and (hist[-1] - hist[-1 - window]) < eps
                div = S.P.std(0).mean()
                if (stag or div < div_min) and T.used() > min_pso * budget:
                    break
                if T.used() >= max_pso * budget:
                    break
            elif T.used() >= 0.5 * budget:
                break
        switch_at = T.used()
        A = ACOState(prob, alpha, beta, rho)
        if seeding:
            A.tau[:] = A.tmax * seed_base
            idx = np.argsort(-S.pbf)[:elite]
            Xe = decode(prob, S.order, S.pb[idx])
            E = len(Xe)
            for r, xe in enumerate(Xe):
                A.tau[np.arange(prob.n), xe] += A.tmax * seed_gain * (E - r) / E
            A.tau[np.arange(prob.n), T.bx] += A.tmax * seed_gain
            np.clip(A.tau, A.tmin, A.tmax, out=A.tau)
        else:
            _deposit(prob, A, T.bx)
    else:
        A = ACOState(prob, alpha, beta, rho, tau0=tau_init)
        switch_at = 0
    while not T.done():
        X = _construct(prob, A, rng, pop)
        if xover > 0 and T.bx is not None:
            # elite recombination: a share of the ants inherit each task from the best-so-far plan with prob. 1/2
            k = int(round(xover * pop))
            mask = rng.random((k, prob.n)) < 0.5
            X[:k] = np.where(mask, T.bx[None], X[:k])
        f = prob.fitness(X); T.update(X, f)
        _deposit(prob, A, X[int(np.argmax(f))] if rng.random() < 0.5 else T.bx)
    if info is not None:
        info["switch"] = int(switch_at)
        info["tau"] = A.tau
    return T


ALGOS = {
    "RS": random_search, "GA": ga, "DE": de, "SA": sa, "PSO": pso, "GWO": gwo, "WOA": woa,
    "ACO": aco, "PSO-ACO": pso_aco_fixed, "IPAA": ipaa,
}
ALG_ORDER = ["RS", "GA", "DE", "SA", "PSO", "GWO", "WOA", "ACO", "PSO-ACO", "IPAA"]
