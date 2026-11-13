"""Optimisers. All share the same evaluator, initial population and NFE budget."""
import numpy as np
from sim import N_OPT, M, U

POP = 30


class Tracker:
    def __init__(self, ev, budget):
        self.ev, self.budget = ev, budget
        self.best_key, self.best_x = np.inf, None
        self.trace = []                      # (nfe, best_key)

    def eval(self, X):
        k, F, v = self.ev(X)
        i = int(np.argmin(k))
        if k[i] < self.best_key:
            self.best_key, self.best_x = float(k[i]), X[i].copy()
        self.trace.append((self.ev.nfe, self.best_key))
        return k, F, v

    def left(self, n=1):
        return self.ev.nfe + n <= self.budget


def init_pop(scen, rng, P=POP):
    X = rng.random((P, scen.dim))
    X[0] = scen.greedy_vector()          # min-estimated-latency heuristic
    X[1] = scen.nearest_edge_vector()
    X[2] = scen.local_vector()
    return X


# ---------------------------------------------------------------- baselines
def random_search(ev, scen, budget, rng, **kw):
    T = Tracker(ev, budget)
    T.eval(init_pop(scen, rng))
    while T.left(POP):
        T.eval(rng.random((POP, scen.dim)))
    return T


def greedy(ev, scen, budget, rng, **kw):
    T = Tracker(ev, budget)
    T.eval(scen.nearest_edge_vector()[None])
    return T


def pso(ev, scen, budget, rng, adaptive=False, **kw):
    T = Tracker(ev, budget)
    D = scen.dim
    X = init_pop(scen, rng)
    V = rng.uniform(-0.1, 0.1, X.shape)
    k, _, _ = T.eval(X)
    pb, pk = X.copy(), k.copy()
    vmax = 0.2
    while T.left(POP):
        frac = ev.nfe / budget
        if adaptive:      # TVAC-PSO (Ratnaweera et al.)
            w = 0.9 - 0.5 * frac; c1 = 2.5 - 2.0 * frac; c2 = 0.5 + 2.0 * frac
        else:             # constriction-equivalent standard PSO
            w, c1, c2 = 0.7298, 1.49618, 1.49618
        g = pb[np.argmin(pk)]
        V = w * V + c1 * rng.random(X.shape) * (pb - X) + c2 * rng.random(X.shape) * (g - X)
        V = np.clip(V, -vmax, vmax)
        X = np.clip(X + V, 0, 1 - 1e-9)
        k, _, _ = T.eval(X)
        imp = k < pk
        pb[imp], pk[imp] = X[imp], k[imp]
    return T


def apso(ev, scen, budget, rng, **kw):
    return pso(ev, scen, budget, rng, adaptive=True)


def ga(ev, scen, budget, rng, **kw):
    T = Tracker(ev, budget)
    D = scen.dim
    X = init_pop(scen, rng)
    k, _, _ = T.eval(X)
    eta_c, eta_m, pc, pm = 15.0, 20.0, 0.9, 1.0 / D
    while T.left(POP - 2):
        def tour():
            a, b = rng.integers(0, POP, 2)
            return X[a] if k[a] < k[b] else X[b]
        kids = []
        while len(kids) < POP - 2:
            p1, p2 = tour(), tour()
            c1, c2 = p1.copy(), p2.copy()
            if rng.random() < pc:      # SBX
                u = rng.random(D)
                beta = np.where(u <= 0.5, (2 * u) ** (1 / (eta_c + 1)), (1 / (2 * (1 - u))) ** (1 / (eta_c + 1)))
                c1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
                c2 = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)
            for c in (c1, c2):         # polynomial mutation
                m = rng.random(D) < pm
                u = rng.random(D)
                dl = np.where(u < 0.5, (2 * u) ** (1 / (eta_m + 1)) - 1, 1 - (2 * (1 - u)) ** (1 / (eta_m + 1)))
                c[m] += dl[m]
                kids.append(np.clip(c, 0, 1 - 1e-9))
        kids = np.array(kids[:POP - 2])
        kk, _, _ = T.eval(kids)
        el = np.argsort(k)[:2]
        X = np.vstack([X[el], kids]); k = np.concatenate([k[el], kk])
    return T


def de(ev, scen, budget, rng, **kw):
    T = Tracker(ev, budget)
    D = scen.dim
    X = init_pop(scen, rng)
    k, _, _ = T.eval(X)
    Fm, CR = 0.5, 0.9
    while T.left(POP):
        idx = np.array([rng.choice(np.delete(np.arange(POP), i), 3, replace=False) for i in range(POP)])
        Vm = X[idx[:, 0]] + Fm * (X[idx[:, 1]] - X[idx[:, 2]])
        cross = rng.random(X.shape) < CR
        cross[np.arange(POP), rng.integers(0, D, POP)] = True
        Tr = np.clip(np.where(cross, Vm, X), 0, 1 - 1e-9)
        kt, _, _ = T.eval(Tr)
        imp = kt < k
        X[imp], k[imp] = Tr[imp], kt[imp]
    return T


# ---------------------------------------------------------------- ACO helpers
class Pheromone:
    def __init__(self, scen, tmin=0.1, tmax=5.0):
        s = scen
        N = s.N
        self.s = s
        self.tt = np.ones((N, N_OPT))
        self.ta = np.ones((N, M + U))
        best = np.min(s.est, 1, keepdims=True)
        valid_t = np.arange(N_OPT)[None] < s.n_valid[:, None]
        est_sorted = np.take_along_axis(s.est, s.order, 1)
        self.et = np.where(valid_t, best / np.where(np.isfinite(est_sorted), est_sorted, 1.0), 0.0)
        valid_a = np.arange(M + U)[None] < s.n_acc[:, None]
        self.ea = np.where(valid_a, 1.0 / (1.0 + np.arange(M + U)[None]), 0.0)
        self.ea[s.n_acc == 0, 0] = 1.0
        self.tmin, self.tmax = tmin, tmax

    def ranks(self, x):
        s = self.s; N = s.N
        rt = np.minimum((x[:N] * N_OPT).astype(int), s.n_valid - 1)
        ra = np.minimum((x[N:2 * N] * np.maximum(s.n_acc, 1)).astype(int), np.maximum(s.n_acc - 1, 0))
        return rt, ra

    def sample(self, rng, n, alpha, beta):
        def draw(tau, eta):
            p = (tau ** alpha) * (eta ** beta)
            p = p / p.sum(1, keepdims=True)
            c = np.cumsum(p, 1)
            u = rng.random((n, tau.shape[0], 1))
            return np.minimum((u > c[None]).sum(-1), tau.shape[1] - 1)
        return draw(self.tt, self.et), draw(self.ta, self.ea)

    def write(self, x, rt, ra, mask=None):
        s = self.s; N = s.N
        x = x.copy()
        if mask is None:
            mask = np.ones(N, bool)
        x[:N] = np.where(mask, (rt + 0.5) / N_OPT, x[:N])
        x[N:2 * N] = np.where(mask, (ra + 0.5) / np.maximum(s.n_acc, 1), x[N:2 * N])
        return x

    def evaporate(self, rho):
        self.tt *= (1 - rho); self.ta *= (1 - rho)
        self.clip()

    def deposit(self, rt, ra, amount):
        N = len(rt)
        self.tt[np.arange(N), rt] += amount
        self.ta[np.arange(N), ra] += amount
        self.clip()

    def clip(self):
        np.clip(self.tt, self.tmin, self.tmax, out=self.tt)
        np.clip(self.ta, self.tmin, self.tmax, out=self.ta)


def aco(ev, scen, budget, rng, alpha=1.0, beta=2.0, rho=0.1, **kw):
    """Standalone MMAS-style ACO over the discrete (target, access) decisions;
    continuous genes stay at the heuristic default (equal shares, max power, hover)."""
    T = Tracker(ev, budget)
    ph = Pheromone(scen)
    base = scen.greedy_vector()
    X0 = init_pop(scen, rng)
    X0[:, 2 * scen.N:] = base[2 * scen.N:]
    k, _, _ = T.eval(X0)
    while T.left(POP):
        rt, ra = ph.sample(rng, POP, alpha, beta)
        X = np.array([ph.write(base, rt[i], ra[i]) for i in range(POP)])
        k, _, _ = T.eval(X)
        ph.evaporate(rho)
        i = int(np.argmin(k))
        ph.deposit(rt[i], ra[i], 0.5)
        brt, bra = ph.ranks(T.best_x)
        ph.deposit(brt, bra, 0.5)
    return T


# ---------------------------------------------------------------- proposed
N_STATE, N_ACT = 27, 7
BOUNDS = dict(dw=(-0.3, 0.3), dc=(-1.0, 1.0), q=(0.05, 0.5), rho=(0.02, 0.4), n=(0, 24))


def _state(impr, acc_rate, frac):
    i = 0 if impr < 1e-4 else (1 if impr < 1e-2 else 2)
    c = 0 if acc_rate == 0 else (1 if acc_rate <= 0.1 else 2)
    g = 0 if frac < 1 / 3 else (1 if frac < 2 / 3 else 2)
    return g * 9 + i * 3 + c


def _act(a, prm):
    if a == 1: prm["dw"] += 0.1
    elif a == 2: prm["dw"] -= 0.1
    elif a == 3: prm["dc"] += 0.25
    elif a == 4: prm["dc"] -= 0.25
    elif a == 5: prm["q"] *= 1.5; prm["rho"] *= 1.5; prm["n"] += 4
    elif a == 6: prm["q"] /= 1.5; prm["rho"] /= 1.5; prm["n"] -= 4
    for key, (lo, hi) in BOUNDS.items():
        prm[key] = float(np.clip(prm[key], lo, hi))


def hybrid(ev, scen, budget, rng, Q=None, use_rl=True, use_aco=True, train=False, eps=0.1,
           alpha=1.0, beta=2.0, K=5, alpha_q=0.1, gamma=0.9, zeta=(100.0, 1.0, 0.0), log=None, n0=8, q0=0.15, **kw):
    """RL-guided PSO-ACO.
    Each iteration: (i) PSO moves the whole swarm in the mixed rank-encoded space;
    (ii) ACO ants either partially re-route elite particles (fraction q of tasks) or construct a
    complete discrete assignment that inherits the continuous allocation of the global best and
    replaces the worst particle if better; (iii) every K iterations a tabular Q-learning agent
    adjusts (w, c1, c2) and the ACO intensity (number of ants n, partial ratio q, evaporation rho)."""
    T = Tracker(ev, budget)
    D = scen.dim; N = scen.N
    prm = dict(dw=0.0, dc=0.0, q=q0, rho=0.1, n=n0)
    X = init_pop(scen, rng)
    V = rng.uniform(-0.1, 0.1, X.shape)
    k, _, viol = T.eval(X)
    pb, pk = X.copy(), k.copy()
    ph = Pheromone(scen)
    it = 0
    win_best = T.best_key; win_nfe = ev.nfe; win_aco = 0; win_acc = 0
    s_prev = a_prev = None
    while True:
        n_ants = int(prm["n"]) if use_aco else 0
        if not T.left(POP + n_ants):
            break
        it += 1
        frac = ev.nfe / budget                      # TVAC base schedule + RL offsets
        w_ = float(np.clip(0.9 - 0.5 * frac + prm["dw"], 0.2, 1.0))
        c1_ = float(np.clip(2.5 - 2.0 * frac + prm["dc"], 0.3, 3.0))
        c2_ = float(np.clip(0.5 + 2.0 * frac - prm["dc"], 0.3, 3.0))
        g = pb[np.argmin(pk)]
        V = w_ * V + c1_ * rng.random(X.shape) * (pb - X) + c2_ * rng.random(X.shape) * (g - X)
        V = np.clip(V, -0.2, 0.2)
        X = np.clip(X + V, 0, 1 - 1e-9)
        k, _, viol = T.eval(X)
        if n_ants:
            rt, ra = ph.sample(rng, n_ants, alpha, beta)
            n_part = n_ants // 2
            order = np.argsort(k)
            el = order[:max(1, n_part)]
            gb = T.best_x
            cand, tgt = [], []
            for i in range(n_ants):
                if i < n_part:                      # partial refinement of an elite particle
                    e = el[i % len(el)]
                    mask = rng.random(N) < prm["q"]
                    cand.append(ph.write(X[e], rt[i], ra[i], mask)); tgt.append(e)
                else:                               # full construction + continuous genes of gbest
                    cand.append(ph.write(gb, rt[i], ra[i])); tgt.append(-1)
            cand = np.array(cand)
            kc, _, vc = T.eval(cand)
            win_aco += len(cand)
            ph.evaporate(prm["rho"])
            worst = list(order[::-1])
            for i, e in enumerate(tgt):
                if e < 0:
                    e = worst.pop(0)
                if kc[i] < k[e]:
                    gain = min((k[e] - kc[i]) / max(abs(k[e]), 1e-9), 1.0)
                    X[e], k[e], viol[e] = cand[i], kc[i], vc[i]
                    V[e] *= 0.5
                    win_acc += 1
                    r1, a1 = ph.ranks(cand[i])
                    ph.deposit(r1, a1, 0.5 + gain)
            brt, bra = ph.ranks(T.best_x)
            ph.deposit(brt, bra, 0.3)
        imp = k < pk
        pb[imp], pk[imp] = X[imp], k[imp]
        if use_rl and it % K == 0:
            impr = float(np.clip((win_best - T.best_key) / max(abs(win_best), 1e-9), 0, 1))
            infeas = float(np.mean(viol > 0))
            share = win_aco / max(ev.nfe - win_nfe, 1)
            acc_rate = win_acc / max(win_aco, 1)
            s = _state(impr, acc_rate, ev.nfe / budget)
            if s_prev is not None and train:
                r = zeta[0] * impr - zeta[1] * infeas - zeta[2] * share
                Q[s_prev, a_prev] += alpha_q * (r + gamma * Q[s].max() - Q[s_prev, a_prev])
            if train and rng.random() < eps:
                a = int(rng.integers(N_ACT))
            else:
                a = int(np.argmax(Q[s])) if Q is not None else 0
            _act(a, prm)
            if log is not None:
                log.append((ev.nfe, s, a, dict(prm)))
            s_prev, a_prev = s, a
            win_best, win_nfe, win_aco, win_acc = T.best_key, ev.nfe, 0, 0
    return T


ALGOS = {"RS": random_search, "Greedy": greedy, "GA": ga, "DE": de, "PSO": pso, "TVAC-PSO": apso,
         "ACO": aco, "PSO-ACO": lambda ev, s, b, r, **kw: hybrid(ev, s, b, r, use_rl=False, **kw),
         "Proposed": hybrid}
