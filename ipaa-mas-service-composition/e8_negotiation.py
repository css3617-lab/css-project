"""E8: concurrent requests competing for capacity-limited services.
Policies: Selfish, FCFS, Negotiation (least marginal loss concedes), Negotiation-SLA (most SLA slack concedes),
and Joint (one centralized IPAA over all requests, overload modelled in the fitness) as a reference."""
import numpy as np
from ipaa import model
from ipaa.algorithms import ipaa
from ipaa.runner import run_jobs, QUICK

N, M, CAP = 10, 30, 2
BUD = 1500 if QUICK else 5000
POLICIES = ["Selfish", "FCFS", "Negotiation", "Negotiation-SLA", "Joint"]


def problems(seed, K):
    rng = np.random.default_rng(seed)
    W = rng.dirichlet(np.ones(4) * 2, K)
    return [model.synthetic(N, M, 4000 + seed, weights=w) for w in W]


def opt(p, rng, budget=BUD):
    p.evals = 0
    return ipaa(p, budget, rng, 1000).bx.copy()


def loads(X):
    L = np.zeros((N, M), int)
    for x in X:
        L[np.arange(N), x] += 1
    return L


def effective(p, x, L):
    idx = np.arange(N)
    f = np.maximum(1.0, L[idx, x] / CAP)
    v0 = p.attrs[0].v
    p.attrs[0].v = v0.copy(); p.attrs[0].v[idx, x] *= f
    u, v, d = p.utility(x)
    p.attrs[0].v = v0
    return float(u[0]), bool(v[0] == 0 and not d[0])


def slack(p, x):
    Q, _ = p.aggregate(x)
    rng_ = np.maximum(p.hi - p.lo, 1e-12)
    return float(min((ub - Q[0, k]) / rng_[k] for k, ub in p.cons))


class Joint:
    """All K requests as one problem of K*N tasks; RT of overloaded services scales with load/capacity."""

    def __init__(self, P):
        self.P, self.K = P, len(P)
        self.n, self.m = self.K * N, M
        self.alive = np.ones((self.n, self.m), bool)
        self.evals = 0

    def local_score(self):
        return np.concatenate([p.local_score() for p in self.P], 0)

    def fitness(self, X):
        X = np.atleast_2d(X)
        Pn = X.shape[0]
        Xr = X.reshape(Pn, self.K, N)
        L = np.zeros((Pn, N, M))
        for k in range(self.K):
            np.add.at(L, (np.arange(Pn)[:, None], np.arange(N)[None], Xr[:, k]), 1)
        F = np.zeros(Pn)
        for k, p in enumerate(self.P):
            xs = Xr[:, k]
            fac = np.maximum(1.0, np.take_along_axis(L, xs[:, :, None], 2)[:, :, 0] / CAP)
            rt = p.attrs[0].v[np.arange(N), xs] * fac
            Qs = [rt.sum(1)] + [model._agg_seq(a.v[np.arange(N), xs], a.kind) for a in p.attrs[1:]]
            Q = np.stack(Qs, 1)
            rng_ = np.maximum(p.hi - p.lo, 1e-12)
            Nn = (Q - p.lo) / rng_
            sg = np.array([a.sign for a in p.attrs])
            u = np.where(sg > 0, Nn, 1 - Nn) @ p.w
            viol = sum(np.maximum(0, Q[:, kk] - ub) / rng_[kk] for kk, ub in p.cons)
            F += u - 2.0 * viol
        self.evals += Pn
        return F / self.K


def negotiate(P, X, K, sla_first, seed):
    msgs = 0
    X = [x.copy() for x in X]
    for _ in range(500):
        L = loads(X)
        over = np.argwhere(L > CAP)
        if len(over) == 0:
            break
        i, s = over[0]
        users = [k for k in range(K) if X[k][i] == s]
        free = np.where(L[i] < CAP)[0]
        offers = []
        for k in users:
            Y = np.repeat(X[k][None], len(free), 0); Y[:, i] = free
            f = P[k].fitness(Y)
            j = int(np.argmax(f))
            _, v, _ = P[k].utility(Y[j][None])
            loss = P[k].fitness(X[k][None])[0] - f[j]
            key = (bool(v[0] > 0), -slack(P[k], Y[j][None])) if sla_first else (bool(v[0] > 0), loss)
            offers.append((key, k, free[j]))
            msgs += 1
        offers.sort(key=lambda o: o[0])
        for _, k, alt in offers[: L[i, s] - CAP]:
            X[k] = X[k].copy(); X[k][i] = alt
            msgs += 1
    for rnd in range(2 if sla_first else 1):
        L = loads(X)
        viol = [k for k in range(K) if not effective(P[k], X[k], L)[1]]
        for k in viol:
            own = np.zeros((N, M), int); own[np.arange(N), X[k]] = 1
            P[k].alive = (L - own) < CAP
            y = opt(P[k], np.random.default_rng(seed * 100 + 80 + k + 1000 * rnd))
            P[k].alive = np.ones((N, M), bool)
            if P[k].fitness(y[None])[0] >= P[k].fitness(X[k][None])[0] or not effective(P[k], X[k], L)[1]:
                L = L - own; L[np.arange(N), y] += 1
                X[k] = y
            msgs += 2
    return X, msgs


def episode(job):
    seed, K, pol = job["seed"], job["K"], job["policy"]
    P = problems(seed, K)
    rng = np.random.default_rng(seed)
    msgs = 0
    if pol in ("Selfish", "Negotiation", "Negotiation-SLA"):
        X = [opt(p, np.random.default_rng(seed * 100 + k)) for k, p in enumerate(P)]
        if pol != "Selfish":
            X, msgs = negotiate(P, X, K, pol == "Negotiation-SLA", seed)
    elif pol == "FCFS":
        X = [None] * K
        L = np.zeros((N, M), int)
        for k in rng.permutation(K):
            P[k].alive = L < CAP
            X[k] = opt(P[k], np.random.default_rng(seed * 100 + 50 + k))
            P[k].alive = np.ones((N, M), bool)
            L[np.arange(N), X[k]] += 1
    else:
        J = Joint(P)
        # generous centralized reference: K x BUD joint evaluations (each joint evaluation scores all K plans)
        x = ipaa(J, BUD * K, np.random.default_rng(seed * 100 + 99), 1000).bx
        X = [x[k * N:(k + 1) * N].copy() for k in range(K)]
    L = loads(X)
    ev = [effective(P[k], X[k], L) for k in range(K)]
    u = np.array([e[0] for e in ev])
    return dict(mean=float(u.mean()), min=float(u.min()), jain=float(u.sum() ** 2 / (K * (u ** 2).sum())),
                sla=float(np.mean([e[1] for e in ev])), msgs=msgs, overload=int((L > CAP).sum()))


if __name__ == "__main__":
    R = 2 if QUICK else 30
    Ks = [10] if QUICK else [10, 20, 30]
    run_jobs("E8_negotiation", [dict(seed=s, K=K, policy=p) for K in Ks for s in range(R) for p in POLICIES], episode)
