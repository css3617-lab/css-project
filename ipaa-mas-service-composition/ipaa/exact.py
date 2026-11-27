"""Exhaustive enumeration of all m**n compositions (small instances only) to obtain the true optimum."""
import numpy as np


def enumerate_optimum(prob, chunk=500_000):
    n, m = prob.n, prob.m
    total = m ** n
    best, bx = -np.inf, None
    saved = prob.evals
    for start in range(0, total, chunk):
        idx = np.arange(start, min(start + chunk, total), dtype=np.int64)
        X = np.empty((len(idx), n), dtype=np.int64)
        r = idx.copy()
        for i in range(n - 1, -1, -1):
            X[:, i] = r % m
            r //= m
        f = prob.fitness(X)
        j = int(np.argmax(f))
        if f[j] > best:
            best, bx = float(f[j]), X[j].copy()
    prob.evals = saved
    return best, bx
