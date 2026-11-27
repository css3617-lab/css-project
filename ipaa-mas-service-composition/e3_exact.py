"""E3: true optimality gap on small instances (exhaustive enumeration of all m**n plans)."""
import numpy as np
from ipaa.runner import run_jobs, QUICK
from ipaa.common import run_alg, make_problem
from ipaa.exact import enumerate_optimum
from ipaa.algorithms import ALG_ORDER

SIZES = [(5, 8)] if QUICK else [(5, 12), (6, 10), (6, 15), (7, 10), (8, 8)]
INST = 2 if QUICK else 30
SRCS = ["syn", "ws1"]
BUDGET, STEP = (1000, 250) if QUICK else (5000, 250)


def opt_job(job):
    p = make_problem(job["spec"])
    f, x = enumerate_optimum(p)
    u, feas = p.describe(x)
    return dict(opt=f, util=u, feasible=feas, x=x.tolist())


if __name__ == "__main__":
    specs = [dict(src=src, n=n, m=m, seed=3000 + s) for src in SRCS for (n, m) in SIZES for s in range(INST)]
    run_jobs("E3_optimum", [dict(spec=sp) for sp in specs], opt_job)
    jobs = [dict(spec=sp, alg=a, budget=BUDGET, step=STEP, run_seed=7000 + sp["seed"]) for sp in specs for a in ALG_ORDER]
    run_jobs("E3_algorithms", jobs, run_alg)
