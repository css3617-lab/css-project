"""E6: large-scale instances (up to 100 tasks x 1000 candidates) and running time."""
from ipaa.runner import run_jobs, QUICK
from ipaa.common import run_alg
from ipaa.algorithms import ALG_ORDER

BUDGET, STEP = (3000, 1000) if QUICK else (50000, 1000)
INST = 1 if QUICK else 10
SIZES = [(50, 200)] if QUICK else [(n, m) for n in (50, 100) for m in (200, 500, 1000)]

if __name__ == "__main__":
    jobs = [dict(spec=dict(src="syn", n=n, m=m, seed=6000 + s), alg=a, budget=BUDGET, step=STEP, run_seed=9000 + s)
            for (n, m) in SIZES for s in range(INST) for a in ALG_ORDER]
    run_jobs("E6_largescale", jobs, run_alg)
