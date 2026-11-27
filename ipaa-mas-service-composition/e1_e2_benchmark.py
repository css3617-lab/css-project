"""E1 (synthetic) and E2 (WS-DREAM #1 real QoS): ten algorithms, 12 sizes, 30 instances, equal budgets.
Usage: python e1_e2_benchmark.py [syn|ws1|all]"""
import sys
from ipaa.runner import run_jobs, QUICK
from ipaa.common import run_alg
from ipaa.algorithms import ALG_ORDER

BUDGET, STEP = (2000, 250) if QUICK else (20000, 250)
INST = 2 if QUICK else 30
SIZES = [(10, 20), (30, 100)] if QUICK else [(n, m) for n in (10, 20, 30, 50) for m in (20, 50, 100)]


def jobs_for(src):
    return [dict(spec=dict(src=src, n=n, m=m, seed=1000 + s), alg=a, budget=BUDGET, step=STEP,
                 run_seed=5000 + s) for (n, m) in SIZES for s in range(INST) for a in ALG_ORDER]


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("syn", "all"):
        run_jobs("E1_synthetic", jobs_for("syn"), run_alg)
    if which in ("ws1", "all"):
        run_jobs("E2_wsdream1", jobs_for("ws1"), run_alg)
