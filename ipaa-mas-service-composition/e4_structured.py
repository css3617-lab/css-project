"""E4: structured workflows (sequence, parallel, exclusive choice, loop) on synthetic and real QoS."""
from ipaa.runner import run_jobs, QUICK
from ipaa.common import run_alg
from ipaa.algorithms import ALG_ORDER

BUDGET, STEP = (2000, 250) if QUICK else (20000, 250)
INST = 2 if QUICK else 30
SIZES = [(20, 50)] if QUICK else [(n, m) for n in (10, 20, 30, 50) for m in (50, 100)]

if __name__ == "__main__":
    jobs = [dict(spec=dict(src=src, n=n, m=m, seed=4000 + s, wf="rand"), alg=a, budget=BUDGET, step=STEP,
                 run_seed=8000 + s)
            for src in ("syn", "ws1") for (n, m) in SIZES for s in range(INST) for a in ALG_ORDER]
    run_jobs("E4_structured", jobs, run_alg)
