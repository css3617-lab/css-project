"""E5: one-at-a-time parameter sensitivity of IPAA (plus hand-over ablations)."""
from ipaa.runner import run_jobs, QUICK
from ipaa.common import run_alg

BUDGET, STEP = (2000, 250) if QUICK else (20000, 250)
INST = 2 if QUICK else 20
SIZES = [(30, 100)] if QUICK else [(30, 100), (50, 100)]

GRID = {
    "window": [2, 4, 8, 16, 32],
    "div_min": [0.005, 0.01, 0.02, 0.05, 0.1],
    "min_pso": [0.01, 0.02, 0.05, 0.1, 0.2],
    "elite": [1, 5, 10, 20, 40],
    "seed_gain": [0.0, 0.1, 0.2, 0.4, 0.8],
    "seed_base": [0.1, 0.3, 0.5, 1.0],
    "rho": [0.05, 0.1, 0.2, 0.3, 0.4],
    "beta": [0.0, 1.0, 2.0, 2.5, 3.0, 5.0],
    "pop": [20, 40, 80],
    "xover": [0.0, 0.25, 0.5, 0.75, 1.0],
}
ABLATIONS = {
    "fixed switch 50%": dict(adaptive=False),
    "no elite seeding": dict(seeding=False),
    "no seeding + fixed switch": dict(adaptive=False, seeding=False),
    "no PSO phase (MMAS + recombination)": dict(skip_pso=True),
    "no elite recombination": dict(xover=0.0),
    "previous IPAA design (Cho et al., 2025)": dict(adaptive=False, seeding=False, xover=0.0, rho=0.1, beta=2.0),
}

if __name__ == "__main__":
    configs = [("default", {})]
    for k, vals in GRID.items():
        for v in vals:
            configs.append((f"{k}={v}", {k: v}))
    for name, p in ABLATIONS.items():
        configs.append((name, p))
    if QUICK:
        configs = configs[:4] + configs[-3:]
    jobs = [dict(spec=dict(src=src, n=n, m=m, seed=1000 + s), alg="IPAA", budget=BUDGET, step=STEP,
                 run_seed=5000 + s, params=p, label=name)
            for src in ("syn", "ws1") for (n, m) in SIZES for s in range(INST) for name, p in configs]
    run_jobs("E5_sensitivity", jobs, run_alg)
