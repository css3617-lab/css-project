import sys, time, numpy as np
sys.path.insert(0, '.')
from runner import run
from algos import N_STATE, N_ACT
Q = np.zeros((N_STATE, N_ACT))
rng = np.random.default_rng(2026)
hist = []
t0 = time.time()
n_runs = int(sys.argv[1]) if len(sys.argv) > 1 else 36
for i in range(n_runs):
    V = int(rng.choice([20, 40, 60, 80, 100]))
    eps = max(0.05, 0.3 * (1 - i / n_runs))
    r = run(1000 + i, V, "Proposed", n_epochs=6, Q=Q, train=True, eps=eps)
    hist.append((i, V, np.mean([x["F"] for x in r])))
    print(i, V, round(eps, 3), round(hist[-1][2], 4), round(time.time() - t0), flush=True)
    np.save("../results/Q.npy", Q)
print(np.round(Q, 2))
