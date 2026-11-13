import sys, numpy as np
sys.path.insert(0, '.')
from runner import run
from multiprocessing import Pool
Q = np.load("../results/Q.npy")
algos = sys.argv[1].split(",")
Vs = [int(v) for v in sys.argv[2].split(",")]
seeds = range(500, 500 + int(sys.argv[3]))
def job(a):
    alg, V, sd = a
    r = run(sd, V, alg, n_epochs=6, Q=Q)
    return alg, V, sd, np.mean([x["F"] for x in r]), np.mean([x["Q"] for x in r]), np.mean([x["L_mean"] for x in r]), np.mean([x["E"] for x in r]), np.mean([x["feasible"] for x in r])
jobs = [(a, V, s) for a in algos for V in Vs for s in seeds]
with Pool(2) as p:
    res = p.map(job, jobs)
for V in Vs:
    for a in algos:
        rr = np.array([r[3:] for r in res if r[0] == a and r[1] == V])
        print(f"V={V} {a:10s} F={rr[:,0].mean():.4f}±{rr[:,0].std():.4f} Q={rr[:,1].mean():.3f} L={rr[:,2].mean()*1e3:.1f} E={rr[:,3].mean():.1f} feas={rr[:,4].mean():.2f}")
    # paired wins of Proposed
    if "Proposed" in algos:
        fp = {r[2]: r[3] for r in res if r[0] == "Proposed" and r[1] == V}
        for a in algos:
            if a == "Proposed": continue
            fa = {r[2]: r[3] for r in res if r[0] == a and r[1] == V}
            print("   Proposed better than", a, sum(fp[s] < fa[s] for s in fp), "/", len(fp))
