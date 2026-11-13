import sys, os, pickle, time, numpy as np
sys.path.insert(0, '.')
from runner import run
from algos import hybrid
from multiprocessing import Pool
Q = np.load("../results/Q.npy")
SEEDS = list(range(1, 21))
EP = 8
ALG = ["Greedy", "RS", "GA", "DE", "PSO", "TVAC-PSO", "ACO", "PSO-ACO", "Proposed"]
def noaco(ev, s, b, r, **kw):
    return hybrid(ev, s, b, r, use_aco=False, **kw)
def job(j):
    exp, tag, alg, V, sd, cfg = j
    t = time.time()
    a = noaco if alg == "noaco" else alg
    recs = run(sd, V, a, n_epochs=EP, Q=Q, **cfg)
    for x in recs:
        x.update(exp=exp, tag=tag, alg=alg)
    return recs
jobs = []
for V in [20, 40, 60, 80, 100]:
    for alg in ALG:
        for sd in SEEDS:
            jobs.append(("main", alg, alg, V, sd, {}))
for sd in SEEDS:
    jobs += [("abl", "w/o ACO", "noaco", 80, sd, {}),
             ("abl", "w/o age-aware", "Proposed", 80, sd, dict(age_aware=False)),
             ("abl", "w/o lifetime", "Proposed", 80, sd, dict(contact_aware=False)),
             ("abl", "w/o UAV move", "Proposed", 80, sd, dict(allow_move=False)),
             ("abl", "w/o rank order", "Proposed", 80, sd, dict(rank_order=False))]
for wname in ["latency", "energy", "equal"]:
    for alg in ["Greedy", "DE", "TVAC-PSO", "ACO", "PSO-ACO", "Proposed"]:
        for sd in SEEDS[:10]:
            jobs.append(("weights", wname, alg, 60, sd, dict(weights=wname)))
for tau in [0.0, 1.0, 4.0]:
    for tag, alg, aw in [("Proposed", "Proposed", True), ("Proposed (age-unaware)", "Proposed", False), ("TVAC-PSO", "TVAC-PSO", True)]:
        for sd in SEEDS:
            jobs.append(("age", f"{tag}|{tau}", alg, 80, sd, dict(tau_max=tau, age_aware=aw)))
for sd in SEEDS:   # tau=2 naive TVAC for completeness
    jobs.append(("age", "TVAC-PSO (age-unaware)|2.0", "TVAC-PSO", 80, sd, dict(age_aware=False)))
order = sorted(range(len(jobs)), key=lambda i: -jobs[i][3])   # big first
jobs = [jobs[i] for i in order]
out = []
t0 = time.time()
with Pool(2) as p:
    for i, recs in enumerate(p.imap_unordered(job, jobs)):
        out += recs
        if i % 20 == 0 or i == len(jobs) - 1:
            pickle.dump(out, open("../results/all.pkl", "wb"))
            print(f"{i+1}/{len(jobs)} {time.time()-t0:.0f}s", flush=True)
pickle.dump(out, open("../results/all.pkl", "wb"))
print("DONE")
