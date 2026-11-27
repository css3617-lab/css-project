"""E7: runtime adaptation.
  7a  synthetic drift + failures + load bursts (300 steps), five strategies
  7b  failure-rate sweep
  7c  trace-driven: real WS-DREAM #2 response time / throughput over 64 time slices (if the data is present)
"""
import time
import numpy as np
from ipaa import normal as norm
from ipaa import model
from ipaa.algorithms import ipaa
from ipaa.runner import run_jobs, QUICK

STRATS = ["Static", "Centralized re-opt.", "MAS local repair only", "MAS-IPAA (cold start)", "MAS-IPAA (warm start)"]
B_FULL, B_WARM = (3000, 600) if QUICK else (10000, 2000)


# ------------------------------------------------------------------ environments
def env_synthetic(seed, n=20, m=50, H=300, pf=0.002, burst=True):
    rng = np.random.default_rng(seed)
    p = model.synthetic(n, m, 3000 + seed)
    rt0, c0 = p.attrs[0].v.copy(), p.attrs[1].v.copy()
    lrt, lc = np.zeros((n, m)), np.zeros((n, m))
    down = np.zeros((n, m), int)
    mask = None
    traj = []
    for t in range(H):
        lrt = 0.95 * lrt + rng.normal(0, 0.04, (n, m))
        lc = 0.98 * lc + rng.normal(0, 0.01, (n, m))
        down = np.maximum(down - 1, 0)
        fail = (rng.random((n, m)) < pf) & (down == 0)
        down[fail] = rng.geometric(1 / 30, fail.sum())
        if burst and t in (100, 200):
            mask = rng.random((n, m)) < 0.2
        if t in (150, 250):
            mask = None
        b = np.where(mask, 2.0, 1.0) if mask is not None else 1.0
        traj.append({0: np.clip(rt0 * np.exp(lrt) * b, 50, 5000), 1: np.clip(c0 * np.exp(lc), 0.05, 8), "alive": down == 0})
    return p, traj


def env_ws2(seed, n=20, m=50):
    from ipaa import ws2
    D = ws2.load()
    rng = np.random.default_rng(seed)
    rt, tp = D["rt"], D["tp"]
    while True:
        u = int(rng.integers(0, rt.shape[0]))
        cover = np.isfinite(rt[u]).mean(1)
        good = np.where((cover >= 0.8) & (D["avail"] > 0.3))[0]
        if len(good) >= n * m:
            break
    S = rng.choice(good, n * m, replace=False).reshape(n, m)
    R, TP = rt[u][S], tp[u][S]            # (n, m, 64)
    med_rt = np.nanmedian(R, 2); med_tp = np.nanmedian(TP, 2)
    av = np.clip(D["avail"][S], 0.3, 0.9999); rel = np.clip(D["rel"][S], 0.2, 0.9999)
    q = (norm.ppf(np.clip(av, 1e-4, 1 - 1e-4)) + np.log(np.maximum(med_tp, 1e-3)) / 3) / 2
    q = (q - q.mean()) / (q.std() + 1e-12)
    cost = np.clip(1.5 + 0.5 * q + 0.5 * rng.standard_normal((n, m)), 0.1, 5.0)
    attrs = [model.Attr("rt", med_rt, "time", -1), model.Attr("tp", np.maximum(med_tp, 1e-3), "bottleneck", +1),
             model.Attr("avail", av, "prob", +1), model.Attr("rel", rel, "prob", +1),
             model.Attr("cost", cost, "additive", -1)]
    p = model.Problem(attrs, model.W5)
    traj = []
    for t in range(R.shape[2]):
        r, h = R[:, :, t], TP[:, :, t]
        alive = np.isfinite(r) & np.isfinite(h) & (r < 19.9)
        traj.append({0: np.where(np.isfinite(r), r, med_rt), 1: np.where(np.isfinite(h), np.maximum(h, 1e-3), med_tp),
                     "alive": alive})
    return p, traj


# ------------------------------------------------------------------ simulation
def opt(p, budget, rng, **kw):
    saved = p.evals; p.evals = 0
    T = ipaa(p, budget, rng, 1000, **kw)
    p.evals = saved + p.evals
    return T


def apply(p, st):
    for k, v in st.items():
        if k == "alive":
            p.alive = v
        else:
            p.attrs[k].v = v


def ok(p, x):
    u, v, d = p.utility(x)
    return bool(v[0] == 0 and not d[0]), float(u[0])


def simulate(job):
    rng = np.random.default_rng(9000 + job["seed"])
    if job["env"] == "syn":
        p, traj = env_synthetic(job["seed"], pf=job.get("pf", 0.002), H=60 if QUICK else 300)
    else:
        p, traj = env_ws2(job["seed"])
    apply(p, traj[0])
    info = {}
    T = opt(p, B_FULL, rng, info=info)
    x, tau = T.bx.copy(), info["tau"]
    p.evals = 0
    cpu, n_adapt, rec = 0.0, 0, []
    okc, util = 0, []
    ref_u = ok(p, x)[1]
    s = job["strategy"]
    period = 10 if job["env"] == "syn" else 1
    for t, st in enumerate(traj):
        apply(p, st)
        feas, u = ok(p, x)
        degrade = feas and (u < ref_u - 0.05) and (t % period == 0)
        if s != "Static" and (not feas or degrade):
            n_adapt += 1
            e0, c0 = p.evals, time.perf_counter()
            if s == "Centralized re-opt.":
                x = opt(p, B_FULL, rng).bx.copy()
            else:
                for i in np.where(~p.alive[np.arange(p.n), x])[0]:
                    Y = np.repeat(x[None], p.m, 0); Y[:, i] = np.arange(p.m)
                    x = Y[int(np.argmax(p.fitness(Y)))].copy()
                feas, u = ok(p, x)
                if (not feas or degrade) and s != "MAS local repair only":
                    if s == "MAS-IPAA (warm start)":
                        inf2 = {}
                        T2 = opt(p, B_WARM, rng, tau_init=0.5 * tau + 0.5 * tau.mean(), info=inf2)
                        tau = inf2["tau"]
                    else:
                        T2 = opt(p, B_WARM, rng)
                    if T2.best > p.fitness(x[None])[0]:
                        x = T2.bx.copy()
            rec.append(p.evals - e0); cpu += time.perf_counter() - c0
            feas, u = ok(p, x)
            ref_u = u
        okc += feas
        util.append(u if feas else 0.0)
    return dict(sla=okc / len(traj), util=float(np.mean(util)), adapt=n_adapt, evals=int(np.sum(rec)) if rec else 0,
                cpu=cpu, trace=[round(v, 4) for v in util])


if __name__ == "__main__":
    R = 2 if QUICK else 30
    run_jobs("E7a_dynamic_synthetic", [dict(env="syn", seed=s, strategy=k) for s in range(R) for k in STRATS], simulate)
    R2 = 2 if QUICK else 20
    pfs = [0.002, 0.01] if QUICK else [0.001, 0.002, 0.005, 0.01]
    run_jobs("E7b_failure_sweep", [dict(env="syn", seed=100 + s, strategy=k, pf=pf) for pf in pfs for s in range(R2)
                                   for k in STRATS[1:]], simulate)
    from ipaa import ws2
    if ws2.available():
        ws2.load()   # build the cache once in the main process
        run_jobs("E7c_dynamic_wsdream2", [dict(env="ws2", seed=s, strategy=k) for s in range(R) for k in STRATS], simulate)
    else:
        print("[E7c] WS-DREAM dataset #2 not found in data/wsdream2 -> trace-driven experiment skipped. "
              "Run download_data.py or place rtdata.txt and tpdata.txt there.", flush=True)
