import time
import numpy as np
from sim import World, Evaluator, M, U
from algos import ALGOS, hybrid, N_STATE, N_ACT

CHECK = np.arange(100, 3001, 100)


def run(seed, V, algo, n_epochs=8, budget=3000, weights="balanced", tau_max=2.0, age_aware=True,
        contact_aware=True, allow_move=True, Q=None, train=False, eps=0.1, algo_kw=None, rank_order=True):
    w = World(V, seed)
    out = []
    f = ALGOS[algo] if isinstance(algo, str) else algo
    for ep in range(n_epochs):
        s = w.snapshot(tau_max=tau_max, age_aware=age_aware, rank_order=rank_order)
        ev = Evaluator(s, weights, contact_aware=contact_aware, allow_move=allow_move)
        rng = np.random.default_rng([seed, 7, ep])
        t0 = time.process_time()
        T = f(ev, s, budget, rng, Q=Q, train=train, eps=eps, **(algo_kw or {}))
        cpu = time.process_time() - t0
        d = Evaluator(s, weights, true=True, allow_move=allow_move)(T.best_x[None], detail=True)
        tr = np.array(T.trace)
        conv = np.interp(CHECK, tr[:, 0], tr[:, 1]) if len(tr) > 1 else np.full(len(CHECK), tr[0, 1])
        cls = d["cls"]
        rec = dict(seed=seed, V=V, ep=ep, N=s.N, cpu=cpu, nfe=ev.nfe, F=d["F"], viol=d["viol"],
                   feasible=float(d["viol"] == 0), L_mean=d["L"].mean(), L_med=np.median(d["L"]),
                   L_p95=np.percentile(d["L"], 95), Q=d["Q"].mean(), R=d["R"].mean(),
                   E=d["E_total"], E_move=d["E_move"], E_tx=d["E_tx"], E_cmp=d["E_cmp"], E_bh=d["E_bh"],
                   cf=d["contact_fail"].mean(), sv=d["safety_viol"].sum() / max((cls == 0).sum(), 1),
                   ubv=d["uav_budget_viol"], conv=conv, est_key=T.best_key,
                   T_up=d["T_up"].mean(), T_bh=d["T_bh"].mean(), T_q=d["T_q"].mean(), T_cmp=d["T_cmp"].mean(), T_dn=d["T_dn"].mean())
        for c in range(3):
            rec[f"Q{c}"] = d["Q"][cls == c].mean() if (cls == c).any() else np.nan
            rec[f"L{c}"] = d["L"][cls == c].mean() if (cls == c).any() else np.nan
        kinds = np.bincount(d["kind"], minlength=4) / s.N
        rec.update(local=kinds[0], mec=kinds[1], uavc=kinds[2], cloud=kinds[3])
        out.append(rec)
        w.apply(d)
        w.step_vehicles()
    return out
