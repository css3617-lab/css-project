"""Instance factory and single-run helper shared by all experiment scripts."""
import os, time
import numpy as np
from . import model
from .algorithms import ALGOS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WS1_DIR = os.path.join(ROOT, "data", "wsdream1")
_ws1 = None


def ws1():
    global _ws1
    if _ws1 is None:
        cache = os.path.join(WS1_DIR, "ws1_cache.npz")
        if os.path.exists(cache):
            d = np.load(cache)
            obj = model.WSDream1.__new__(model.WSDream1)
            obj.rt, obj.tp, obj.avail, obj.rel = d["rt"], d["tp"], d["avail"], d["rel"]
        else:
            obj = model.WSDream1(WS1_DIR)
            try:
                np.savez(cache, rt=obj.rt, tp=obj.tp, avail=obj.avail, rel=obj.rel)
            except OSError:
                pass
        _ws1 = obj
    return _ws1


def make_problem(spec):
    """spec: dict(src='syn'|'ws1', n, m, seed, wf=None|'rand')"""
    n, m, seed = spec["n"], spec["m"], spec["seed"]
    wf = None
    if spec.get("wf") == "rand":
        wf = model.random_workflow(n, np.random.default_rng(10_000 + seed))
    if spec["src"] == "syn":
        return model.synthetic(n, m, seed, workflow=wf)
    if spec["src"] == "ws1":
        return ws1().instance(n, m, seed, workflow=wf)
    raise ValueError(spec["src"])


def run_alg(job):
    """job: dict(spec, alg, budget, step, run_seed, params)"""
    prob = make_problem(job["spec"])
    rng = np.random.default_rng(job["run_seed"])
    info = {}
    params = dict(job.get("params", {}))
    t0 = time.perf_counter()
    if job["alg"] == "IPAA":
        T = ALGOS["IPAA"](prob, job["budget"], rng, job["step"], info=info, **params)
    else:
        T = ALGOS[job["alg"]](prob, job["budget"], rng, job["step"], **params)
    dt = time.perf_counter() - t0
    u, feas = prob.describe(T.bx)
    return dict(fit=T.best, util=u, feasible=feas, time=dt, curve=[round(c, 6) for c in T.curve],
                switch=info.get("switch"))
