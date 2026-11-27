"""Quick text summary of whatever results exist (the full statistics are done when writing the paper)."""
import json, os, collections
import numpy as np

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def load(name):
    p = os.path.join(RES, name + ".jsonl")
    if not os.path.exists(p):
        return []
    return [json.loads(l) for l in open(p, encoding="utf8") if l.strip()]


def bench(name):
    R = load(name)
    if not R:
        return
    print(f"\n== {name}: mean best fitness (final)")
    D = collections.defaultdict(list)
    for r in R:
        j = r["_job"]; s = j["spec"]
        D[(s["n"], s["m"], j["alg"])].append(r["fit"])
    sizes = sorted({(k[0], k[1]) for k in D})
    algs = [a for a in ["RS", "GA", "DE", "SA", "PSO", "GWO", "WOA", "ACO", "PSO-ACO", "IPAA"] if any(k[2] == a for k in D)]
    print("size     " + " ".join(f"{a:>8}" for a in algs))
    for sz in sizes:
        print(f"{sz[0]:>3}x{sz[1]:<4} " + " ".join(f"{np.mean(D[(sz[0], sz[1], a)]):8.4f}" for a in algs))


def generic(name, keys, group):
    R = load(name)
    if not R:
        return
    print(f"\n== {name}")
    D = collections.defaultdict(list)
    for r in R:
        D[tuple(r["_job"].get(g) for g in group)].append(r)
    for k in sorted(D, key=str):
        print(k, " ".join(f"{f}={np.mean([r[f] for r in D[k]]):.4f}" for f in keys))


if __name__ == "__main__":
    for n in ("E1_synthetic", "E2_wsdream1", "E3_algorithms", "E4_structured", "E6_largescale"):
        bench(n)
    generic("E5_sensitivity", ["fit"], ["label"])
    generic("E7a_dynamic_synthetic", ["sla", "util", "evals", "cpu"], ["strategy"])
    generic("E7b_failure_sweep", ["sla", "util", "evals"], ["pf", "strategy"])
    generic("E7c_dynamic_wsdream2", ["sla", "util", "evals", "cpu"], ["strategy"])
    generic("E8_negotiation", ["mean", "min", "jain", "sla", "msgs"], ["K", "policy"])
