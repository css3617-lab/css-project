"""Statistics, LaTeX tables and figures for the manuscript.

Usage: python analysis/make_tables_figures.py <output_dir> [results_dir]
Reads the JSON-lines result files written by the experiment scripts (default: results/)."""
import json, collections, os, sys
import numpy as np
from scipy.stats import wilcoxon, rankdata, friedmanchisquare, norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = sys.argv[1]
RESDIR = sys.argv[2] if len(sys.argv) > 2 else "results"
os.makedirs(OUT, exist_ok=True)
ALG = ["RS", "GA", "DE", "SA", "PSO", "GWO", "WOA", "ACO", "PSO-ACO", "IPAA"]
LAB = {"ACO": "MMAS"}
lab = lambda a: LAB.get(a, a)
STEP = 250
NUM = {}


def load(n):
    return [json.loads(l) for l in open(os.path.join(RESDIR, n + ".jsonl")) if l.strip()]


def holm(ps):
    ps = np.asarray(ps); o = np.argsort(ps); m = len(ps)
    adj = np.empty(m); run = 0
    for k, i in enumerate(o):
        run = max(run, (m - k) * ps[i]); adj[i] = min(1, run)
    return adj


def marks(M, ctrl=-1):
    """Wilcoxon signed-rank (Holm-corrected over the other algorithms) of the control column vs each other."""
    c = M[:, ctrl]; ps, dirs = [], []
    for i in range(M.shape[1] - 1):
        if np.allclose(c, M[:, i]):
            ps.append(1.0)
        else:
            ps.append(wilcoxon(c, M[:, i]).pvalue)
        dirs.append(np.sign(c.mean() - M[:, i].mean()))
    adj = holm(ps)
    return ["$\\approx$" if p >= 0.05 else ("$+$" if d > 0 else "$-$") for p, d in zip(adj, dirs)], adj


def friedman_holm(Mall):
    """Friedman average ranks over all instances and Holm-adjusted post-hoc p vs IPAA (control)."""
    N, k = Mall.shape
    R = np.array([rankdata(-row) for row in Mall]).mean(0)
    se = np.sqrt(k * (k + 1) / (6 * N))
    z = (R[:-1] - R[-1]) / se
    p = 2 * (1 - norm.cdf(np.abs(z)))
    return R, holm(p), friedmanchisquare(*Mall.T).pvalue


def bench_matrix(R, budget_idx, key=("n", "m")):
    D = collections.defaultdict(lambda: collections.defaultdict(dict))
    for r in R:
        j = r["_job"]; s = j["spec"]
        D[tuple(s[k] for k in key)][j["alg"]][s["seed"]] = r
    out = {}
    for sz in sorted(D):
        seeds = sorted(D[sz]["IPAA"])
        out[sz] = np.array([[D[sz][a][s]["curve"][budget_idx] for a in ALG] for s in seeds])
    return out, D


def bench_table(name, title, label, R, budgets=(5000, 20000), wf=False):
    lines = []
    lines.append("\\begin{table*}[!t]\\centering\\scriptsize")
    lines.append(f"\\caption{{{title}}}\\label{{{label}}}")
    lines.append("\\setlength{\\tabcolsep}{2.6pt}")
    lines.append("\\begin{tabular}{@{}l" + "r" * len(ALG) + "@{}}\\toprule")
    lines.append("$n\\times m$ & " + " & ".join(lab(a) for a in ALG) + " \\\\\\midrule")
    summ = {}
    for b in budgets:
        Ms, D = bench_matrix(R, b // STEP - 1)
        lines.append(f"\\multicolumn{{{len(ALG)+1}}}{{@{{}}l}}{{\\textit{{Budget {b:,} evaluations}}}} \\\\".replace(",", "{,}"))
        wins = collections.Counter()
        for sz, M in Ms.items():
            mu = M.mean(0); best = mu.max()
            mk, _ = marks(M)
            cells = []
            for i, a in enumerate(ALG):
                s = f"{mu[i]:.4f}"
                if abs(mu[i] - best) < 5e-5:
                    s = "\\textbf{" + s + "}"
                if i < len(ALG) - 1:
                    s += mk[i]; wins[(a, mk[i])] += 1
                cells.append(s)
            lines.append(f"{sz[0]}$\\times${sz[1]} & " + " & ".join(cells) + " \\\\")
        Mall = np.concatenate(list(Ms.values()))
        Rk, ph, pf = friedman_holm(Mall)
        lines.append("\\cmidrule(l){2-" + str(len(ALG) + 1) + "}")
        lines.append("$+/\\approx/-$ & " + " & ".join(f"{wins[(a,'$+$')]}/{wins[(a,'$\\approx$')]}/{wins[(a,'$-$')]}" for a in ALG[:-1]) + " & -- \\\\")
        lines.append("Avg.\\ rank & " + " & ".join(f"{x:.2f}" for x in Rk) + " \\\\")
        lines.append("Holm $p$ & " + " & ".join(("$<$0.001" if p < 0.001 else f"{p:.3f}") for p in ph) + " & -- \\\\")
        if b != budgets[-1]:
            lines.append("\\midrule")
        summ[b] = dict(rank=dict(zip(ALG, Rk.round(3).tolist())), holm=dict(zip(ALG[:-1], ph.tolist())), friedman_p=pf,
                       wins={f"{a}{t}": c for (a, t), c in wins.items()},
                       means={f"{k[0]}x{k[1]}": dict(zip(ALG, M.mean(0).round(4).tolist())) for k, M in Ms.items()})
    lines.append("\\bottomrule\\end{tabular}")
    lines.append("\\par\\smallskip\\parbox{\\textwidth}{\\scriptsize Mean best penalized fitness over 30 instances. "
                 "Marks: Wilcoxon signed-rank test of IPAA against each method with Holm correction ($\\alpha=0.05$); "
                 "$+$ IPAA significantly better, $\\approx$ no significant difference, $-$ IPAA significantly worse. "
                 "Rows ``Avg.\\ rank'' and ``Holm $p$'': Friedman ranks over all 360 instances and Holm-adjusted post-hoc $p$-values with IPAA as control.}")
    lines.append("\\end{table*}")
    open(f"{OUT}/{name}.tex", "w").write("\n".join(lines))
    return summ


# ------------------------------------------------------------------ E1 / E2 / E4
E1, E2, E4 = load("E1_synthetic"), load("E2_wsdream1"), load("E4_structured")
NUM["E1"] = bench_table("tab_e1", "Synthetic benchmark (E1): sequential workflows, correlated synthetic QoS.", "tab:e1", E1)
NUM["E2"] = bench_table("tab_e2", "Real-QoS benchmark (E2): sequential workflows on WS-DREAM dataset \\#1.", "tab:e2", E2)


def bench_table_wf():
    lines = ["\\begin{table*}[!t]\\centering\\scriptsize",
             "\\caption{Structured workflows (E4): random series-parallel workflows with exclusive choices and loops, budget 20{,}000 evaluations.}\\label{tab:e4}",
             "\\setlength{\\tabcolsep}{2.6pt}", "\\begin{tabular}{@{}l" + "r" * len(ALG) + "@{}}\\toprule",
             "$n\\times m$ & " + " & ".join(lab(a) for a in ALG) + " \\\\\\midrule"]
    summ = {}
    for src, tit in (("syn", "Synthetic QoS"), ("ws1", "WS-DREAM \\#1 QoS")):
        R = [r for r in E4 if r["_job"]["spec"]["src"] == src]
        Ms, _ = bench_matrix(R, 20000 // STEP - 1)
        lines.append(f"\\multicolumn{{{len(ALG)+1}}}{{@{{}}l}}{{\\textit{{{tit}}}}} \\\\")
        wins = collections.Counter()
        for sz, M in Ms.items():
            mu = M.mean(0); best = mu.max(); mk, _ = marks(M)
            cells = []
            for i, a in enumerate(ALG):
                s = f"{mu[i]:.4f}"
                if abs(mu[i] - best) < 5e-5:
                    s = "\\textbf{" + s + "}"
                if i < len(ALG) - 1:
                    s += mk[i]; wins[(a, mk[i])] += 1
                cells.append(s)
            lines.append(f"{sz[0]}$\\times${sz[1]} & " + " & ".join(cells) + " \\\\")
        Mall = np.concatenate(list(Ms.values()))
        Rk, ph, pf = friedman_holm(Mall)
        lines.append("\\cmidrule(l){2-" + str(len(ALG) + 1) + "}")
        lines.append("$+/\\approx/-$ & " + " & ".join(f"{wins[(a,'$+$')]}/{wins[(a,'$\\approx$')]}/{wins[(a,'$-$')]}" for a in ALG[:-1]) + " & -- \\\\")
        lines.append("Avg.\\ rank & " + " & ".join(f"{x:.2f}" for x in Rk) + " \\\\")
        lines.append("Holm $p$ & " + " & ".join(("$<$0.001" if p < 0.001 else f"{p:.3f}") for p in ph) + " & -- \\\\")
        if src == "syn":
            lines.append("\\midrule")
        summ[src] = dict(rank=dict(zip(ALG, Rk.round(3).tolist())), holm=dict(zip(ALG[:-1], ph.tolist())),
                         wins={f"{a}{t}": c for (a, t), c in wins.items()},
                         means={f"{k[0]}x{k[1]}": dict(zip(ALG, M.mean(0).round(4).tolist())) for k, M in Ms.items()})
    lines += ["\\bottomrule\\end{tabular}", "\\par\\smallskip\\parbox{\\textwidth}{\\scriptsize Notation as in Table~\\ref{tab:e1}.}", "\\end{table*}"]
    open(f"{OUT}/tab_e4.tex", "w").write("\n".join(lines))
    return summ


NUM["E4"] = bench_table_wf()

# feasibility and switch points
for nm, R in (("E1", E1), ("E2", E2)):
    fe = collections.defaultdict(list); sw = collections.defaultdict(list); tm = collections.defaultdict(list)
    for r in R:
        j = r["_job"]; fe[j["alg"]].append(r["feasible"]); tm[j["alg"]].append(r["time"])
        if j["alg"] == "IPAA":
            sw[(j["spec"]["n"], j["spec"]["m"])].append(r["switch"])
    NUM[nm + "_feas"] = {a: float(np.mean(v)) for a, v in fe.items()}
    NUM[nm + "_time"] = {a: float(np.mean(v)) for a, v in tm.items()}
    allsw = np.concatenate([np.array(v) for v in sw.values()])
    NUM[nm + "_switch"] = dict(mean=float(allsw.mean()), median=float(np.median(allsw)), max=float(allsw.max()),
                               min=float(allsw.min()), share=float(allsw.mean() / 20000))

# ------------------------------------------------------------------ convergence figure (E1, E2 largest size)
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "savefig.dpi": 300})
COL = {"RS": "#9e9e9e", "GA": "#e69f00", "DE": "#8c564b", "SA": "#7f7f7f", "PSO": "#56b4e9", "GWO": "#0072b2",
       "WOA": "#17becf", "ACO": "#009e73", "PSO-ACO": "#cc79a7", "IPAA": "#d55e00"}
LS = {"GA": "-.", "DE": ":", "SA": ":", "PSO": "--", "GWO": "-", "WOA": "--", "ACO": "-", "PSO-ACO": (0, (5, 1, 1, 1)), "IPAA": "-"}
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
for ax, (R, tt) in zip(axes, ((E1, "Synthetic, $n=50$, $m=100$"), (E2, "WS-DREAM #1, $n=50$, $m=100$"))):
    for a in ALG[1:]:
        C = np.array([r["curve"] for r in R if r["_job"]["alg"] == a and r["_job"]["spec"]["n"] == 50 and r["_job"]["spec"]["m"] == 100])
        x = np.arange(1, C.shape[1] + 1) * STEP
        ax.plot(x, C.mean(0), color=COL[a], ls=LS[a], lw=1.7 if a == "IPAA" else 1.0, label=lab(a))
    ax.set_title(tt, fontsize=8); ax.set_xlabel("Fitness evaluations"); ax.grid(alpha=0.25, lw=0.4)
    ax.set_xticks([0, 5000, 10000, 15000, 20000]); ax.set_xticklabels(["0", "5k", "10k", "15k", "20k"])
axes[0].set_ylabel("Best fitness (mean of 30)")
axes[0].set_ylim(0.25, None); axes[1].set_ylim(0.55, None)
axes[1].legend(ncol=3, fontsize=6.5, loc="lower right")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_convergence.pdf", bbox_inches="tight"); plt.close(fig)

# ------------------------------------------------------------------ E3 exact optimum
OPT = {(r["_job"]["spec"]["src"], r["_job"]["spec"]["n"], r["_job"]["spec"]["m"], r["_job"]["spec"]["seed"]): r for r in load("E3_optimum")}
A3 = load("E3_algorithms")
E3 = collections.defaultdict(lambda: collections.defaultdict(list))
for r in A3:
    s = r["_job"]["spec"]; k = (s["src"], s["n"], s["m"], s["seed"]); opt = OPT[k]["opt"]
    for b in (2000, 5000):
        f = r["curve"][b // STEP - 1]
        E3[(s["src"], s["n"], s["m"], b)][r["_job"]["alg"]].append((abs(f - opt) < 2e-6, (opt - f) / abs(opt) * 100))
A3L = ALG[1:]
lines = ["\\begin{table*}[!t]\\centering\\scriptsize",
         "\\caption{Comparison with the exact optimum obtained by exhaustive enumeration (E3): share of runs that found the optimal composition (\\%) and, in parentheses, mean gap to the optimum (\\%) after 5{,}000 evaluations.}\\label{tab:e3}",
         "\\setlength{\\tabcolsep}{2.2pt}", "\\begin{tabular}{@{}llr" + "c" * len(A3L) + "@{}}\\toprule",
         "QoS & $n\\times m$ & $m^n$ & " + " & ".join(lab(a) for a in A3L) + " \\\\\\midrule"]
e3num = {}
for src, tit in (("syn", "Synthetic"), ("ws1", "WS-DREAM")):
    first = True
    for (n, m) in [(5, 12), (6, 10), (6, 15), (7, 10), (8, 8)]:
        d = E3[(src, n, m, 5000)]
        hit = {a: 100 * np.mean([h for h, _ in d[a]]) for a in A3L}
        gap = {a: np.mean([g for _, g in d[a]]) for a in A3L}
        besth = max(hit.values())
        cells = []
        for a in A3L:
            s = f"{hit[a]:.0f} ({gap[a]:.2f})"
            if abs(hit[a] - besth) < 1e-9:
                s = "\\textbf{" + s + "}"
            cells.append(s)
        lines.append((tit if first else "") + f" & {n}$\\times${m} & {m**n:,} & ".replace(",", "{,}") + " & ".join(cells) + " \\\\")
        first = False
        e3num[f"{src}|{n}x{m}"] = dict(hit=hit, gap=gap, hit2k={a: 100 * np.mean([h for h, _ in E3[(src, n, m, 2000)][a]]) for a in A3L})
    if src == "syn":
        lines.append("\\midrule")
lines += ["\\bottomrule\\end{tabular}", "\\end{table*}"]
open(f"{OUT}/tab_e3.tex", "w").write("\n".join(lines))
pool = {a: (np.mean([e3num[k]["hit"][a] for k in e3num]), np.mean([e3num[k]["gap"][a] for k in e3num]),
            np.mean([e3num[k]["hit2k"][a] for k in e3num])) for a in A3L}
NUM["E3"] = dict(by=e3num, pooled={a: dict(hit5k=v[0], gap5k=v[1], hit2k=v[2]) for a, v in pool.items()},
                 opt_feasible=float(np.mean([r["feasible"] for r in OPT.values()])))

# ------------------------------------------------------------------ E5 sensitivity / ablation
E5 = load("E5_sensitivity")
S = collections.defaultdict(lambda: collections.defaultdict(dict))
for r in E5:
    j = r["_job"]; s = j["spec"]
    S[j["label"]][(s["src"], s["n"], s["m"])][s["seed"]] = (r["curve"][5000 // STEP - 1], r["curve"][-1])
cells = sorted(S["default"])


def vec(label, which):
    return np.array([S[label][c][sd][which] for c in cells for sd in sorted(S["default"][c])])


abl = ["default", "fixed switch 50%", "no elite seeding", "no seeding + fixed switch", "no PSO phase (MMAS + recombination)",
       "no elite recombination", "previous IPAA design (Cho et al., 2025)"]
ABN = {"default": "Full IPAA", "fixed switch 50%": "Fixed hand-over at 50\\%", "no elite seeding": "No elite seeding",
       "no seeding + fixed switch": "Fixed hand-over, no seeding", "no PSO phase (MMAS + recombination)": "No PSO phase (MMAS + recombination)",
       "no elite recombination": "No elite recombination", "previous IPAA design (Cho et al., 2025)": "Previous IPAA design~\\citep{cho2025}"}
lines = ["\\begin{table}[!t]\\centering\\small", "\\caption{Ablation (E5): mean best fitness over 80 instances (synthetic and WS-DREAM, $n\\in\\{30,50\\}$, $m=100$, 20 each). $p$: Wilcoxon signed-rank test against full IPAA.}\\label{tab:ablation}",
         "\\begin{tabular}{@{}lcccc@{}}\\toprule", "Variant & 5k evals & $p$ & 20k evals & $p$ \\\\\\midrule"]
ab = {}
d5, d20 = vec("default", 0), vec("default", 1)
for a in abl:
    v5, v20 = vec(a, 0), vec(a, 1)
    p5 = wilcoxon(d5, v5).pvalue if a != "default" and not np.allclose(d5, v5) else None
    p20 = wilcoxon(d20, v20).pvalue if a != "default" and not np.allclose(d20, v20) else None
    fp = lambda p: "--" if p is None else ("$<$0.001" if p < 0.001 else f"{p:.3f}")
    lines.append(f"{ABN[a]} & {v5.mean():.4f} & {fp(p5)} & {v20.mean():.4f} & {fp(p20)} \\\\")
    ab[a] = dict(m5=float(v5.mean()), p5=p5, m20=float(v20.mean()), p20=p20)
lines += ["\\bottomrule\\end{tabular}", "\\end{table}"]
open(f"{OUT}/tab_ablation.tex", "w").write("\n".join(lines))
NUM["E5_ablation"] = ab
# sensitivity figure
params = [("window", "Stagnation window $W$"), ("div_min", "Diversity threshold $\\delta$"), ("min_pso", "Minimum PSO share $\\beta_{\\min}$"),
          ("elite", "Elite archive size $E$"), ("seed_gain", "Seeding gain $\\gamma$"), ("rho", "Evaporation $\\rho$"),
          ("beta", "Heuristic exponent $\\beta$"), ("xover", "Recombination share $q$"), ("pop", "Population size")]
DEF = dict(window=8, div_min=0.02, min_pso=0.05, elite=10, seed_gain=0.2, rho=0.2, beta=2.5, xover=0.5, pop=40)
fig, axes = plt.subplots(3, 3, figsize=(7.0, 5.6))
sens = {}
for ax, (p, t) in zip(axes.ravel(), params):
    labs = [l for l in S if l.startswith(p + "=")]
    vals = sorted(labs, key=lambda l: float(l.split("=")[1]))
    xs = [float(l.split("=")[1]) for l in vals]
    for which, c, nm in ((0, "#0072b2", "5k"), (1, "#d55e00", "20k")):
        ys = [vec(l, which).mean() for l in vals]
        ax.plot(range(len(xs)), ys, marker="o", ms=3, color=c, label=nm)
        sens[f"{p}|{nm}"] = dict(zip([str(x) for x in xs], [round(y, 4) for y in ys]))
    k = [i for i, x in enumerate(xs) if abs(x - DEF[p]) < 1e-9]
    if k:
        ax.axvline(k[0], color="#999", lw=0.6, ls=":")
    ax.set_xticks(range(len(xs))); ax.set_xticklabels([("%g" % x) for x in xs], fontsize=7)
    ax.set_title(t, fontsize=8); ax.grid(alpha=0.25, lw=0.4)
axes[0, 0].legend(fontsize=7)
for a in axes[:, 0]:
    a.set_ylabel("Mean best fitness")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_sensitivity.pdf", bbox_inches="tight"); plt.close(fig)
NUM["E5_sens"] = sens

# ------------------------------------------------------------------ E6 large scale
E6 = load("E6_largescale")
G = collections.defaultdict(lambda: collections.defaultdict(list))
for r in E6:
    s = r["_job"]["spec"]; G[(s["n"], s["m"])][r["_job"]["alg"]].append((r["fit"], r["time"], r["curve"][19]))
lines = ["\\begin{table*}[!t]\\centering\\scriptsize",
         "\\caption{Large-scale instances (E6): mean best fitness after 50{,}000 evaluations (10 instances each) and mean wall-clock time per run in seconds.}\\label{tab:e6}",
         "\\setlength{\\tabcolsep}{2.6pt}", "\\begin{tabular}{@{}l" + "r" * len(A3L) + "@{}}\\toprule",
         "$n\\times m$ & " + " & ".join(lab(a) for a in A3L) + " \\\\\\midrule"]
e6 = {}
for sz in sorted(G):
    mu = {a: np.mean([x[0] for x in G[sz][a]]) for a in A3L}
    best = max(mu.values())
    lines.append(f"{sz[0]}$\\times${sz[1]} & " + " & ".join(("\\textbf{%.4f}" if abs(mu[a] - best) < 5e-5 else "%.4f") % mu[a] for a in A3L) + " \\\\")
    e6[f"{sz[0]}x{sz[1]}"] = dict(fit=mu, time={a: np.mean([x[1] for x in G[sz][a]]) for a in A3L},
                                  fit20k={a: np.mean([x[2] for x in G[sz][a]]) for a in A3L})
lines.append("\\midrule")
for sz in sorted(G):
    lines.append(f"Time {sz[0]}$\\times${sz[1]} & " + " & ".join(f"{e6[f'{sz[0]}x{sz[1]}']['time'][a]:.2f}" for a in A3L) + " \\\\")
lines += ["\\bottomrule\\end{tabular}", "\\end{table*}"]
open(f"{OUT}/tab_e6.tex", "w").write("\n".join(lines))
NUM["E6"] = e6

# ------------------------------------------------------------------ E7 dynamic
STR = ["Static", "Centralized re-opt.", "MAS local repair only", "MAS-IPAA (cold start)", "MAS-IPAA (warm start)"]
SLB = {"Static": "Static", "Centralized re-opt.": "Centralized re-optimization", "MAS local repair only": "MAS, local repair only",
       "MAS-IPAA (cold start)": "MAS, cold-start escalation", "MAS-IPAA (warm start)": "MAS, warm-start escalation (proposed)"}
E7a, E7b, E7c = load("E7a_dynamic_synthetic"), load("E7b_failure_sweep"), load("E7c_dynamic_wsdream2")


def dyn_rows(R):
    out = {}
    for s in STR:
        rr = sorted([r for r in R if r["_job"]["strategy"] == s], key=lambda r: r["_job"]["seed"])
        out[s] = {f: np.array([r[f] for r in rr]) for f in ("sla", "util", "adapt", "evals", "cpu")}
    return out


lines = ["\\begin{table*}[!t]\\centering\\small",
         "\\caption{Runtime adaptation (E7). Left: synthetic drift, failures and load bursts (300 steps, 30 traces). Right: trace-driven replay of real response time and throughput from WS-DREAM dataset \\#2 (64 time slices of 15 min, 30 traces).}\\label{tab:e7}",
         "\\setlength{\\tabcolsep}{3pt}", "\\begin{tabular}{@{}lrrrrrrrr@{}}\\toprule",
         "& \\multicolumn{4}{c}{Synthetic (E7a)} & \\multicolumn{4}{c}{WS-DREAM \\#2 (E7c)} \\\\\\cmidrule(lr){2-5}\\cmidrule(l){6-9}",
         "Strategy & SLA (\\%) & Utility & Evals & CPU (s) & SLA (\\%) & Utility & Evals & CPU (s) \\\\\\midrule"]
da, dc = dyn_rows(E7a), dyn_rows(E7c)
for s in STR:
    a, c = da[s], dc[s]
    lines.append(f"{SLB[s]} & {a['sla'].mean()*100:.2f} & {a['util'].mean():.4f} & {a['evals'].mean():,.0f} & {a['cpu'].mean():.3f} & "
                 f"{c['sla'].mean()*100:.2f} & {c['util'].mean():.4f} & {c['evals'].mean():,.0f} & {c['cpu'].mean():.3f} \\\\".replace(",", "{,}"))
lines += ["\\bottomrule\\end{tabular}", "\\end{table*}"]
open(f"{OUT}/tab_e7.tex", "w").write("\n".join(lines))
w = lambda D, f, x, y: float(wilcoxon(D[x][f], D[y][f]).pvalue) if not np.allclose(D[x][f], D[y][f]) else 1.0
NUM["E7a"] = {s: {f: float(v.mean()) for f, v in da[s].items()} for s in STR}
NUM["E7c"] = {s: {f: float(v.mean()) for f, v in dc[s].items()} for s in STR}
for nm, D in (("E7a", da), ("E7c", dc)):
    NUM[nm + "_p"] = {f"{f}|warm_vs_{o}": w(D, f, "MAS-IPAA (warm start)", o) for f in ("sla", "util") for o in STR if o != "MAS-IPAA (warm start)"}
# E7b figure + numbers
pfs = sorted({r["_job"]["pf"] for r in E7b})
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3))
C2 = {"Centralized re-opt.": "#0072b2", "MAS local repair only": "#9e9e9e", "MAS-IPAA (cold start)": "#e69f00", "MAS-IPAA (warm start)": "#d55e00"}
e7b = {}
for s in STR[1:]:
    sla = [np.mean([r["sla"] for r in E7b if r["_job"]["pf"] == pf and r["_job"]["strategy"] == s]) * 100 for pf in pfs]
    ut = [np.mean([r["util"] for r in E7b if r["_job"]["pf"] == pf and r["_job"]["strategy"] == s]) for pf in pfs]
    ev = [np.mean([r["evals"] for r in E7b if r["_job"]["pf"] == pf and r["_job"]["strategy"] == s]) for pf in pfs]
    e7b[s] = dict(sla=sla, util=ut, evals=ev)
    for ax, y in zip(axes, (sla, ut, ev)):
        ax.plot(pfs, y, marker="o", ms=3, color=C2[s], label=SLB[s].replace(" (proposed)", ""))
for ax, t in zip(axes, ("SLA compliance (%)", "Delivered utility", "Optimization evaluations")):
    ax.set_xscale("log"); ax.set_xticks(pfs); ax.set_xticklabels([f"{p:g}" for p in pfs], fontsize=7); ax.minorticks_off()
    ax.set_xlabel("Failure probability per step"); ax.set_title(t, fontsize=8); ax.grid(alpha=0.25, lw=0.4)
axes[2].set_yscale("log")
axes[0].legend(fontsize=6, loc="lower left")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_failure_sweep.pdf", bbox_inches="tight"); plt.close(fig)
NUM["E7b"] = dict(pfs=pfs, data=e7b)
# trace figure for E7a
fig, ax = plt.subplots(figsize=(7.0, 2.2))
L2 = {"Static": ":", "Centralized re-opt.": "--", "MAS local repair only": ":", "MAS-IPAA (cold start)": "-.", "MAS-IPAA (warm start)": "-"}
C2["Static"] = "#cccccc"
for s in STR:
    U = np.array([r["trace"] for r in E7a if r["_job"]["strategy"] == s]).mean(0)
    U = np.array([U[max(0, i - 2): i + 3].mean() for i in range(len(U))])
    ax.plot(U, color=C2[s], ls=L2[s], lw=1.5 if "warm" in s else 1.0, label=SLB[s].replace(" (proposed)", ""))
for a_, b_ in ((100, 150), (200, 250)):
    ax.axvspan(a_, b_, color="k", alpha=0.06, lw=0)
ax.set_xlim(0, 299); ax.set_ylim(0, 0.85); ax.set_xlabel("Time step"); ax.set_ylabel("Delivered utility")
ax.grid(alpha=0.25, lw=0.4); ax.legend(ncol=5, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, 1.2))
fig.tight_layout(); fig.savefig(f"{OUT}/fig_dynamic.pdf", bbox_inches="tight"); plt.close(fig)

# ------------------------------------------------------------------ E8 negotiation
E8 = load("E8_negotiation")
POL = ["Selfish", "FCFS", "Negotiation", "Negotiation-SLA", "Joint"]
PLB = {"Selfish": "Selfish", "FCFS": "FCFS", "Negotiation": "Negotiation (loss bids)", "Negotiation-SLA": "Negotiation (slack bids)",
       "Joint": "Centralized joint IPAA$^{\\dagger}$"}
lines = ["\\begin{table}[!t]\\centering\\small",
         "\\caption{Concurrent requests sharing capacity-limited services (E8), capacity 2 per service, mean of 30 episodes. Jain's index is reported only when all utilities are non-negative.}\\label{tab:e8}",
         "\\setlength{\\tabcolsep}{3pt}", "\\begin{tabular}{@{}clrrrrr@{}}\\toprule",
         "$K$ & Policy & Mean $U$ & Min $U$ & Jain & SLA (\\%) & Msgs \\\\\\midrule"]
e8 = {}
for K in (10, 20, 30):
    for p in POL:
        rr = sorted([r for r in E8 if r["_job"]["K"] == K and r["_job"]["policy"] == p], key=lambda r: r["_job"]["seed"])
        v = {f: np.array([r[f] for r in rr]) for f in ("mean", "min", "jain", "sla", "msgs", "overload")}
        e8[f"{K}|{p}"] = {f: float(x.mean()) for f, x in v.items()}
        jain = "--" if (v["min"] < 0).any() else f"{v['jain'].mean():.3f}"
        lines.append((f"\\multirow{{5}}{{*}}{{{K}}}" if p == "Selfish" else "") +
                     f" & {PLB[p]} & {v['mean'].mean():.3f} & {v['min'].mean():.3f} & {jain} & {v['sla'].mean()*100:.1f} & {v['msgs'].mean():.0f} \\\\")
    if K != 30:
        lines.append("\\midrule")
lines += ["\\bottomrule\\end{tabular}",
          "\\par\\smallskip\\parbox{\\linewidth}{\\footnotesize $^{\\dagger}$Reference that optimizes all $K$ plans jointly with $K$ times the evaluation budget of one request.}",
          "\\end{table}"]
open(f"{OUT}/tab_e8.tex", "w").write("\n".join(lines))
for K in (10, 20, 30):
    for f in ("mean", "min", "sla"):
        for a_, b_ in (("Negotiation", "FCFS"), ("Negotiation-SLA", "FCFS"), ("Negotiation-SLA", "Negotiation"), ("Joint", "Negotiation")):
            x = np.array([r[f] for r in sorted([r for r in E8 if r["_job"]["K"] == K and r["_job"]["policy"] == a_], key=lambda r: r["_job"]["seed"])])
            y = np.array([r[f] for r in sorted([r for r in E8 if r["_job"]["K"] == K and r["_job"]["policy"] == b_], key=lambda r: r["_job"]["seed"])])
            e8[f"p|{K}|{f}|{a_}_vs_{b_}"] = float(wilcoxon(x, y).pvalue) if not np.allclose(x, y) else 1.0
NUM["E8"] = e8

json.dump(NUM, open(f"{OUT}/numbers.json", "w"), indent=1, default=float)
print("ok")
