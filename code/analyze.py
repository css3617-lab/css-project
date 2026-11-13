import pickle, json, sys
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "../results/"
P = "../paper/"
df = pd.DataFrame(pickle.load(open(R + "all.pkl", "rb")))
ORDER = ["Greedy", "RS", "GA", "DE", "PSO", "TVAC-PSO", "ACO", "PSO-ACO", "Proposed"]
COL = {"Proposed": "#2a78d6", "PSO-ACO": "#eb6834", "TVAC-PSO": "#1baf7a", "PSO": "#eda100",
       "ACO": "#e87ba4", "DE": "#008300", "GA": "#6250d6", "Greedy": "#e34948", "RS": "#8a8984"}
MK = {"Proposed": "o", "PSO-ACO": "s", "TVAC-PSO": "^", "PSO": "v", "ACO": "D", "DE": "P", "GA": "X", "Greedy": "*", "RS": "."}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#8a8984", "axes.linewidth": 0.6, "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "axes.labelcolor": "#222222", "grid.color": "#e6e5e0", "grid.linewidth": 0.5, "legend.frameon": False})
METRICS = ["F", "Q", "L_mean", "L_p95", "E", "R", "feasible", "sv", "cf", "cpu", "Q0", "Q1", "Q2", "E_move", "E_tx", "E_cmp", "E_bh",
           "T_up", "T_bh", "T_q", "T_cmp", "T_dn", "local", "mec", "uavc", "cloud", "N"]


def per_seed(d):
    return d.groupby(["tag", "alg", "V", "seed"])[METRICS].mean().reset_index()


def ci95(x):
    x = np.asarray(x); n = len(x)
    return stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0


def holm(p):
    p = np.asarray(p); m = len(p); o = np.argsort(p); adj = np.empty(m); run = 0
    for i, j in enumerate(o):
        run = max(run, (m - i) * p[j]); adj[j] = min(run, 1.0)
    return adj


main = per_seed(df[df.exp == "main"])
out = {}

# ---------------------------------------------------------------- main table per density
def table(V):
    d = main[main.V == V]
    rows = []
    ref = d[d.alg == "Proposed"].set_index("seed")
    pv = {}
    for a in ORDER:
        x = d[d.alg == a].set_index("seed")
        if a != "Proposed" and len(x):
            s = ref.index.intersection(x.index)
            try:
                pv[a] = stats.wilcoxon(ref.loc[s, "F"], x.loc[s, "F"]).pvalue
            except ValueError:
                pv[a] = 1.0
    names = list(pv.keys()); adj = dict(zip(names, holm([pv[n] for n in names])))
    for a in ORDER:
        x = d[d.alg == a]
        if not len(x):
            continue
        r = dict(alg=a, n=len(x))
        for m in METRICS:
            r[m] = x[m].mean(); r[m + "_sd"] = x[m].std(ddof=1); r[m + "_ci"] = ci95(x[m])
        r["p_holm"] = adj.get(a, np.nan)
        if a != "Proposed":
            xs = x.set_index("seed"); s = ref.index.intersection(xs.index)
            r["wins"] = int((ref.loc[s, "F"] < xs.loc[s, "F"]).sum())
        rows.append(r)
    return pd.DataFrame(rows)

tabs = {V: table(V) for V in sorted(main.V.unique())}
for V, t in tabs.items():
    t.to_csv(R + f"table_V{V}.csv", index=False)
    print(f"\n=== V={V} ===")
    print(t[["alg", "n", "F", "F_sd", "Q", "L_mean", "L_p95", "E", "E_move", "feasible", "sv", "cf", "cpu", "p_holm", "wins"]].round(4).to_string(index=False))

# ---------------------------------------------------------------- scalability figure
algs8 = ["Greedy", "GA", "DE", "PSO", "TVAC-PSO", "ACO", "PSO-ACO", "Proposed"]
fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.7), dpi=300)
for ax, (m, lab, sc) in zip(axs, [("F", "Objective $F$ (Greedy off-scale)", 1), ("Q", "QoS satisfaction (%)", 100),
                                  ("L_mean", "Mean latency (ms)", 1e3), ("L_p95", "95th-pct latency (ms)", 1e3)]):
    for a in algs8:
        if m == "F" and a == "Greedy":
            continue
        g = main[main.alg == a].groupby("V")[m]
        mu = g.mean() * sc; ci = g.apply(ci95) * sc
        ax.errorbar(mu.index, mu.values, yerr=ci.values, color=COL[a], marker=MK[a], ms=3.6 if a != "Proposed" else 4.5,
                    lw=2.0 if a == "Proposed" else 1.0, capsize=1.5, elinewidth=0.6, label=a, zorder=5 if a == "Proposed" else 3)
    ax.set_xlabel("Number of vehicles"); ax.set_title(lab, fontsize=7.6, color="#222222", loc="left")
    ax.set_xticks([20, 40, 60, 80, 100]); ax.grid(axis="y")
h, l = axs[1].get_legend_handles_labels()
fig.legend(h, l, ncol=8, loc="lower center", bbox_to_anchor=(0.5, 0.0), fontsize=6.6, handlelength=1.6, columnspacing=0.9)
plt.tight_layout(rect=(0, 0.09, 1, 1))
plt.savefig(P + "fig4_scalability.png", dpi=300); plt.close()

# ---------------------------------------------------------------- convergence figure
conv = df[(df.exp == "main") & df.alg.isin(["GA", "DE", "PSO", "TVAC-PSO", "ACO", "PSO-ACO", "Proposed", "RS"])]
CHECK = np.arange(100, 3001, 100)
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9), dpi=300)
for ax, V in zip(axs, [60, 100]):
    for a in ["GA", "DE", "PSO", "TVAC-PSO", "ACO", "PSO-ACO", "Proposed"]:
        c = np.stack(conv[(conv.alg == a) & (conv.V == V)].conv.values)
        # normalise per interval by the best final key among all algorithms on that interval
        med = np.median(c, 0)
        ax.plot(CHECK, med, color=COL[a], lw=2.0 if a == "Proposed" else 1.0, label=a, marker=MK[a], markevery=5, ms=3.2)
    ax.set_xlabel("Objective evaluations (NFE)"); ax.set_title(f"{V} vehicles: median best key $\\tilde F$", fontsize=7.6, loc="left")
    ax.grid(axis="y")
    fin = np.median(np.stack(conv[(conv.alg == "Proposed") & (conv.V == V)].conv.values), 0)[-1]
    ax.set_ylim(fin * 0.95, fin * 1.6)
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, ncol=7, loc="lower center", bbox_to_anchor=(0.5, 0.0), fontsize=6.8)
plt.tight_layout(rect=(0, 0.08, 1, 1))
plt.savefig(P + "fig3_convergence.png", dpi=300); plt.close()
# NFE to reach within 1% of final PSO-ACO / TVAC values
cv = {}
for V in [60, 100]:
    for a in ["PSO", "TVAC-PSO", "ACO", "PSO-ACO", "Proposed", "DE", "GA"]:
        c = np.median(np.stack(conv[(conv.alg == a) & (conv.V == V)].conv.values), 0)
        cv[f"{a}|{V}"] = c.tolist()
json.dump(cv, open(R + "conv_medians.json", "w"))

# ---------------------------------------------------------------- latency decomposition + offloading mix at V=80
d80 = main[main.V == 80]
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.5), dpi=300)
comp = [("T_up", "Uplink", "#2a78d6"), ("T_bh", "Backhaul", "#eb6834"), ("T_q", "Queueing", "#eda100"), ("T_cmp", "Computation", "#1baf7a"), ("T_dn", "Downlink", "#6250d6")]
xs = np.arange(len(algs8))
bottom = np.zeros(len(algs8))
for key, lab, col in comp:
    v = np.array([d80[d80.alg == a][key].mean() * 1e3 for a in algs8])
    axs[0].bar(xs, v, bottom=bottom, color=col, width=0.7, label=lab, edgecolor="white", linewidth=0.8)
    bottom += v
axs[0].set_xticks(xs); axs[0].set_xticklabels(algs8, rotation=40, ha="right", fontsize=6.8)
axs[0].set_title("Mean latency components (ms), 80 vehicles", fontsize=7.6, loc="left"); axs[0].legend(fontsize=6.4, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.38), columnspacing=0.8, handlelength=1.2)
mix = [("local", "Local", "#8a8984"), ("mec", "RSU-MEC", "#2a78d6"), ("uavc", "UAV", "#eb6834"), ("cloud", "Cloud", "#6250d6")]
bottom = np.zeros(len(algs8))
for key, lab, col in mix:
    v = np.array([d80[d80.alg == a][key].mean() * 100 for a in algs8])
    axs[1].bar(xs, v, bottom=bottom, color=col, width=0.7, label=lab, edgecolor="white", linewidth=0.8)
    bottom += v
axs[1].set_xticks(xs); axs[1].set_xticklabels(algs8, rotation=40, ha="right", fontsize=6.8)
axs[1].set_title("Execution location (% of tasks), 80 vehicles", fontsize=7.6, loc="left"); axs[1].legend(fontsize=6.4, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.38))
plt.tight_layout()
plt.savefig(P + "fig5_breakdown.png", dpi=300, bbox_inches="tight", pad_inches=0.05); plt.close()

# ---------------------------------------------------------------- ablation
abl = per_seed(df[df.exp == "abl"])
base = main[(main.V == 80) & (main.alg.isin(["Proposed", "PSO-ACO"]))].copy()
base["tag"] = base.alg.map({"Proposed": "Full method", "PSO-ACO": "w/o RL"})
ab = pd.concat([base, abl])
labs = ["Full method", "w/o RL", "w/o ACO", "w/o rank order", "w/o age-aware", "w/o lifetime", "w/o UAV move"]
full = ab[ab.tag == "Full method"].set_index("seed")
arows = []
for t in labs:
    x = ab[ab.tag == t].set_index("seed")
    if not len(x):
        continue
    s = full.index.intersection(x.index)
    p = stats.wilcoxon(full.loc[s, "F"], x.loc[s, "F"]).pvalue if t != "Full method" else np.nan
    arows.append(dict(variant=t, n=len(x), F=x.F.mean(), F_ci=ci95(x.F), Q=x.Q.mean() * 100, Q_ci=ci95(x.Q) * 100, L=x.L_mean.mean() * 1e3,
                      p95=x.L_p95.mean() * 1e3, E=x.E.mean(), cf=x.cf.mean() * 100, sv=x.sv.mean() * 100, R=x.R.mean(), feas=x.feasible.mean(), p=p))
at = pd.DataFrame(arows)
if len(at) > 1:
    at["p_holm"] = np.r_[np.nan, holm(at.p.values[1:])]
at.to_csv(R + "ablation.csv", index=False)
print("\n=== ablation V=80 ===\n", at.round(4).to_string(index=False))

# ---------------------------------------------------------------- weights
wt = per_seed(df[df.exp == "weights"])
bal = main[(main.V == 60) & (main.seed <= 10) & main.alg.isin(["Greedy", "DE", "TVAC-PSO", "ACO", "PSO-ACO", "Proposed"])].copy()
bal["tag"] = "balanced"
wt = pd.concat([wt, bal])
wrows = []
for wname in ["balanced", "latency", "energy", "equal"]:
    for a in ["Greedy", "DE", "TVAC-PSO", "ACO", "PSO-ACO", "Proposed"]:
        x = wt[(wt.tag == wname) & (wt.alg == a)]
        if len(x):
            wrows.append(dict(weights=wname, alg=a, n=len(x), F=x.F.mean(), F_sd=x.F.std(ddof=1), Q=x.Q.mean() * 100, L=x.L_mean.mean() * 1e3, E=x.E.mean(), E_move=x.E_move.mean()))
wtab = pd.DataFrame(wrows)
wtab.to_csv(R + "weights.csv", index=False)
print("\n=== weights V=60 ===\n", wtab.round(4).to_string(index=False))

# ---------------------------------------------------------------- state age
ag = df[df.exp == "age"].copy()
ag["variant"] = ag.tag.str.split("|").str[0]; ag["tau"] = ag.tag.str.split("|").str[1].astype(float)
m80 = df[(df.exp == "main") & (df.V == 80) & df.alg.isin(["Proposed", "TVAC-PSO"])].copy()
m80["variant"] = m80.alg; m80["tau"] = 2.0
a2 = df[(df.exp == "abl") & (df.tag == "w/o age-aware")].copy(); a2["variant"] = "Proposed (age-unaware)"; a2["tau"] = 2.0
ag = pd.concat([ag, m80, a2])
ags = ag.groupby(["variant", "tau", "seed"])[["F", "Q", "cf", "L_mean", "R", "sv"]].mean().reset_index()
agt = ags.groupby(["variant", "tau"]).agg(F=("F", "mean"), F_ci=("F", ci95), Q=("Q", "mean"), Q_ci=("Q", ci95), cf=("cf", "mean"), cf_ci=("cf", ci95), R=("R", "mean"), sv=("sv", "mean")).reset_index()
agt.to_csv(R + "age.csv", index=False)
print("\n=== state age V=80 ===\n", agt.round(4).to_string(index=False))
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.4), dpi=300)
vc = {"Proposed": ("#2a78d6", "o", "-"), "Proposed (age-unaware)": ("#2a78d6", "o", "--"), "TVAC-PSO": ("#1baf7a", "^", "-"), "TVAC-PSO (age-unaware)": ("#1baf7a", "^", "--")}
for v, (c, mk, ls) in vc.items():
    x = agt[agt.variant == v].sort_values("tau")
    if len(x) < 2:
        continue
    axs[0].errorbar(x.tau, x.Q * 100, yerr=x.Q_ci * 100, color=c, marker=mk, ls=ls, lw=1.4, ms=4, capsize=1.5, label=v)
    axs[1].errorbar(x.tau, x.cf * 100, yerr=x.cf_ci * 100, color=c, marker=mk, ls=ls, lw=1.4, ms=4, capsize=1.5, label=v)
axs[0].set_title("QoS satisfaction (%)", fontsize=7.6, loc="left"); axs[1].set_title("Link-lifetime failures (% of tasks)", fontsize=7.6, loc="left")
for ax in axs:
    ax.set_xlabel("Maximum report age $\\tau_{max}$ (s)"); ax.grid(axis="y"); ax.set_xticks([0, 1, 2, 4])
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.0), fontsize=6.8)
plt.tight_layout(rect=(0, 0.09, 1, 1))
plt.savefig(P + "fig6_age.png", dpi=300); plt.close()

# ---------------------------------------------------------------- per-class QoS & runtime
cls = main[main.V.isin([60, 100])].groupby(["V", "alg"])[["Q0", "Q1", "Q2", "sv", "cpu", "N"]].mean().reset_index()
cls.to_csv(R + "class_runtime.csv", index=False)
print("\n=== class QoS / runtime ===\n", cls.round(4).to_string(index=False))
rt = main.groupby(["V", "alg"]).cpu.mean().unstack().round(3)
rt.to_csv(R + "runtime.csv")
print(rt)
print("tasks per interval:", main.groupby("V").N.mean().round(1).to_dict())
