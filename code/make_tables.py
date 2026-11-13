import pandas as pd, numpy as np, json
R="../results/"
ORDER=["Greedy","RS","GA","DE","PSO","TVAC-PSO","ACO","PSO-ACO","Proposed"]
def star(p):
    if np.isnan(p): return ""
    return "‡" if p<0.001 else ("†" if p<0.01 else ("*" if p<0.05 else ""))
out={}
# Table 4: V=80 full comparison
for V in [80]:
    t=pd.read_csv(R+f"table_V{V}.csv").set_index("alg")
    rows=["| Method | $F$ | QoS (%) | Mean latency (ms) | P95 latency (ms) | Energy (J) | Feasible (%) | Safety viol. (%) | CPU (s) |","|------------|----------------------|------------|---------------|------------|--------|--------|--------|------|"]
    best={m:(t[m].min() if m in("F","L_mean","L_p95","E") else t[m].max()) for m in ["F","Q","L_mean","L_p95"]}
    for a in ORDER:
        r=t.loc[a]
        f=f"{r.F:.4f} ± {r.F_ci:.4f}{star(r.p_holm)}"
        q=f"{r.Q*100:.1f} ± {r.Q_ci*100:.1f}"
        L=f"{r.L_mean*1e3:.1f} ± {r.L_mean_ci*1e3:.1f}"
        p=f"{r.L_p95*1e3:.0f} ± {r.L_p95_ci*1e3:.0f}"
        e=f"{r.E:.1f}"
        if a=="Proposed": f,q,L,p=[f"**{x}**" for x in (f,q,L,p)]
        rows.append(f"| {a} | {f} | {q} | {L} | {p} | {e} | {r.feasible*100:.1f} | {r.sv*100:.2f} | {r.cpu:.2f} |")
    out["t4"]="\n".join(rows)
# Table 5: F across densities
rows=["| Method | 20 | 40 | 60 | 80 | 100 |","|----------------|----------------|----------------|----------------|----------------|----------------|"]
T={V:pd.read_csv(R+f"table_V{V}.csv").set_index("alg") for V in [20,40,60,80,100]}
for a in ORDER:
    cells=[]
    for V in [20,40,60,80,100]:
        r=T[V].loc[a]; mn=T[V].F.min()
        s=f"{r.F:.4f} ({r.F_sd:.4f}){star(r.p_holm)}"
        if abs(r.F-mn)<1e-12: s=f"**{s}**"
        cells.append(s)
    rows.append(f"| {a} | "+" | ".join(cells)+" |")
rows.append("| Proposed wins vs. best baseline | "+" | ".join(
    f"{int(T[V].drop('Proposed').loc[T[V].drop('Proposed').F.idxmin(),'wins'])}/20 vs. {T[V].drop('Proposed').F.idxmin()}" for V in [20,40,60,80,100])+" |")
out["t5"]="\n".join(rows)
# Table 6 ablation
a=pd.read_csv(R+"ablation.csv")
rows=["| Variant | $F$ | QoS (%) | Mean latency (ms) | P95 latency (ms) | Energy (J) | Lifetime failures (%) | Holm-adjusted $p$ |","|----------------|-----------------|------------|----------|---------|--------|----------|----------|"]
for _,r in a.iterrows():
    ph="–" if np.isnan(r.p_holm) else (f"{r.p_holm:.1e}" if r.p_holm<0.001 else f"{r.p_holm:.3f}")
    rows.append(f"| {r.variant} | {r.F:.4f} ± {r.F_ci:.4f} | {r.Q:.1f} ± {r.Q_ci:.1f} | {r.L:.1f} | {r.p95:.0f} | {r.E:.1f} | {r.cf:.2f} | {ph} |")
out["t6"]="\n".join(rows)
# Table 7 weights
w=pd.read_csv(R+"weights.csv")
rows=["| Weights $(w_L,w_E,w_R,w_Q)$ | Method | $F$ | QoS (%) | Mean latency (ms) | Energy (J) |","|------------------------|------------|----------|----------|------------|----------|"]
names={"balanced":"Balanced (0.3, 0.2, 0.2, 0.3)","latency":"Latency-first (0.5, 0.1, 0.1, 0.3)","energy":"Energy-first (0.15, 0.55, 0.1, 0.2)","equal":"Equal (0.25, 0.25, 0.25, 0.25)"}
for wn in ["balanced","latency","energy","equal"]:
    sub=w[w.weights==wn]; mn=sub.F.min()
    for i,(_,r) in enumerate(sub.iterrows()):
        f=f"{r.F:.4f}"; f=f"**{f}**" if abs(r.F-mn)<1e-12 else f
        rows.append(f"| {names[wn] if i==0 else ''} | {r.alg} | {f} | {r.Q:.1f} | {r.L:.1f} | {r.E:.1f} |")
out["t7"]="\n".join(rows)
# class table V=60/100
c=pd.read_csv(R+"class_runtime.csv")
rows=["| Method | Safety (60) | Perception (60) | Infotainment (60) | Safety (100) | Perception (100) | Infotainment (100) |","|------------|----------|-----------|------------|----------|-----------|------------|"]
for a_ in ORDER:
    r6=c[(c.V==60)&(c.alg==a_)].iloc[0]; r10=c[(c.V==100)&(c.alg==a_)].iloc[0]
    rows.append(f"| {a_} | {r6.Q0*100:.1f} | {r6.Q1*100:.1f} | {r6.Q2*100:.1f} | {r10.Q0*100:.1f} | {r10.Q1*100:.1f} | {r10.Q2*100:.1f} |")
out["t8"]="\n".join(rows)
json.dump(out,open(R+"tables.json","w"),indent=1)
for k,v in out.items(): print(k); print(v); print()
# key derived numbers
for V in [60,80,100]:
    t=T[V]
    P=t.loc["Proposed"]
    for b in ["TVAC-PSO","PSO-ACO","Greedy","ACO"]:
        B=t.loc[b]
        print(V,b,f"dF={100*(P.F-B.F)/B.F:.1f}% dQ={100*(P.Q-B.Q):.1f}pp dL={100*(P.L_mean-B.L_mean)/B.L_mean:.1f}% dP95={100*(P.L_p95-B.L_p95)/B.L_p95:.1f}% dE={100*(P.E-B.E)/B.E:.1f}% feasB={B.feasible*100:.1f}")
for V in [20]:
    t=T[V]; print(V, t[["F","p_holm","wins"]])
