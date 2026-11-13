# Age- and Link-Lifetime-Aware Resource Allocation in UAV–IoV–MEC Networks via RL-Guided PSO–ACO

Simulator, optimizers, trained Q-table, and per-seed results for the article

> S. Cho, Y. Lee, and C. W. Nam, "Age- and Link-Lifetime-Aware Resource Allocation in UAV–IoV–MEC Networks via RL-Guided PSO–ACO," *Computer Modeling in Engineering & Sciences*, 2026.

Repository: https://github.com/css3617-lab/css-project  
Archived version (Zenodo): DOI to be added after the first release.

## Contents

| Path | Contents |
|---|---|
| `code/sim.py` | UAV–IoV–MEC simulator: mobility, stale reports and dead-reckoning extrapolation, channels and sub-band interference, HARQ, multi-hop backhaul, processor sharing, MEC background load and backlog, link lifetime, UAV energy (Sections 3–4) |
| `code/algos.py` | Greedy, RS, GA, DE, PSO, TVAC-PSO, ACO, PSO–ACO, and the proposed RL-guided PSO–ACO, all under the same evaluation budget (Sections 5, 6.2) |
| `code/runner.py` | Closed-loop run over the control intervals of one seed |
| `code/train_rl.py` | Offline Q-learning on training seeds 1000–1059 |
| `code/validate.py` | Design checks on validation seeds 500–505 |
| `code/experiments.py` | All test-seed experiments (seeds 1–20); writes `results/all.pkl` |
| `code/analyze.py`, `code/make_tables.py` | Statistics (t-based 95% CIs, Wilcoxon signed-rank tests with Holm correction), figures, and tables |
| `results/Q.npy` | Trained, frozen Q-table used in all test runs |
| `results/per_interval.csv` | One row per experiment, variant, algorithm, density, seed, and control interval |
| `results/per_seed.csv` | Per-seed values (mean over the eight intervals of a run); all statistics in the article are computed from this file |
| `results/table_V{20,40,60,80,100}.csv` | Mean, SD, and 95% CI of every metric per algorithm and density (Tables 3–5, Fig. 3) |
| `results/ablation.csv` | Ablation at 80 vehicles (Tables 6–7) |
| `results/age.csv` | State-age experiment (Fig. 6) |
| `results/weights.csv` | Weight sensitivity at 60 vehicles, 10 seeds (Table 8) |
| `results/class_runtime.csv`, `results/runtime.csv` | Per-class QoS (Table 5) and CPU time (Section 7.8) |
| `results/conv_medians.json` | Median convergence curves (Fig. 4) |
| `results/all.pkl` | Raw per-interval records as produced by `experiments.py` (numpy/pandas objects) |
| `supplementary/Supplementary_Materials.docx` | Supplementary Tables S1–S4 |
| `supplementary/supplementary_ci.csv` | Mean, SD, and 95% CI half-width of every metric for every experiment, variant, algorithm, and density |

## Units in the CSV files

`F` objective; `Q`, `Q0`–`Q2` (safety, perception, infotainment), `feasible`, `sv` (safety-deadline violations), `cf` (link-lifetime failures), `local`, `mec`, `uavc`, `cloud` are fractions (multiply by 100 for %); `L_mean`, `L_p95`, `T_*` are seconds; `E`, `E_*` are joules per interval (controllable energy, excluding hover); `cpu` is seconds per interval; `N` is the number of tasks per interval.

## Reproducing the results

Requirements: Python 3.10+, numpy, scipy, pandas, matplotlib. Run the scripts from inside `code/`:

```
python3 train_rl.py 60      # Q-table (optional; results/Q.npy is provided)
python3 validate.py         # design checks on validation seeds
python3 experiments.py      # all test experiments -> results/all.pkl
python3 analyze.py          # statistics and figures
python3 make_tables.py      # tables
```

Seeds: Q-table training 1000–1059, validation 500–505, test 1–20. CPU times depend on the machine; all other results are deterministic for fixed seeds.

## License

Code: MIT License (see `LICENSE`). Data and results in `results/` and `supplementary/`: Creative Commons Attribution 4.0 International (CC BY 4.0).

## Citation

Please cite the article above and the archived version on Zenodo (see `CITATION.cff`).
