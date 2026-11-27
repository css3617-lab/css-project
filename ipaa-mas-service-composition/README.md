# Pheromone-retaining multi-agent QoS-aware service composition (IPAA)

Repository folder: https://github.com/css3617-lab/css-project/tree/main/ipaa-mas-service-composition

Code, data pointers and complete result files for the manuscript

> S. Cho, *Pheromone-retaining multi-agent service composition: a particle–ant hybrid with warm-start
> adaptation and capacity negotiation evaluated on real QoS data* (submitted).

The repository contains

* `ipaa/` – composition model (sequential and structured workflows, synthetic and WS-DREAM QoS),
  ten optimizers under a common evaluation budget (RS, GA, DE, SA, PSO, GWO, WOA, MMAS, PSO–ACO relay, IPAA),
  exact enumeration, resumable parallel runner, WS-DREAM loaders;
* `e1_e2_benchmark.py` … `e8_negotiation.py` – the eight experiments of the paper (E1–E8);
* `results/` – the raw JSON-lines output of every run reported in the paper (21,370 runs), the run log and a text summary;
* `analysis/make_tables_figures.py` – regenerates all statistics, LaTeX tables and figures from `results/`.

## Experiments

| Script | Paper | Content |
|---|---|---|
| `e1_e2_benchmark.py` | E1, E2 | synthetic and WS-DREAM #1 QoS, n ∈ {10,20,30,50}, m ∈ {20,50,100}, 30 instances, 10 algorithms, 20,000 evaluations |
| `e3_exact.py` | E3 | comparison with exact optima obtained by enumerating all mⁿ plans (up to 1.68 × 10⁷) |
| `e4_structured.py` | E4 | random series–parallel workflows with exclusive choices and loops |
| `e5_sensitivity.py` | E5 | one-at-a-time parameter sensitivity and ablation of IPAA |
| `e6_largescale.py` | E6 | up to 100 tasks × 1,000 candidates, 50,000 evaluations, run time |
| `e7_dynamic.py` | E7 | runtime adaptation: synthetic drift/failures/bursts, failure-rate sweep, trace-driven WS-DREAM #2 replay |
| `e8_negotiation.py` | E8 | concurrent requests with capacity-limited services (selfish, FCFS, loss bids, slack bids, centralized joint) |

## Running

```bash
pip install -r requirements.txt
python download_data.py          # WS-DREAM dataset #2 from Zenodo (needed only for E7c)
bash run_all.sh                  # or run_all.bat on Windows; every experiment resumes if interrupted
pip install -r analysis/requirements.txt
python analysis/make_tables_figures.py out results
```

Run all commands from inside this folder (`ipaa-mas-service-composition/`).

Environment variables: `IPAA_WORKERS` (number of processes, default CPU count − 1) and `IPAA_QUICK=1`
(a few-minute smoke test with tiny settings; its results are not those of the paper).
All instances and runs are seeded. The results in `results/` were produced on an Ubuntu 22.04 virtual
machine with 2 vCPUs and 3 GB RAM (Python 3.10, NumPy 2.2) using two worker processes; run times are
therefore comparable only between methods.

## Data

* **WS-DREAM dataset #1** (339 users × 5,825 services; response time and throughput) is included in
  `data/wsdream1/` under its MIT License.
* **WS-DREAM dataset #2** (142 users × 4,500 services × 64 time slices) is not redistributed because of its size;
  `download_data.py` fetches it from Zenodo (record 1133476), or place `rtdata.txt` and `tpdata.txt` in `data/wsdream2/`.
* Please cite Z. Zheng, Y. Zhang and M. R. Lyu, "Investigating QoS of real-world Web services",
  *IEEE Transactions on Services Computing* 7(1):32–39, 2014, when using the WS-DREAM data.
* WS-DREAM contains no prices; costs are synthetic and correlated with observed quality (see the paper).

## License

Code: MIT (see `LICENSE`). WS-DREAM data: MIT, © 2016 WS-DREAM, CUHK.
