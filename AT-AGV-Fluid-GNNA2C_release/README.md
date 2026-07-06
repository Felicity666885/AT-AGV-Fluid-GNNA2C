# Dynamic Scheduling Algorithm for Internal Container Trucks in Container Terminals Based on a Continuous-Time Fluid Model and GNN-A2C

本仓库提供论文《基于流体模型与GNN-A2C的集装箱码头内集卡动态调度算法》的算法代码、算例实现、正式实验配置和主要结果复现脚本。

## Method overview

The released implementation combines:

- an analytical continuous-time fluid model for idle, loaded, and rebalancing vehicles;
- deterministic FCFS task matching under non-anticipative task releases;
- a directed GNN Actor-Critic that outputs a Dirichlet distribution over target idle-vehicle proportions;
- largest-remainder integer target rounding;
- integer minimum-travel-time rebalancing;
- Uniform and demand-driven proportional rebalancing baselines.

The policy changes only the rebalancing target. All methods share the same task
arrivals, FCFS matching, travel-time matrix, fluid dynamics, reward definition,
initial fleet, and termination criteria.

## Official case-study configuration

- Vessel case: MSAIV610A
- Processed tasks: 3,207
- Regions: 16
- Directed regional edges: 34
- Fleet: 64 internal trucks, initially 4 per region
- Decision period: 15 min
- Training seeds: 20260801, 20260811, 20260821
- Training length: 1,000 episodes
- Discount factor: 0.95
- n-step length: 5
- Entropy coefficient: 0.01
- Actor/Critic learning rates: 1e-4 / 1e-4
- Gradient clipping threshold: 0.5

The machine-readable specification is [`configs/official_experiment.yaml`](configs/official_experiment.yaml).

## Installation

Python 3.11 or newer is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
```

Alternatively:

```bash
conda env create -f environment.yml
conda activate at-agv-fluid-gnna2c
```

The released code uses native PyTorch and does not require PyTorch Geometric or DGL.

## Data

The repository contains de-identified, processed research data only. It does
not include the original terminal database, raw workbooks, container numbers,
personnel data, absolute timestamps, local paths, or other-vessel records.
Task identifiers are synthetic, and release times are minutes relative to the
case-study origin.

```bash
python scripts/prepare_data.py
```

This validates `scenario_MSAIV610A_processed.json` and regenerates the public
CSV views. See [`data/README.md`](data/README.md).

The abstract 16-region layout is a model-based schematic. This repository does
not distribute a Google Maps base image or claim surveyed geographic boundaries.

## Reproduce released tables and figures

These commands do not train a model:

```bash
python scripts/reproduce_tables.py
python scripts/reproduce_figures.py
```

Paper correspondence:

| Paper item | Reproduction source | Output |
|---|---|---|
| Table 2 | `results/metrics/formal_training_summary.csv` | `results/tables/table2_training_window_comparison.csv` |
| Table 3 | `results/metrics/method_comparison_raw_results.csv` | `results/tables/table3_method_comparison.csv` |
| Fig. 5 | released control-chain implementation | `figures/fig5_task_matching_rebalancing.png` |
| Fig. 6 | processed tasks + abstract layout | `figures/fig6_task_origin_destination_distribution.png` |
| Fig. 7 | `task_arrivals_15min.csv` | `figures/fig7_task_arrival_distribution.png` |
| Fig. 8 | three-seed 1,000-episode training summary | `figures/fig8_training_stability.png` |
| Fig. 9 | unified method-comparison metrics | `figures/fig9_cost_service_tradeoff.png` |

Each plotted result also has a compact CSV under `figures/source_data/`.

## Deterministic evaluation

The bundled final checkpoints can be re-evaluated together with both baselines:

```bash
python scripts/evaluate_official_methods.py --config configs/official_experiment.yaml
```

This evaluates the final Episode-1000 checkpoint for every official seed. It
does not select the best checkpoint and does not update model or optimizer
state. Uniform and Demand-driven are deterministic single runs; GNN-A2C is
reported over three independently trained seeds.

## Re-training

Full training is computationally expensive. Start or resume the official setup with:

```bash
python scripts/train_gnn_a2c.py --config configs/official_experiment.yaml --resume
```

Checkpoints and retrained summaries are written beneath `outputs/`, which is
ignored by Git. Training results can vary slightly with software versions,
hardware, and random-number implementations.

## Scope and licensing

This public release excludes all unpublished ablation work, failed runs,
temporary artifacts, development checkpoints, raw terminal data, and private
map imagery. It contains only the main GNN-A2C method and the two baselines
reported in the paper.

The project builds on Gammelli et al.'s GNN-for-AMoD research code. The upstream
MIT license is preserved in [`LICENSE`](LICENSE); modifications and attribution
are described in [`THIRD_PARTY_NOTICE.md`](THIRD_PARTY_NOTICE.md).

