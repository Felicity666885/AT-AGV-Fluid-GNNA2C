# GitHub release package final check

## Release location

`AT-AGV-Fluid-GNNA2C_release/`

This directory is a newly assembled release package. No source file, raw data
file, checkpoint, or experiment artifact in the original project was deleted,
renamed, or overwritten.

## Included modules

- strict processed-scenario loading and public CSV regeneration;
- 16-region directed network construction and six-dimensional node features;
- FCFS task ledger and matching;
- analytical continuous-time fluid state update;
- largest-remainder target rounding and integer rebalancing optimization;
- independent directed-GNN Actor and Critic with Dirichlet actions;
- n-step A2C rollout, reward, optimization, drain, and bootstrap interfaces;
- resumable official three-seed training entry point;
- deterministic Uniform, Demand-driven, and GNN-A2C evaluation;
- Table 2, Table 3, and copyright-safe Figure 5--Figure 9 reproduction scripts.

## Exclusion and privacy checks

- **Unpublished ablation content:** excluded. No ablation experiment directory,
  model, result, checkpoint, table, or figure is present.
- **Raw data:** excluded. No Excel workbook or original private scenario JSON is
  present. The published scenario and CSV files contain only synthetic task IDs,
  relative release minutes, decision steps, region OD pairs, regional edges,
  metadata, and mean travel times.
- **Private fields:** no container number, personnel information, workbook name,
  worksheet name, original row number, absolute datetime, other-vessel name,
  internal source field, or machine path is present in the released data.
- **Map imagery:** excluded. No Google Maps image or original terminal map is
  distributed. Figure 6 uses the abstract 16-region model layout.
- **Local absolute paths:** recursive text scan found no machine-specific user,
  desktop, home-directory, or temporary-upload path reference.
- **Temporary/development files:** no cache directory, `.pyc`, `.tmp`, `.bak`,
  `.log`, IDE configuration, Episode-50, Episode-200, or `latest.pt` file is present.
- **Unpublished document drafts:** no Word, PDF, PowerPoint, or spreadsheet draft
  is present.

## Released data and checkpoints

- Processed MSAIV610A tasks: 3,207 unique synthetic task IDs.
- Region network: 16 regions and 34 directed edges.
- Travel-time data: 240 non-self OD values.
- Final checkpoints: exactly three Episode-1000 files for seeds 20260801,
  20260811, and 20260821.
- Checkpoint checks: seed and episode metadata match; Actor, Critic, both Adam
  states, and Python/NumPy/PyTorch RNG states are present; network tensors are finite.

No file exceeds 10 MiB. The package is small enough for ordinary Git storage;
Git LFS is not required for the included files.

## Paper result provenance

- **Table 2:** generated from `results/metrics/formal_training_summary.csv`,
  using all three official seeds and the Episode 801--900 and 901--1000 windows.
- **Table 3:** generated from
  `results/metrics/method_comparison_raw_results.csv`; Uniform and Demand-driven
  are deterministic samples of size one, while GNN-A2C uses three official seeds.
- **Figure 5:** generated from the released control-chain definition.
- **Figure 6:** generated from the processed tasks and abstract region layout.
- **Figure 7:** generated from processed 15-minute task arrivals.
- **Figure 8:** generated from all 3,000 official training-summary rows.
- **Figure 9:** generated from the unified deterministic method comparison.

## Reproduction check

Executed successfully from the release root:

```bash
python scripts/reproduce_tables.py
python scripts/reproduce_figures.py
```

Successfully regenerated:

- `results/tables/table2_training_window_comparison.csv`;
- `results/tables/table3_method_comparison.csv`;
- `figures/fig5_task_matching_rebalancing.png`;
- `figures/fig6_task_origin_destination_distribution.png`;
- `figures/fig7_task_arrival_distribution.png`;
- `figures/fig8_training_stability.png`;
- `figures/fig9_cost_service_tradeoff.png`.

`python -m compileall -q scripts src` also completed successfully. The processed
scenario was loaded by the release copy of the strict scenario loader, and all
released checkpoints passed metadata/state validation. Full 1,000-episode
re-training was intentionally not repeated during packaging.

## Final status

The folder is ready to be copied into the target GitHub repository. It contains
no author metadata file (`CITATION.cff`) because author, affiliation, ORCID, DOI,
and publication-year details were not supplied and were not guessed.
