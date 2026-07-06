"""Deterministically evaluate Uniform, Demand-driven, and released GNN-A2C."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from _common import ROOT, load_official_config, load_scenario_and_graph
from src.experiments.final_experiment import make_training_objects, state_digest, validate_checkpoint
from src.experiments.method_comparison import run_unified_deterministic_evaluation


COLUMNS = (
    "method", "run_id", "seed", "sample_size", "cumulative_scaled_reward",
    "service_rate_by_release_horizon", "mean_waiting_time", "median_waiting_time",
    "p95_waiting_time", "total_waiting_cost", "total_loaded_travel_cost",
    "total_rebalancing_cost", "total_rebalanced_vehicles",
    "backlog_at_release_horizon", "matched_by_release_horizon",
    "total_episode_steps", "queue_cleared_step",
    "maximum_vehicle_conservation_error", "maximum_task_flow_conservation_error",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/official_experiment.yaml")
    parser.add_argument("--output", default="outputs/evaluation/method_comparison_raw_results.csv")
    args = parser.parse_args()
    config = load_official_config(args.config)
    scenario, graph = load_scenario_and_graph(config)
    common = dict(
        scenario=scenario, graph=graph, feature_scaler=config.feature_scales,
        reward_config=config.reward_config,
        initial_idle_distribution=config.initial_idle,
        fleet_size=config.fleet_size,
        arrival_window_steps=int(config.raw["a2c_config"]["arrival_window_steps"]),
        transit_clear_tolerance=float(config.raw["a2c_config"]["transit_clear_tolerance"]),
        max_decision_steps=int(config.raw["max_decision_steps"]),
    )
    rows: list[dict[str, object]] = []
    for mode, label in (("uniform", "Uniform"), ("demand_driven", "Demand-driven")):
        result = run_unified_deterministic_evaluation(
            method_mode=mode, run_id=f"{mode}_deterministic", **common
        )
        rows.append({"method": label, "sample_size": 1, **result.metrics})

    for seed in config.formal_seeds:
        actor, critic, updater = make_training_objects(config)
        checkpoint = ROOT / "results" / "checkpoints" / f"gnn_a2c_seed_{seed}_episode_1000.pt"
        validate_checkpoint(checkpoint, updater, config, seed, 1000)
        before = state_digest(updater)
        result = run_unified_deterministic_evaluation(
            method_mode="gnn_a2c", run_id=f"episode1000_seed_{seed}", seed=seed,
            actor=actor, critic=critic, **common,
        )
        if before != state_digest(updater):
            raise RuntimeError("deterministic evaluation modified model or optimizer")
        rows.append({"method": "GNN-A2C", "sample_size": 3, **result.metrics})

    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COLUMNS), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(output)


if __name__ == "__main__":
    main()

