"""Train the three official GNN-A2C seeds with resumable checkpoints."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import time

from _common import ROOT, load_official_config, load_scenario_and_graph
from src.experiments.final_experiment import (
    TRAINING_COLUMNS,
    atomic_save_checkpoint,
    checkpoint_metadata,
    make_training_objects,
    optimizer_steps,
    set_all_seeds,
    training_row,
    validate_checkpoint,
    validate_episode,
)
from src.experiments.full_episode_runner import run_full_episode


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(TRAINING_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/official_experiment.yaml")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = load_official_config(args.config)
    scenario, graph = load_scenario_and_graph(config)
    output = ROOT / str(config.raw["output_dir"])
    summary_path = output / "formal_training_summary.csv"
    rows: list[dict[str, object]] = list(read_rows(summary_path))
    target = int(config.raw["episodes"])
    interval = int(config.raw["checkpoint_interval"])

    for seed in config.formal_seeds:
        checkpoint_dir = output / "checkpoints" / f"formal_seed_{seed}"
        latest = checkpoint_dir / "latest.pt"
        final = checkpoint_dir / f"final_episode_{target}.pt"
        actor, critic, updater = make_training_objects(config)
        completed = 0
        cumulative_runtime = 0.0
        if args.resume and latest.exists():
            metadata = validate_checkpoint(latest, updater, config, seed, None)
            completed = int(metadata["episode_number"])
            cumulative_runtime = float(metadata.get("cumulative_runtime_seconds", 0.0))
            rows = [r for r in rows if int(r["seed"]) != seed or int(r["episode"]) <= completed]
        else:
            set_all_seeds(seed)
            rows = [r for r in rows if int(r["seed"]) != seed]

        for episode in range(completed + 1, target + 1):
            before = optimizer_steps(updater)
            result = run_full_episode(
                scenario=scenario, graph=graph, feature_scaler=config.feature_scales,
                reward_config=config.reward_config,
                initial_idle_distribution=config.initial_idle,
                fleet_size=config.fleet_size, config=config.episode_config,
                run_type="training", episode_number=episode,
                actor=actor, critic=critic, updater=updater,
            )
            after = optimizer_steps(updater)
            validate_episode(result, scenario.task_count, before, after)
            cumulative_runtime += float(result.summary["episode_runtime_seconds"])
            rows.append(training_row(result, "formal", seed, after))
            rows.sort(key=lambda row: (int(row["seed"]), int(row["episode"])))
            write_rows(summary_path, rows)
            if episode % interval == 0 or episode == target:
                metadata = checkpoint_metadata(
                    config, seed, episode, cumulative_runtime,
                    config.config_hash, "formal",
                )
                atomic_save_checkpoint(latest, updater, metadata)
                if episode == target:
                    atomic_save_checkpoint(final, updater, metadata)
                print(f"seed={seed} episode={episode}/{target}", flush=True)


if __name__ == "__main__":
    main()

