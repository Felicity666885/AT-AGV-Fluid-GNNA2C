"""Minimal checkpoint, summary, and trend helpers for Stage 5B continuation."""

from __future__ import annotations

import copy
import csv
import random
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.experiments.experiment_metrics import moving_average
from src.experiments.full_episode_runner import FullEpisodeResult
from src.training.a2c_update import A2CUpdater


CONTINUATION_COLUMNS = (
    "episode", "cumulative_scaled_reward", "matched_by_release_horizon",
    "service_rate_by_release_horizon", "backlog_at_release_horizon",
    "mean_waiting_time", "p95_waiting_time", "waiting_cost", "rebalancing_cost",
    "total_rebalanced_vehicles", "total_episode_steps", "queue_clear_step",
    "actor_loss_mean", "critic_loss_mean", "actor_entropy_mean", "return_mean",
    "advantage_mean", "actor_grad_norm_mean", "critic_grad_norm_mean",
    "episode_runtime_seconds", "terminated", "truncated", "service_component",
    "loaded_cost", "status",
)

TREND_METRICS = (
    "cumulative_scaled_reward", "service_rate_by_release_horizon", "waiting_cost",
    "rebalancing_cost", "mean_waiting_time", "p95_waiting_time", "total_episode_steps",
)


def result_to_continuation_row(result: FullEpisodeResult) -> dict[str, object]:
    """Flatten only the episode-level fields requested for Stage 5B."""

    summary = result.summary
    return {
        "episode": result.episode_number,
        "cumulative_scaled_reward": summary["scaled_reward"],
        "matched_by_release_horizon": summary["matched_by_release_horizon"],
        "service_rate_by_release_horizon": summary["service_rate_by_release_horizon"],
        "backlog_at_release_horizon": summary["backlog_at_release_horizon"],
        "mean_waiting_time": summary["mean_waiting_time"],
        "p95_waiting_time": summary["p95_waiting_time"],
        "waiting_cost": summary["waiting_cost"],
        "rebalancing_cost": summary["rebalancing_travel_cost"],
        "total_rebalanced_vehicles": summary["total_rebalanced_vehicles"],
        "total_episode_steps": summary["total_episode_steps"],
        "queue_clear_step": summary["queue_clear_step"],
        "actor_loss_mean": summary["actor_loss_mean"],
        "critic_loss_mean": summary["critic_loss_mean"],
        "actor_entropy_mean": summary["actor_entropy_mean"],
        "return_mean": summary["return_mean"],
        "advantage_mean": summary["advantage_mean"],
        "actor_grad_norm_mean": summary["actor_grad_norm_mean"],
        "critic_grad_norm_mean": summary["critic_grad_norm_mean"],
        "episode_runtime_seconds": summary["episode_runtime_seconds"],
        "terminated": summary["terminated"],
        "truncated": summary["truncated"],
        "service_component": summary["service_component"],
        "loaded_cost": summary["loaded_travel_cost"],
        "status": result.status,
    }


def convert_stage5a_row(row: Mapping[str, str]) -> dict[str, object]:
    """Map an existing Stage-5A summary row into the Stage-5B schema."""

    mapped = {
        "episode": int(row["episode"]),
        "cumulative_scaled_reward": float(row["scaled_reward"]),
        "matched_by_release_horizon": int(row["matched_by_release_horizon"]),
        "service_rate_by_release_horizon": float(row["service_rate_by_release_horizon"]),
        "backlog_at_release_horizon": float(row["backlog_at_release_horizon"]),
        "mean_waiting_time": float(row["mean_waiting_time"]),
        "p95_waiting_time": float(row["p95_waiting_time"]),
        "waiting_cost": float(row["waiting_cost"]),
        "rebalancing_cost": float(row["rebalancing_travel_cost"]),
        "total_rebalanced_vehicles": int(row["total_rebalanced_vehicles"]),
        "total_episode_steps": int(row["total_episode_steps"]),
        "queue_clear_step": int(row["queue_clear_step"]),
        "actor_loss_mean": float(row["actor_loss_mean"]),
        "critic_loss_mean": float(row["critic_loss_mean"]),
        "actor_entropy_mean": float(row["actor_entropy_mean"]),
        "return_mean": float(row["return_mean"]),
        "advantage_mean": float(row["advantage_mean"]),
        "actor_grad_norm_mean": float(row["actor_grad_norm_mean"]),
        "critic_grad_norm_mean": float(row["critic_grad_norm_mean"]),
        "episode_runtime_seconds": float(row["episode_runtime_seconds"]),
        "terminated": row["terminated"],
        "truncated": row["truncated"],
        "service_component": float(row["service_component"]),
        "loaded_cost": float(row["loaded_travel_cost"]),
        "status": row["status"],
    }
    return mapped


def capture_rng_states() -> dict[str, object]:
    return {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state().clone(),
    }


def restore_rng_states(payload: Mapping[str, object]) -> bool:
    """Restore all three RNGs only when every state is available."""

    keys = ("python_random_state", "numpy_random_state", "torch_rng_state")
    if not all(key in payload for key in keys):
        return False
    random.setstate(payload["python_random_state"])
    np.random.set_state(payload["numpy_random_state"])
    torch.set_rng_state(payload["torch_rng_state"])
    return True


def load_training_checkpoint(path: str | Path, updater: A2CUpdater) -> tuple[dict[str, object], bool]:
    """Restore Actor, Critic, both Adam states, metadata, and RNG when present."""

    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    required = {
        "actor_state_dict", "critic_state_dict", "actor_optimizer_state_dict",
        "critic_optimizer_state_dict", "metadata",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"checkpoint is missing required keys: {sorted(missing)}")
    updater.actor.load_state_dict(payload["actor_state_dict"])
    updater.critic.load_state_dict(payload["critic_state_dict"])
    updater.actor_optimizer.load_state_dict(payload["actor_optimizer_state_dict"])
    updater.critic_optimizer.load_state_dict(payload["critic_optimizer_state_dict"])
    return copy.deepcopy(dict(payload["metadata"])), restore_rng_states(payload)


def save_training_checkpoint(
    path: str | Path, updater: A2CUpdater, metadata: Mapping[str, object]
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "actor_state_dict": updater.actor.state_dict(),
            "critic_state_dict": updater.critic.state_dict(),
            "actor_optimizer_state_dict": updater.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": updater.critic_optimizer.state_dict(),
            "metadata": copy.deepcopy(dict(metadata)),
            **capture_rng_states(),
        },
        destination,
    )


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: str | Path, rows: Sequence[Mapping[str, object]], columns: Sequence[str]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def window_comparison(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Return means and standard deviations for episodes 1-10, 11-20, 21-30."""

    by_episode = {int(row["episode"]): row for row in rows}
    if set(by_episode) != set(range(1, 31)):
        raise ValueError("window comparison requires exactly episodes 1 through 30")
    output: list[dict[str, object]] = []
    for label, start, end in (("episodes_1_10", 1, 10), ("episodes_11_20", 11, 20), ("episodes_21_30", 21, 30)):
        item: dict[str, object] = {"window": label, "start_episode": start, "end_episode": end}
        selected = [by_episode[episode] for episode in range(start, end + 1)]
        for metric in TREND_METRICS:
            values = np.asarray([float(row[metric]) for row in selected], dtype=np.float64)
            item[f"{metric}_mean"] = float(values.mean())
            item[f"{metric}_std"] = float(values.std())
        output.append(item)
    return output


PLOT_COLUMNS = (
    ("cumulative_scaled_reward", "cumulative_scaled_reward_1_30.png"),
    ("service_rate_by_release_horizon", "service_rate_by_release_horizon_1_30.png"),
    ("waiting_cost", "waiting_cost_1_30.png"),
    ("rebalancing_cost", "rebalancing_cost_1_30.png"),
    ("mean_waiting_time", "mean_waiting_time_1_30.png"),
)


def create_five_plots(rows: Sequence[Mapping[str, object]], output_dir: str | Path) -> None:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    episodes = np.asarray([int(row["episode"]) for row in rows])
    for metric, filename in PLOT_COLUMNS:
        values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
        figure, axis = plt.subplots(figsize=(8.0, 4.5))
        axis.plot(episodes, values, marker="o", markersize=3, label="episode value")
        axis.plot(episodes, moving_average(values, 5), linewidth=2, label="5-episode moving average")
        axis.set_xlabel("Episode")
        axis.set_ylabel(metric)
        axis.grid(alpha=0.3)
        axis.legend()
        figure.tight_layout()
        figure.savefig(directory / filename, dpi=160)
        plt.close(figure)
