"""Configuration, recovery, assessment, and reporting helpers for the final run.

This module composes the reviewed Stage-1--5 control/training interfaces.  It
does not redefine fluid dynamics, matching, rebalancing, policy, reward, or
A2C mathematics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import hashlib
import io
import json
import logging
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Iterable, Mapping, Sequence
import uuid

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.experiments.experiment_metrics import moving_average
from src.experiments.full_episode_runner import FullEpisodeConfig, FullEpisodeResult, run_full_episode
from src.experiments.stage5b_tools import load_training_checkpoint, save_training_checkpoint
from src.policy.actor import DirichletActor
from src.policy.critic import GNNCritic
from src.policy.directed_graph import IncomingNormalizedGraph
from src.policy.node_features import FeatureScaleConfig
from src.training.a2c_update import A2CLossConfig, A2CUpdater
from src.training.reward import RewardConfig


TRAINING_COLUMNS = (
    "phase", "seed", "episode", "cumulative_scaled_reward",
    "matched_by_release_horizon", "service_rate_by_release_horizon",
    "backlog_at_release_horizon", "mean_waiting_time", "p95_waiting_time",
    "waiting_cost", "rebalancing_cost", "loaded_cost",
    "total_rebalanced_vehicles", "total_episode_steps", "queue_clear_step",
    "actor_loss_mean", "critic_loss_mean", "actor_entropy_mean", "return_mean",
    "advantage_mean", "actor_grad_norm_mean", "critic_grad_norm_mean",
    "actor_adam_step_end", "critic_adam_step_end", "episode_runtime_seconds",
    "terminated", "truncated",
)

DETERMINISTIC_COLUMNS = (
    "phase", "seed", "episode", "cumulative_scaled_reward",
    "matched_by_release_horizon", "backlog_at_release_horizon",
    "service_rate_by_release_horizon", "mean_waiting_time", "median_waiting_time",
    "p90_waiting_time", "p95_waiting_time", "max_waiting_time", "waiting_cost",
    "rebalancing_cost", "loaded_cost", "total_rebalanced_vehicles",
    "total_episode_steps", "queue_clear_step", "maximum_conservation_error",
    "parameter_unchanged", "optimizer_unchanged",
)

PLATEAU_METRICS = (
    "cumulative_scaled_reward", "service_rate_by_release_horizon",
    "mean_waiting_time", "p95_waiting_time", "waiting_cost",
    "rebalancing_cost", "total_episode_steps",
)

FORMAL_METRICS = (
    "cumulative_scaled_reward", "service_rate_by_release_horizon",
    "mean_waiting_time", "median_waiting_time", "p90_waiting_time",
    "p95_waiting_time", "max_waiting_time", "waiting_cost",
    "rebalancing_cost", "loaded_cost", "total_rebalanced_vehicles",
    "total_episode_steps", "queue_clear_step", "maximum_conservation_error",
)


_T_975 = {
    1: 12.706205, 2: 4.302653, 3: 3.182446, 4: 2.776445,
    5: 2.570582, 6: 2.446912, 7: 2.364624, 8: 2.306004,
    9: 2.262157, 10: 2.228139, 11: 2.200985, 12: 2.178813,
    13: 2.160369, 14: 2.144787, 15: 2.131450, 16: 2.119905,
    17: 2.109816, 18: 2.100922, 19: 2.093024, 20: 2.085963,
    21: 2.079614, 22: 2.073873, 23: 2.068658, 24: 2.063899,
    25: 2.059539, 26: 2.055529, 27: 2.051831, 28: 2.048407,
    29: 2.045230, 30: 2.042272,
}


def student_t_975(df: int) -> float:
    """Return the two-sided 95% Student-t critical value without SciPy.

    The table covers small configurable formal-sample sizes; for larger
    samples the normal-limit value is sufficient for reporting at the
    precision used by this experiment.
    """
    if df <= 0:
        raise ValueError("Student-t degrees of freedom must be positive")
    if df in _T_975:
        return _T_975[df]
    return 1.959964


@dataclass(frozen=True)
class FinalExperimentConfig:
    """Validated config and its exact canonical hash."""

    source_path: Path
    raw: Mapping[str, Any]
    config_hash: str

    @property
    def experiment_name(self) -> str:
        return str(self.raw["experiment_name"])

    @property
    def fleet_size(self) -> int:
        return int(self.raw["fleet_size"])

    @property
    def initial_idle(self) -> np.ndarray:
        return np.asarray(self.raw["initial_idle_distribution"], dtype=np.float64)

    @property
    def feature_scales(self) -> FeatureScaleConfig:
        return FeatureScaleConfig(*map(float, self.raw["feature_scales"]))

    @property
    def reward_config(self) -> RewardConfig:
        return RewardConfig(**self.raw["reward_config"])

    @property
    def loss_config(self) -> A2CLossConfig:
        values = self.raw["a2c_config"]
        return A2CLossConfig(
            float(values["gamma"]), int(values["n_steps"]),
            float(values["entropy_coef"]), float(values["max_grad_norm"]),
        )

    @property
    def episode_config(self) -> FullEpisodeConfig:
        values = self.raw["a2c_config"]
        return FullEpisodeConfig(
            arrival_window_steps=int(values["arrival_window_steps"]),
            transit_clear_tolerance=float(values["transit_clear_tolerance"]),
            max_decision_steps=int(self.raw["max_decision_steps"]),
            max_updates=int(self.raw["max_updates"]),
        )

    @property
    def development_seeds(self) -> tuple[int, ...]:
        return tuple(map(int, self.raw["development_seeds"]))

    @property
    def formal_seeds(self) -> tuple[int, ...]:
        return tuple(map(int, self.raw["formal_seeds"]))


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_final_config(path: str | Path) -> FinalExperimentConfig:
    source = Path(path).expanduser().resolve()
    raw = json.loads(source.read_text(encoding="utf-8"))
    required = {
        "experiment_name", "scenario_path", "fleet_size", "initial_idle_distribution",
        "feature_scales", "reward_config", "a2c_config", "development_seeds",
        "development_checkpoint_paths", "training_length_candidates",
        "minimum_selected_episode", "maximum_selected_episode", "plateau_window",
        "formal_seeds", "checkpoint_interval", "max_decision_steps", "max_updates",
        "uniform_baseline",
    }
    missing = required - set(raw)
    if missing:
        raise ValueError(f"final config missing keys: {sorted(missing)}")
    if len(raw["initial_idle_distribution"]) != 16 or len(raw["feature_scales"]) != 6:
        raise ValueError("current final pipeline requires 16 idle entries and six feature scales")
    if sum(map(float, raw["initial_idle_distribution"])) != float(raw["fleet_size"]):
        raise ValueError("initial_idle_distribution must sum to fleet_size")
    candidates = list(map(int, raw["training_length_candidates"]))
    if candidates != sorted(set(candidates)) or not candidates:
        raise ValueError("training_length_candidates must be sorted unique integers")
    minimum, maximum = int(raw["minimum_selected_episode"]), int(raw["maximum_selected_episode"])
    if minimum < 1 or maximum < minimum or max(candidates) != maximum:
        raise ValueError("selected-episode bounds and candidates are inconsistent")
    if int(raw["plateau_window"]) <= 0 or int(raw["checkpoint_interval"]) <= 0:
        raise ValueError("plateau_window and checkpoint_interval must be positive")
    if int(raw.get("workers", 1)) != 1:
        raise ValueError("this CPU/SCIP-safe implementation currently requires workers=1")
    if set(map(int, raw["development_seeds"])) & set(map(int, raw["formal_seeds"])):
        raise ValueError("development and formal seeds must be disjoint")
    config = FinalExperimentConfig(source, raw, sha256_bytes(canonical_json_bytes(raw)))
    # Construct typed configs now so invalid coefficients fail before any output.
    config.feature_scales
    config.reward_config
    config.loss_config
    config.episode_config
    return config


def resolve_existing_path(project_root: Path, configured: str | Path) -> Path:
    candidate = Path(configured).expanduser()
    possibilities = [candidate] if candidate.is_absolute() else [
        project_root / candidate,
        project_root.parent / candidate,
        project_root / "configs" / candidate,
    ]
    existing = []
    for item in possibilities:
        resolved = item.resolve()
        if resolved.exists() and resolved not in existing:
            existing.append(resolved)
    if len(existing) == 1:
        return existing[0]
    if not existing:
        raise FileNotFoundError(f"configured path does not exist: {configured}")
    raise RuntimeError("configured path is ambiguous:\n" + "\n".join(map(str, existing)))


def resolve_development_checkpoint(project_root: Path, config: FinalExperimentConfig, seed: int) -> Path:
    configured = config.raw["development_checkpoint_paths"].get(str(seed))
    if configured:
        try:
            return resolve_existing_path(project_root, configured)
        except FileNotFoundError:
            pass
    search_root = project_root / "artifacts"
    candidates = sorted(
        path.resolve()
        for path in search_root.rglob("episode_50.pt")
        if f"seed_{seed}" in str(path.parent)
    )
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"no Episode-50 checkpoint found for development seed {seed}")
    raise RuntimeError(
        f"multiple Episode-50 checkpoints found for seed {seed}; refusing to guess:\n"
        + "\n".join(map(str, candidates))
    )


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Serialize JSON once and atomically publish with Windows lock retries."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f"{destination.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    )
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    maximum_attempts = 10
    initial_delay_seconds = 0.1
    for attempt in range(1, maximum_attempts + 1):
        try:
            os.replace(temporary, destination)
            return
        except PermissionError as exc:
            if attempt == maximum_attempts:
                raise PermissionError(
                    "JSON atomic replace failed after "
                    f"{maximum_attempts} attempts; complete temporary JSON "
                    f"preserved at '{temporary}'; destination remains '{destination}'"
                ) from exc
            delay = min(initial_delay_seconds * (2 ** (attempt - 1)), 2.0)
            print(
                f"json replace locked (attempt {attempt}/{maximum_attempts}); "
                "retrying...",
                flush=True,
            )
            time.sleep(delay)


def read_csv(path: str | Path) -> list[dict[str, str]]:
    source = Path(path)
    if not source.exists():
        return []
    with source.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def atomic_write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, destination)


def upsert_rows(
    path: str | Path,
    new_rows: Iterable[Mapping[str, Any]],
    columns: Sequence[str],
    key_fields: Sequence[str],
) -> list[dict[str, Any]]:
    combined: dict[tuple[str, ...], dict[str, Any]] = {
        tuple(str(row[field]) for field in key_fields): dict(row) for row in read_csv(path)
    }
    for row in new_rows:
        combined[tuple(str(row[field]) for field in key_fields)] = dict(row)
    def sort_value(value: Any) -> tuple[int, Any]:
        try:
            return (0, int(value))
        except (TypeError, ValueError):
            return (1, str(value))
    rows = sorted(
        combined.values(),
        key=lambda row: tuple(sort_value(row[field]) for field in key_fields),
    )
    atomic_write_csv(path, rows, columns)
    return rows


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_training_objects(config: FinalExperimentConfig) -> tuple[DirichletActor, GNNCritic, A2CUpdater]:
    # Network shape is the reviewed Stage-3 architecture; training coefficients
    # and optimizer learning rates come only from the config.
    actor = DirichletActor(input_dim=6, hidden_dim=64, embedding_dim=64, num_layers=2)
    critic = GNNCritic(input_dim=6, hidden_dim=64, embedding_dim=64, num_layers=2, value_hidden_dim=64)
    values = config.raw["a2c_config"]
    updater = A2CUpdater(
        actor, critic, config.loss_config,
        float(values["actor_lr"]), float(values["critic_lr"]),
    )
    return actor, critic, updater


def optimizer_steps(updater: A2CUpdater) -> tuple[float, float]:
    def maximum(optimizer: torch.optim.Optimizer) -> float:
        values = [float(state["step"].detach().cpu()) for state in optimizer.state.values() if "step" in state]
        return max(values) if values else 0.0
    return maximum(updater.actor_optimizer), maximum(updater.critic_optimizer)


def module_parameters_finite(updater: A2CUpdater) -> bool:
    return all(
        torch.isfinite(parameter).all()
        for module in (updater.actor, updater.critic)
        for parameter in module.parameters()
    )


def state_digest(updater: A2CUpdater) -> bytes:
    buffer = io.BytesIO()
    torch.save({
        "actor": updater.actor.state_dict(), "critic": updater.critic.state_dict(),
        "actor_optimizer": updater.actor_optimizer.state_dict(),
        "critic_optimizer": updater.critic_optimizer.state_dict(),
    }, buffer)
    return buffer.getvalue()


def validate_checkpoint(
    path: str | Path,
    updater: A2CUpdater,
    config: FinalExperimentConfig,
    expected_seed: int,
    expected_episode: int | None,
    expected_manifest_hash: str | None = None,
) -> dict[str, Any]:
    metadata, rng_restored = load_training_checkpoint(path, updater)
    if not rng_restored:
        raise ValueError(f"checkpoint lacks complete Python/NumPy/PyTorch RNG state: {path}")
    if int(metadata.get("seed", -1)) != expected_seed:
        raise ValueError(f"checkpoint seed mismatch: expected {expected_seed}, got {metadata.get('seed')}")
    if expected_episode is not None and int(metadata.get("episode_number", -1)) != expected_episode:
        raise ValueError("checkpoint episode mismatch")
    if optimizer_steps(updater)[0] <= 0 or optimizer_steps(updater)[1] <= 0:
        raise ValueError("checkpoint Adam step must be positive")
    if not module_parameters_finite(updater):
        raise ValueError("checkpoint network contains NaN or Inf")
    expected = {
        "K": config.fleet_size,
        "initial_idle_distribution": config.initial_idle.tolist(),
        "FeatureScaleConfig": asdict(config.feature_scales),
        "RewardConfig": asdict(config.reward_config),
        "A2CConfig": asdict(config.loss_config),
        "actor_lr": float(config.raw["a2c_config"]["actor_lr"]),
        "critic_lr": float(config.raw["a2c_config"]["critic_lr"]),
        "arrival_window_steps": int(config.raw["a2c_config"]["arrival_window_steps"]),
        "transit_clear_tolerance": float(config.raw["a2c_config"]["transit_clear_tolerance"]),
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"checkpoint configuration mismatch for {key}")
    if expected_manifest_hash is not None and metadata.get("manifest_hash") != expected_manifest_hash:
        raise ValueError("formal checkpoint manifest hash mismatch")
    return metadata


def checkpoint_metadata(
    config: FinalExperimentConfig,
    seed: int,
    episode: int,
    elapsed_seconds: float,
    manifest_hash: str | None,
    phase: str,
) -> dict[str, Any]:
    return {
        "phase": phase,
        "seed": seed,
        "episode_number": episode,
        "manifest_hash": manifest_hash,
        "config_hash": config.config_hash,
        "K": config.fleet_size,
        "initial_idle_distribution": config.initial_idle.tolist(),
        "FeatureScaleConfig": asdict(config.feature_scales),
        "RewardConfig": asdict(config.reward_config),
        "A2CConfig": asdict(config.loss_config),
        "actor_lr": float(config.raw["a2c_config"]["actor_lr"]),
        "critic_lr": float(config.raw["a2c_config"]["critic_lr"]),
        "arrival_window_steps": int(config.raw["a2c_config"]["arrival_window_steps"]),
        "transit_clear_tolerance": float(config.raw["a2c_config"]["transit_clear_tolerance"]),
        "scenario_file": Path(config.raw["scenario_path"]).name,
        "cumulative_runtime_seconds": float(elapsed_seconds),
    }


def atomic_save_checkpoint(
    path: str | Path,
    updater: A2CUpdater,
    metadata: Mapping[str, Any],
) -> None:
    """Serialize once, then atomically publish with bounded Windows lock retries.

    A unique temporary file prevents concurrent or interrupted saves from
    sharing a name.  ``PermissionError`` retries only ``os.replace``; the
    checkpoint is never serialized twice and the existing destination is
    never deleted.  After the final failed attempt, the complete temporary
    checkpoint remains available for manual recovery.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f"{destination.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    )
    save_training_checkpoint(temporary, updater, metadata)
    maximum_attempts = 10
    initial_delay_seconds = 0.1
    for attempt in range(1, maximum_attempts + 1):
        try:
            os.replace(temporary, destination)
            return
        except PermissionError as exc:
            if attempt == maximum_attempts:
                raise PermissionError(
                    "checkpoint atomic replace failed after "
                    f"{maximum_attempts} attempts; complete temporary checkpoint "
                    f"preserved at '{temporary}'; destination remains '{destination}'"
                ) from exc
            delay = min(initial_delay_seconds * (2 ** (attempt - 1)), 2.0)
            logging.getLogger(__name__).warning(
                "checkpoint replace locked (attempt %d/%d); retrying in %.1fs: %s -> %s",
                attempt,
                maximum_attempts,
                delay,
                temporary,
                destination,
            )
            time.sleep(delay)


def training_row(result: FullEpisodeResult, phase: str, seed: int, steps: tuple[float, float]) -> dict[str, Any]:
    summary = result.summary
    return {
        "phase": phase, "seed": seed, "episode": result.episode_number,
        "cumulative_scaled_reward": summary["scaled_reward"],
        "matched_by_release_horizon": summary["matched_by_release_horizon"],
        "service_rate_by_release_horizon": summary["service_rate_by_release_horizon"],
        "backlog_at_release_horizon": summary["backlog_at_release_horizon"],
        "mean_waiting_time": summary["mean_waiting_time"],
        "p95_waiting_time": summary["p95_waiting_time"],
        "waiting_cost": summary["waiting_cost"],
        "rebalancing_cost": summary["rebalancing_travel_cost"],
        "loaded_cost": summary["loaded_travel_cost"],
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
        "actor_adam_step_end": steps[0], "critic_adam_step_end": steps[1],
        "episode_runtime_seconds": summary["episode_runtime_seconds"],
        "terminated": summary["terminated"], "truncated": summary["truncated"],
    }


def validate_episode(result: FullEpisodeResult, task_count: int, before: tuple[float, float], after: tuple[float, float]) -> None:
    summary = result.summary
    if result.status != "completed" or not result.episode.terminated or result.episode.truncated:
        raise RuntimeError("episode did not naturally terminate")
    if len(result.episode.matched_task_ids) != task_count:
        raise RuntimeError("episode did not uniquely match every scenario task")
    if result.episode.task_queue.total_waiting_tasks() != 0 or float(result.episode.physical_state.backlog.sum()) > 1e-10:
        raise RuntimeError("episode ended with nonempty task queue/backlog")
    if float(summary["max_vehicle_conservation_error"]) > 1e-8:
        raise RuntimeError("vehicle conservation error exceeded tolerance")
    if float(summary["max_queue_backlog_difference"]) != 0.0:
        raise RuntimeError("queue and backlog diverged")
    if bool(summary["nonfinite_detected"]) or after[0] <= before[0] or after[1] <= before[1]:
        raise RuntimeError("nonfinite diagnostic or non-increasing Adam step")


def deterministic_evaluate(
    phase: str,
    seed: int,
    episode: int,
    updater: A2CUpdater,
    scenario: Any,
    graph: IncomingNormalizedGraph,
    config: FinalExperimentConfig,
) -> dict[str, Any]:
    before = state_digest(updater)
    result = run_full_episode(
        scenario=scenario, graph=graph, feature_scaler=config.feature_scales,
        reward_config=config.reward_config, initial_idle_distribution=config.initial_idle,
        fleet_size=config.fleet_size, config=config.episode_config,
        run_type="deterministic_evaluation", episode_number=episode,
        actor=updater.actor, critic=updater.critic,
    )
    after = state_digest(updater)
    if before != after:
        raise RuntimeError("deterministic evaluation changed network or Adam state")
    summary = result.summary
    if result.status != "completed" or not result.episode.terminated or result.episode.truncated:
        raise RuntimeError("deterministic evaluation did not naturally terminate")
    if len(result.episode.matched_task_ids) != scenario.task_count:
        raise RuntimeError("deterministic evaluation did not uniquely match all tasks")
    if result.episode.task_queue.total_waiting_tasks() or float(result.episode.physical_state.backlog.sum()) > 1e-10:
        raise RuntimeError("deterministic evaluation ended with nonempty queue/backlog")
    if float(summary["max_vehicle_conservation_error"]) > 1e-8 or float(summary["max_queue_backlog_difference"]) != 0.0:
        raise RuntimeError("deterministic evaluation violated conservation or queue synchronization")
    if bool(summary["nonfinite_detected"]):
        raise RuntimeError("deterministic evaluation produced a nonfinite diagnostic")
    return {
        "phase": phase, "seed": seed, "episode": episode,
        "cumulative_scaled_reward": summary["scaled_reward"],
        "matched_by_release_horizon": summary["matched_by_release_horizon"],
        "backlog_at_release_horizon": summary["backlog_at_release_horizon"],
        "service_rate_by_release_horizon": summary["service_rate_by_release_horizon"],
        "mean_waiting_time": summary["mean_waiting_time"],
        "median_waiting_time": summary["median_waiting_time"],
        "p90_waiting_time": summary["p90_waiting_time"],
        "p95_waiting_time": summary["p95_waiting_time"],
        "max_waiting_time": summary["max_waiting_time"],
        "waiting_cost": summary["waiting_cost"],
        "rebalancing_cost": summary["rebalancing_travel_cost"],
        "loaded_cost": summary["loaded_travel_cost"],
        "total_rebalanced_vehicles": summary["total_rebalanced_vehicles"],
        "total_episode_steps": summary["total_episode_steps"],
        "queue_clear_step": summary["queue_clear_step"],
        "maximum_conservation_error": summary["max_vehicle_conservation_error"],
        "parameter_unchanged": True, "optimizer_unchanged": True,
    }


def _means(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    return {
        metric: float(np.mean([float(row[metric]) for row in rows]))
        for metric in PLATEAU_METRICS
    }


def metric_change(old: float, new: float, higher_better: bool) -> float:
    if higher_better:
        return new - old
    return old - new


def assess_plateau(
    training_rows: Sequence[Mapping[str, Any]],
    previous_deterministic: Mapping[str, Any],
    current_deterministic: Mapping[str, Any],
    candidate: int,
    window: int,
) -> dict[str, Any]:
    """Apply the exact two-window and deterministic plateau thresholds."""

    by_episode = {int(row["episode"]): row for row in training_rows}
    previous_range = range(candidate - 2 * window + 1, candidate - window + 1)
    recent_range = range(candidate - window + 1, candidate + 1)
    if not all(episode in by_episode for episode in (*previous_range, *recent_range)):
        raise ValueError("insufficient training rows for plateau windows")
    old, new = _means([by_episode[ep] for ep in previous_range]), _means([by_episode[ep] for ep in recent_range])
    training_thresholds = {
        "cumulative_scaled_reward": max(1.0, abs(old["cumulative_scaled_reward"]) * 0.02),
        "service_rate_by_release_horizon": 0.002,
        "mean_waiting_time": old["mean_waiting_time"] * 0.02,
        "p95_waiting_time": old["p95_waiting_time"] * 0.02,
        "waiting_cost": old["waiting_cost"] * 0.02,
        "rebalancing_cost": old["rebalancing_cost"] * 0.02,
        "total_episode_steps": old["total_episode_steps"] * 0.02,
    }
    higher = {"cumulative_scaled_reward", "service_rate_by_release_horizon"}
    training_improved, training_degraded = [], []
    for metric, threshold in training_thresholds.items():
        change = metric_change(old[metric], new[metric], metric in higher)
        if change >= threshold:
            training_improved.append(metric)
        elif change <= -threshold:
            training_degraded.append(metric)

    deterministic_metrics = (
        "cumulative_scaled_reward", "service_rate_by_release_horizon", "mean_waiting_time",
        "p95_waiting_time", "waiting_cost", "total_episode_steps",
    )
    deterministic_thresholds = {
        "cumulative_scaled_reward": max(1.0, abs(float(previous_deterministic["cumulative_scaled_reward"])) * 0.03),
        "service_rate_by_release_horizon": 0.003,
        "mean_waiting_time": float(previous_deterministic["mean_waiting_time"]) * 0.03,
        "p95_waiting_time": float(previous_deterministic["p95_waiting_time"]) * 0.03,
        "waiting_cost": float(previous_deterministic["waiting_cost"]) * 0.03,
        "total_episode_steps": float(previous_deterministic["total_episode_steps"]) * 0.03,
    }
    deterministic_improved, deterministic_degraded = [], []
    for metric in deterministic_metrics:
        change = metric_change(
            float(previous_deterministic[metric]), float(current_deterministic[metric]), metric in higher
        )
        threshold = deterministic_thresholds[metric]
        if change >= threshold:
            deterministic_improved.append(metric)
        elif change <= -threshold:
            deterministic_degraded.append(metric)
    degraded_business = set(training_degraded) | set(deterministic_degraded)
    plateau = (
        len(training_improved) <= 1
        and len(deterministic_improved) <= 1
        and len(degraded_business) < 3
    )
    return {
        "candidate_episode": candidate,
        "training_previous_start": candidate - 2 * window + 1,
        "training_previous_end": candidate - window,
        "training_recent_start": candidate - window + 1,
        "training_recent_end": candidate,
        "training_improvement_count": len(training_improved),
        "training_improved_metrics": ";".join(training_improved),
        "training_degradation_count": len(training_degraded),
        "training_degraded_metrics": ";".join(training_degraded),
        "deterministic_previous_episode": int(previous_deterministic["episode"]),
        "deterministic_improvement_count": len(deterministic_improved),
        "deterministic_improved_metrics": ";".join(deterministic_improved),
        "deterministic_degradation_count": len(deterministic_degraded),
        "deterministic_degraded_metrics": ";".join(deterministic_degraded),
        "combined_degraded_business_count": len(degraded_business),
        "plateau_signal": plateau,
    }


def initial_pipeline_state(config: FinalExperimentConfig) -> dict[str, Any]:
    return {
        "version": 1,
        "experiment_name": config.experiment_name,
        "config_hash": config.config_hash,
        "current_phase": "development",
        "selected_N": None,
        "training_length_status": None,
        "development": {
            str(seed): {
                "completed_episode": 50,
                "latest_checkpoint": None,
                "completed_evaluations": [50],
                "plateau_streak": 0,
                "status": "pending",
            }
            for seed in config.development_seeds
        },
        "formal": {
            str(seed): {
                "completed_episode": 0,
                "latest_checkpoint": None,
                "evaluation_completed": False,
                "status": "pending",
            }
            for seed in config.formal_seeds
        },
        "manifest_hash": None,
        "failure": None,
    }


def load_or_create_pipeline_state(path: Path, config: FinalExperimentConfig, resume: bool) -> dict[str, Any]:
    if path.exists():
        if not resume:
            raise RuntimeError("pipeline_state.json already exists; rerun with --resume")
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("config_hash") != config.config_hash:
            raise RuntimeError("config changed after pipeline initialization")
        return state
    state = initial_pipeline_state(config)
    atomic_write_json(path, state)
    return state


def build_manifest(
    project_root: Path,
    scenario_path: Path,
    config: FinalExperimentConfig,
    selected_n: int,
    selection_method: str,
) -> dict[str, Any]:
    code_files = (
        "src/data/scenario_loader.py", "src/envs/fluid_state.py", "src/envs/fluid_dynamics.py",
        "src/control/task_queue.py", "src/control/matching_solver.py",
        "src/control/target_rounding.py", "src/control/rebalancing_solver.py",
        "src/control/fluid_control_step.py", "src/policy/node_features.py",
        "src/policy/directed_graph.py", "src/policy/directed_gnn.py", "src/policy/actor.py",
        "src/policy/critic.py", "src/policy/policy_forward.py", "src/training/reward.py",
        "src/training/prepared_decision.py", "src/training/rollout_buffer.py",
        "src/training/a2c_update.py", "src/training/episode_state.py",
        "src/experiments/full_episode_runner.py", "src/experiments/final_experiment.py",
        "scripts/run_final_experiment.py",
    )
    hashes = {name: sha256_file(project_root / name) for name in code_files}
    content: dict[str, Any] = {
        "experiment_name": config.experiment_name,
        "config_hash": config.config_hash,
        "scenario_path": str(scenario_path),
        "scenario_sha256": sha256_file(scenario_path),
        "selected_N": selected_n,
        "selected_N_method": selection_method,
        "fleet_size": config.fleet_size,
        "initial_idle_distribution": config.initial_idle.tolist(),
        "feature_scales": config.feature_scales.as_array().tolist(),
        "reward_config": asdict(config.reward_config),
        "a2c_config": dict(config.raw["a2c_config"]),
        "network_structure": {
            "input_dim": 6, "hidden_dim": 64, "embedding_dim": 64,
            "num_layers": 2, "critic_value_hidden_dim": 64,
            "actor_distribution": "Dirichlet.sample training / mean evaluation",
        },
        "formal_seeds": list(config.formal_seeds),
        "checkpoint_selection_rule": "final_episode_N only; no best-checkpoint selection",
        "deterministic_evaluation_rule": "Dirichlet mean, no_grad, final_episode_N",
        "uniform_baseline": dict(config.raw["uniform_baseline"]),
        "code_sha256": hashes,
    }
    content["manifest_hash"] = sha256_bytes(canonical_json_bytes(content))
    return content


def verify_manifest(path: Path, expected_hash: str | None = None) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    recorded = manifest.pop("manifest_hash")
    actual = sha256_bytes(canonical_json_bytes(manifest))
    manifest["manifest_hash"] = recorded
    if recorded != actual or (expected_hash is not None and recorded != expected_hash):
        raise RuntimeError("frozen manifest hash verification failed")
    return manifest


def formal_statistics(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if len(rows) < 2:
        raise ValueError("formal statistics require at least two completed seeds")
    outputs: list[dict[str, Any]] = []
    n = len(rows)
    critical = student_t_975(n - 1)
    for metric in FORMAL_METRICS:
        values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
        mean = float(values.mean())
        sample_std = float(values.std(ddof=1))
        half_width = critical * sample_std / math.sqrt(n)
        outputs.append({
            "metric": metric, "n": n, "mean": mean,
            "sample_standard_deviation": sample_std,
            "median": float(np.median(values)), "minimum": float(values.min()),
            "maximum": float(values.max()), "ci95_low": mean - half_width,
            "ci95_high": mean + half_width,
        })
    return outputs


def baseline_comparison(statistics: Sequence[Mapping[str, Any]], baseline: Mapping[str, Any]) -> dict[str, Any]:
    means = {row["metric"]: float(row["mean"]) for row in statistics}
    return {
        "reward_difference": means["cumulative_scaled_reward"] - float(baseline["cumulative_scaled_reward"]),
        "service_rate_percentage_point_change": 100.0 * (
            means["service_rate_by_release_horizon"] - float(baseline["service_rate_by_release_horizon"])
        ),
        "mean_waiting_improvement_percent": 100.0 * (
            float(baseline["mean_waiting_time"]) - means["mean_waiting_time"]
        ) / float(baseline["mean_waiting_time"]),
        "waiting_cost_improvement_percent": 100.0 * (
            float(baseline["waiting_cost"]) - means["waiting_cost"]
        ) / float(baseline["waiting_cost"]),
        "rebalancing_cost_change_percent": 100.0 * (
            means["rebalancing_cost"] - float(baseline["rebalancing_cost"])
        ) / float(baseline["rebalancing_cost"]),
        "total_steps_improvement_percent": 100.0 * (
            float(baseline["total_episode_steps"]) - means["total_episode_steps"]
        ) / float(baseline["total_episode_steps"]),
    }


def generate_final_plots(
    output_dir: Path,
    development_rows: Sequence[Mapping[str, Any]],
    formal_training_rows: Sequence[Mapping[str, Any]],
    formal_eval_rows: Sequence[Mapping[str, Any]],
    baseline: Mapping[str, Any],
    selected_n: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    for seed in sorted({int(row["seed"]) for row in development_rows}):
        rows = sorted((row for row in development_rows if int(row["seed"]) == seed), key=lambda row: int(row["episode"]))
        episodes = [int(row["episode"]) for row in rows]
        for axis, metric in zip(axes, ("cumulative_scaled_reward", "service_rate_by_release_horizon", "mean_waiting_time")):
            axis.plot(episodes, moving_average([float(row[metric]) for row in rows], 20), label=str(seed))
            axis.set_ylabel(metric)
            axis.grid(alpha=0.25)
    axes[-1].set_xlabel("Episode")
    axes[0].legend()
    figure.tight_layout()
    figure.savefig(output_dir / "development_convergence_selected_N.png", dpi=160)
    plt.close(figure)

    for metric, filename in (
        ("cumulative_scaled_reward", "formal_reward_mean_std.png"),
        ("mean_waiting_time", "formal_mean_waiting_mean_std.png"),
    ):
        by_episode: dict[int, list[float]] = {}
        for row in formal_training_rows:
            by_episode.setdefault(int(row["episode"]), []).append(float(row[metric]))
        episodes = np.asarray(sorted(by_episode))
        means = np.asarray([np.mean(by_episode[ep]) for ep in episodes])
        stds = np.asarray([np.std(by_episode[ep], ddof=1) for ep in episodes])
        figure, axis = plt.subplots(figsize=(8, 4.5))
        axis.plot(episodes, means, label="5-seed mean")
        axis.fill_between(episodes, means - stds, means + stds, alpha=0.25, label="±1 sample std")
        axis.set_xlabel("Episode")
        axis.set_ylabel(metric)
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_dir / filename, dpi=160)
        plt.close(figure)

    metrics = (
        ("cumulative_scaled_reward", "cumulative_scaled_reward"),
        ("service_rate_by_release_horizon", "service_rate_by_release_horizon"),
        ("mean_waiting_time", "mean_waiting_time"),
        ("waiting_cost", "waiting_cost"),
        ("rebalancing_cost", "rebalancing_cost"),
        ("total_episode_steps", "total_episode_steps"),
    )
    figure, axes = plt.subplots(2, 3, figsize=(13, 7))
    for axis, (metric, baseline_key) in zip(axes.flat, metrics):
        values = [float(row[metric]) for row in formal_eval_rows]
        axis.bar(["formal mean", "uniform"], [np.mean(values), float(baseline[baseline_key])])
        axis.set_title(metric, fontsize=9)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle(f"Deterministic final_episode_{selected_n} vs uniform")
    figure.tight_layout()
    figure.savefig(output_dir / "formal_deterministic_vs_uniform.png", dpi=160)
    plt.close(figure)
