"""Pure Stage-5A metric aggregation helpers."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np

from src.data.scenario_loader import TaskRecord
from src.policy.node_features import NODE_FEATURE_NAMES, FeatureScaleConfig
from src.training.reward import RewardBreakdown


@dataclass(frozen=True)
class TaskWaitingRecord:
    task_id: int
    release_minute: float
    origin: int
    destination: int
    matching_step: int
    matching_time_minute: float
    waiting_time_minute: float
    run_type: str
    episode: int


def build_waiting_records(
    selected_task_ids: Sequence[int],
    task_lookup: dict[int, TaskRecord],
    matching_step: int,
    delta_t: float,
    run_type: str,
    episode: int,
) -> list[TaskWaitingRecord]:
    """Build unique task waiting records using decision time minus LIFTTIME."""

    if len(set(selected_task_ids)) != len(selected_task_ids):
        raise ValueError("selected_task_ids contains duplicates")
    matching_minute = float(matching_step) * float(delta_t)
    result: list[TaskWaitingRecord] = []
    for task_id in selected_task_ids:
        task = task_lookup[int(task_id)]
        waiting = matching_minute - task.release_minute
        if waiting < -1e-10:
            raise RuntimeError("a task was matched before its release time")
        result.append(
            TaskWaitingRecord(
                task_id=task.task_id,
                release_minute=task.release_minute,
                origin=task.origin,
                destination=task.destination,
                matching_step=int(matching_step),
                matching_time_minute=matching_minute,
                waiting_time_minute=max(0.0, waiting),
                run_type=str(run_type),
                episode=int(episode),
            )
        )
    return result


class FeatureAccumulator:
    """Collect copied raw and normalized 16-by-6 policy observations."""

    def __init__(self, scales: FeatureScaleConfig) -> None:
        self.scales = scales
        self._raw: list[np.ndarray] = []
        self._normalized: list[np.ndarray] = []

    def add(self, raw: np.ndarray, normalized: np.ndarray) -> None:
        raw_copy = np.array(raw, dtype=np.float64, copy=True)
        norm_copy = np.array(normalized, dtype=np.float64, copy=True)
        if raw_copy.shape != (16, 6) or norm_copy.shape != (16, 6):
            raise ValueError("feature samples must both have shape (16, 6)")
        if not np.all(np.isfinite(raw_copy)) or not np.all(np.isfinite(norm_copy)):
            raise ValueError("feature sample is nonfinite")
        self._raw.append(raw_copy)
        self._normalized.append(norm_copy)

    def rows(self, run_type: str, episode: int) -> list[dict[str, object]]:
        if not self._raw:
            return []
        raw = np.concatenate(self._raw, axis=0)
        normalized = np.concatenate(self._normalized, axis=0)
        rows: list[dict[str, object]] = []
        for index, name in enumerate(NODE_FEATURE_NAMES):
            values = raw[:, index]
            normalized_values = normalized[:, index]
            normalized_p95 = float(np.percentile(normalized_values, 95))
            normalized_maximum = float(np.max(normalized_values))
            if normalized_p95 < 0.01:
                diagnostic = "scale_may_be_too_large"
            elif normalized_maximum > 5.0:
                diagnostic = "scale_may_be_too_small_or_extreme_value"
            else:
                diagnostic = "numerically_usable_in_this_pretest"
            rows.append(
                {
                    "run_type": run_type,
                    "episode": episode,
                    "feature": name,
                    "scale": float(self.scales.as_array()[index]),
                    "count": int(values.size),
                    "mean": float(np.mean(values)),
                    "standard_deviation": float(np.std(values)),
                    "median": float(np.median(values)),
                    "p90": float(np.percentile(values, 90)),
                    "p95": float(np.percentile(values, 95)),
                    "p99": float(np.percentile(values, 99)),
                    "maximum": float(np.max(values)),
                    "normalized_mean": float(np.mean(normalized_values)),
                    "normalized_p95": normalized_p95,
                    "normalized_maximum": normalized_maximum,
                    "scale_diagnostic": diagnostic,
                }
            )
        return rows


def aggregate_reward(
    rewards: Iterable[RewardBreakdown], run_type: str, episode: int
) -> dict[str, object]:
    """Aggregate period reward components and report absolute component shares."""

    items = tuple(rewards)
    totals = {
        "service_component": sum(item.service_component for item in items),
        "loaded_travel_cost": sum(item.loaded_travel_cost for item in items),
        "rebalancing_travel_cost": sum(item.rebalancing_travel_cost for item in items),
        "waiting_cost": sum(item.waiting_cost for item in items),
        "unscaled_reward": sum(item.unscaled_reward for item in items),
        "scaled_reward": sum(item.scaled_reward for item in items),
    }
    denominator = sum(abs(totals[key]) for key in (
        "service_component", "loaded_travel_cost", "rebalancing_travel_cost", "waiting_cost"
    ))
    shares = {
        f"{key}_absolute_share": (abs(totals[key]) / denominator if denominator else 0.0)
        for key in ("service_component", "loaded_travel_cost", "rebalancing_travel_cost", "waiting_cost")
    }
    dominant = [key for key, value in shares.items() if value > 0.9]
    row: dict[str, object] = {"run_type": run_type, "episode": episode, **totals, **shares}
    row["reward_component_dominance"] = ";".join(dominant)
    if not all(math.isfinite(float(value)) for value in totals.values()):
        raise RuntimeError("aggregated reward is nonfinite")
    return row


def waiting_summary(records: Sequence[TaskWaitingRecord]) -> dict[str, float]:
    values = np.asarray([record.waiting_time_minute for record in records], dtype=np.float64)
    if values.size == 0:
        return {key: math.nan for key in ("mean_waiting_time", "median_waiting_time", "p90_waiting_time", "p95_waiting_time", "max_waiting_time")}
    return {
        "mean_waiting_time": float(np.mean(values)),
        "median_waiting_time": float(np.median(values)),
        "p90_waiting_time": float(np.percentile(values, 90)),
        "p95_waiting_time": float(np.percentile(values, 95)),
        "max_waiting_time": float(np.max(values)),
    }


def moving_average(values: Sequence[float], window: int = 3) -> np.ndarray:
    """Return a trailing moving average with a shorter initial window."""

    array = np.asarray(values, dtype=np.float64)
    if window <= 0:
        raise ValueError("window must be positive")
    return np.asarray([np.mean(array[max(0, i - window + 1) : i + 1]) for i in range(len(array))])
