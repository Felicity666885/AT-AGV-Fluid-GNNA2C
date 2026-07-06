"""Explicit mutable episode bookkeeping outside the physical FluidState."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.control.task_queue import TaskQueueLedger
from src.envs.fluid_state import FluidState
from src.training.prepared_decision import PreparedDecision


FloatArray = NDArray[np.float64]


@dataclass
class EpisodeState:
    """Physical/queue state, causal arrival history, cache, and episode flags."""

    physical_state: FluidState
    task_queue: TaskQueueLedger
    fleet_size: float
    episode_id: int | str
    decision_step: int = 0
    past_arrivals: list[FloatArray] = field(default_factory=list)
    prepared_decision: PreparedDecision | None = None
    terminated: bool = False
    truncated: bool = False
    matched_task_ids: set[int] = field(default_factory=set)

    def append_past_arrival(self, arrivals: FloatArray) -> None:
        """Append a copied completed-period arrival matrix after its transition."""

        values = np.array(arrivals, dtype=np.float64, copy=True)
        if values.shape != (FluidState.NUM_REGIONS, FluidState.NUM_REGIONS):
            raise ValueError("arrival history item must have shape (16, 16)")
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("arrival history item must be finite and nonnegative")
        self.past_arrivals.append(values)

    def record_matched_tasks(self, task_ids: tuple[int, ...]) -> None:
        """Reject any task identity selected more than once in the episode."""

        selected = set(task_ids)
        duplicate = selected & self.matched_task_ids
        if duplicate:
            raise RuntimeError(f"tasks matched more than once: {sorted(duplicate)[:5]}")
        self.matched_task_ids.update(selected)


def initialize_episode_state(
    fleet_size: float,
    initial_idle_distribution: FloatArray,
    episode_id: int | str,
) -> EpisodeState:
    """Create the required empty-queue/empty-transit initial physical state."""

    fleet = float(fleet_size)
    if not math.isfinite(fleet) or fleet < 0.0:
        raise ValueError("fleet_size must be finite and nonnegative")
    idle = np.array(initial_idle_distribution, dtype=np.float64, copy=True)
    if idle.shape != (FluidState.NUM_REGIONS,):
        raise ValueError("initial_idle_distribution must have shape (16,)")
    if not np.all(np.isfinite(idle)) or np.any(idle < 0.0):
        raise ValueError("initial idle distribution must be finite and nonnegative")
    if not math.isclose(float(idle.sum()), fleet, rel_tol=1e-12, abs_tol=1e-8):
        raise ValueError("initial idle distribution must sum to fleet_size")
    state = FluidState(
        backlog=np.zeros((16, 16), dtype=np.float64),
        idle=idle,
        loaded=np.zeros((16, 16), dtype=np.float64),
        rebalancing=np.zeros((16, 16), dtype=np.float64),
    )
    state.validate()
    return EpisodeState(state, TaskQueueLedger(), fleet, episode_id)


def all_tasks_released(episode: EpisodeState, release_horizon_steps: int) -> bool:
    """Return true after every scenario release interval has been processed."""

    return episode.decision_step >= release_horizon_steps


def task_system_empty(episode: EpisodeState, tolerance: float = 1e-10) -> bool:
    """Check identity queue and aggregate backlog without using future tasks."""

    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and nonnegative")
    episode.task_queue.validate_against_backlog(episode.physical_state.backlog, tolerance)
    return (
        episode.task_queue.total_waiting_tasks() == 0
        and float(episode.physical_state.backlog.sum()) <= tolerance
    )


def is_drain_mode(
    episode: EpisodeState,
    release_horizon_steps: int,
    tolerance: float = 1e-10,
) -> bool:
    """Enter no-policy drain after releases and task backlog are exhausted."""

    return all_tasks_released(episode, release_horizon_steps) and task_system_empty(
        episode, tolerance
    )


def natural_termination_reached(
    episode: EpisodeState,
    release_horizon_steps: int,
    transit_clear_tolerance: float,
) -> bool:
    """Check released/queue/backlog and approximate exponential transit clearance."""

    if not math.isfinite(transit_clear_tolerance) or transit_clear_tolerance < 0.0:
        raise ValueError("transit_clear_tolerance must be finite and nonnegative")
    if not is_drain_mode(episode, release_horizon_steps):
        return False
    return (
        float(episode.physical_state.loaded.sum()) <= transit_clear_tolerance
        and float(episode.physical_state.rebalancing.sum()) <= transit_clear_tolerance
    )

