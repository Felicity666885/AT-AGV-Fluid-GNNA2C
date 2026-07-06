"""Deterministic per-origin FCFS task matching with integer dispatch limits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.control.task_queue import TaskQueueLedger
from src.envs.fluid_state import FluidState


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
DEFAULT_INTEGER_TOLERANCE = 1e-10
MAX_INTEGER_TOLERANCE = 1e-6


@dataclass(frozen=True)
class OriginMatchingDiagnostics:
    """FCFS capacity and result for one origin."""

    origin: int
    waiting_before: int
    dispatchable_idle: int
    matched: int
    unmatched: int
    destination_counts: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class MatchingPlan:
    """Integer task matching quantities and selected task identities."""

    matching: IntArray
    selected_task_ids: tuple[int, ...]
    matched_task_count: int
    dispatchable_idle_before: IntArray
    unmatched_task_count: int
    per_origin_diagnostics: tuple[OriginMatchingDiagnostics, ...]


def dispatchable_integer_idle(
    idle: FloatArray, tolerance: float = DEFAULT_INTEGER_TOLERANCE
) -> IntArray:
    """Return ``floor(idle_i + tolerance)`` without rounding up fluid remnants."""

    values = np.array(idle, dtype=np.float64, copy=True)
    if values.shape != (FluidState.NUM_REGIONS,):
        raise ValueError(f"idle must have shape ({FluidState.NUM_REGIONS},), got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("idle must contain only finite values")
    if not np.isfinite(tolerance) or not 0.0 <= tolerance <= MAX_INTEGER_TOLERANCE:
        raise ValueError(f"tolerance must be between 0 and {MAX_INTEGER_TOLERANCE}")
    if np.any(values < -tolerance):
        raise ValueError("idle contains a materially negative value")
    values[(values < 0.0) & (values >= -tolerance)] = 0.0
    dispatchable = np.floor(values + tolerance).astype(np.int64)
    if np.any(dispatchable.astype(np.float64) > values + tolerance):
        raise RuntimeError("integer idle conversion created vehicle mass")
    return dispatchable


def solve_fcfs_matching(
    state: FluidState,
    task_queue: TaskQueueLedger,
    tolerance: float = DEFAULT_INTEGER_TOLERANCE,
) -> MatchingPlan:
    """Select the earliest waiting tasks independently at each origin.

    The decision matrix ``M`` is an integer task/vehicle quantity for one
    decision period.  The function does not add current-period arrivals, mutate
    inputs, invoke fluid dynamics, or move vehicles between origins.
    """

    state.validate(tolerance)
    task_queue.validate_against_backlog(state.backlog, tolerance)
    available = dispatchable_integer_idle(state.idle, tolerance)
    matching = np.zeros((FluidState.NUM_REGIONS, FluidState.NUM_REGIONS), dtype=np.int64)
    selected_ids: list[int] = []
    diagnostics: list[OriginMatchingDiagnostics] = []

    for origin in range(FluidState.NUM_REGIONS):
        waiting = task_queue.tasks_at_origin(origin)
        selected = waiting[: int(available[origin])]
        destination_counts: dict[int, int] = {}
        for task in selected:
            matching[origin, task.destination] += 1
            destination_counts[task.destination] = destination_counts.get(task.destination, 0) + 1
            selected_ids.append(task.task_id)
        diagnostics.append(
            OriginMatchingDiagnostics(
                origin=origin,
                waiting_before=len(waiting),
                dispatchable_idle=int(available[origin]),
                matched=len(selected),
                unmatched=len(waiting) - len(selected),
                destination_counts=tuple(sorted(destination_counts.items())),
            )
        )

    if np.any(np.diag(matching) != 0):
        raise RuntimeError("FCFS matching unexpectedly created a self-loop")
    if np.any(matching.astype(np.float64) - state.backlog > tolerance):
        raise RuntimeError("FCFS matching exceeds physical backlog")
    if np.any(matching.sum(axis=1) > available):
        raise RuntimeError("FCFS matching exceeds dispatchable origin idle")

    matching.setflags(write=False)
    available.setflags(write=False)
    matched_count = len(selected_ids)
    return MatchingPlan(
        matching=matching,
        selected_task_ids=tuple(selected_ids),
        matched_task_count=matched_count,
        dispatchable_idle_before=available,
        unmatched_task_count=task_queue.total_waiting_tasks() - matched_count,
        per_origin_diagnostics=tuple(diagnostics),
    )
