"""Pure deterministic control chain joining matching, targets, ILP, and fluid dynamics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from src.control.matching_solver import (
    DEFAULT_INTEGER_TOLERANCE,
    MatchingPlan,
    dispatchable_integer_idle,
    solve_fcfs_matching,
)
from src.control.rebalancing_solver import RebalancingPlan, solve_integer_rebalancing
from src.control.target_rounding import largest_remainder_round, normalize_proportions
from src.control.task_queue import TaskQueueLedger
from src.data.scenario_loader import TaskRecord
from src.envs.fluid_dynamics import (
    FluidStepDiagnostics,
    build_post_matching_snapshot,
    fluid_step,
)
from src.envs.fluid_state import FluidState


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class ControlStepResult:
    """All deterministic decisions and the next synchronized physical/queue state."""

    next_state: FluidState
    next_task_queue: TaskQueueLedger
    matching_plan: MatchingPlan
    post_matching_snapshot: FluidState
    target_proportions: FloatArray
    target_idle: IntArray
    rebalancing_plan: RebalancingPlan
    fluid_step_diagnostics: FluidStepDiagnostics


def execute_fluid_control_step(
    state: FluidState,
    task_queue: TaskQueueLedger,
    arrivals_matrix: FloatArray,
    released_tasks: Sequence[TaskRecord],
    target_proportions: FloatArray,
    tau: FloatArray,
    delta_t: float,
    e_reb_mask: BoolArray,
    fleet_size: float | None = None,
    tolerance: float = DEFAULT_INTEGER_TOLERANCE,
) -> ControlStepResult:
    """Execute ``s_t -> M -> tilde_s_t -> q -> target -> X -> fluid_step``.

    ``M`` and ``X`` are integer vehicle quantities for the period.  The fluid
    update starts from the original physical ``s_t``, never ``tilde_s_t``.
    Current interval releases are removed from matching eligibility and are
    appended to the next task ledger only after the physical update.
    """

    matching_plan = solve_fcfs_matching(state, task_queue, tolerance)
    post_matching = build_post_matching_snapshot(state, matching_plan.matching)
    return execute_fluid_control_step_from_precomputed_matching(
        state=state,
        task_queue=task_queue,
        matching_plan=matching_plan,
        post_matching_snapshot=post_matching,
        arrivals_matrix=arrivals_matrix,
        released_tasks=released_tasks,
        target_proportions=target_proportions,
        tau=tau,
        delta_t=delta_t,
        e_reb_mask=e_reb_mask,
        fleet_size=fleet_size,
        tolerance=tolerance,
    )


def _validate_precomputed_matching(
    state: FluidState,
    task_queue: TaskQueueLedger,
    matching_plan: MatchingPlan,
    post_matching_snapshot: FluidState,
    tolerance: float,
) -> None:
    """Validate cached FCFS identities and ``tilde_s_t`` without solving again."""

    state.validate(tolerance)
    task_queue.validate_against_backlog(state.backlog, tolerance)
    available = dispatchable_integer_idle(state.idle, tolerance)
    expected_matching = np.zeros((FluidState.NUM_REGIONS, FluidState.NUM_REGIONS), dtype=np.int64)
    expected_ids: list[int] = []
    for origin in range(FluidState.NUM_REGIONS):
        selected = task_queue.tasks_at_origin(origin)[: int(available[origin])]
        for task in selected:
            expected_matching[origin, task.destination] += 1
            expected_ids.append(task.task_id)
    if not np.array_equal(np.asarray(matching_plan.matching), expected_matching):
        raise ValueError("precomputed MatchingPlan does not match current deterministic FCFS result")
    if tuple(matching_plan.selected_task_ids) != tuple(expected_ids):
        raise ValueError("precomputed selected_task_ids do not match current FCFS identities")
    if matching_plan.matched_task_count != len(expected_ids):
        raise ValueError("precomputed matched_task_count is inconsistent")
    if matching_plan.unmatched_task_count != task_queue.total_waiting_tasks() - len(expected_ids):
        raise ValueError("precomputed unmatched_task_count is inconsistent")
    if not np.array_equal(np.asarray(matching_plan.dispatchable_idle_before), available):
        raise ValueError("precomputed dispatchable idle is inconsistent with current state")

    expected_snapshot = build_post_matching_snapshot(state, expected_matching)
    for label, actual, expected in (
        ("backlog", post_matching_snapshot.backlog, expected_snapshot.backlog),
        ("idle", post_matching_snapshot.idle, expected_snapshot.idle),
        ("loaded", post_matching_snapshot.loaded, expected_snapshot.loaded),
        ("rebalancing", post_matching_snapshot.rebalancing, expected_snapshot.rebalancing),
    ):
        if not np.allclose(actual, expected, atol=tolerance, rtol=0.0):
            raise ValueError(f"precomputed post-matching {label} is inconsistent")


def execute_fluid_control_step_from_precomputed_matching(
    state: FluidState,
    task_queue: TaskQueueLedger,
    matching_plan: MatchingPlan,
    post_matching_snapshot: FluidState,
    arrivals_matrix: FloatArray,
    released_tasks: Sequence[TaskRecord],
    target_proportions: FloatArray,
    tau: FloatArray,
    delta_t: float,
    e_reb_mask: BoolArray,
    fleet_size: float | None = None,
    tolerance: float = DEFAULT_INTEGER_TOLERANCE,
) -> ControlStepResult:
    """Execute target, ILP, and fluid updates from one precomputed FCFS plan.

    The function never calls :func:`solve_fcfs_matching`.  It verifies the plan
    and post-matching snapshot against current task identities and physical
    state, then starts :func:`fluid_step` from the original ``s_t``.
    """

    _validate_precomputed_matching(
        state, task_queue, matching_plan, post_matching_snapshot, tolerance
    )
    state_before = state.copy()
    queue_ids_before = task_queue.task_ids
    arrivals_before = np.array(arrivals_matrix, copy=True)
    tau_before = np.array(tau, copy=True)
    mask_before = np.array(e_reb_mask, copy=True)
    proportions_before = np.array(target_proportions, copy=True)

    released_ledger = TaskQueueLedger(tuple(released_tasks))
    arrival_values = np.asarray(arrivals_matrix, dtype=np.float64)
    if arrival_values.shape != (FluidState.NUM_REGIONS, FluidState.NUM_REGIONS):
        raise ValueError("arrivals_matrix must have shape (16, 16)")
    if not np.array_equal(released_ledger.count_matrix().astype(np.float64), arrival_values):
        raise ValueError("released_tasks counts must exactly equal arrivals_matrix")

    available = dispatchable_integer_idle(post_matching_snapshot.idle, tolerance)
    normalized_proportions = normalize_proportions(target_proportions)
    target_idle = largest_remainder_round(normalized_proportions, int(available.sum()))
    rebalancing_plan = solve_integer_rebalancing(available, target_idle, tau, e_reb_mask)

    fluid_result = fluid_step(
        state=state,
        arrivals=arrival_values,
        matching=matching_plan.matching,
        rebalancing_flow=rebalancing_plan.flow_matrix,
        tau=tau,
        delta_t=delta_t,
        e_reb_mask=e_reb_mask,
        fleet_size=fleet_size,
    )
    next_queue = task_queue.remove_selected_tasks(
        matching_plan.selected_task_ids
    ).add_released_tasks(released_tasks)
    next_queue.validate_against_backlog(fluid_result.next_state.backlog, tolerance)

    state_components_before = (
        state_before.backlog,
        state_before.idle,
        state_before.loaded,
        state_before.rebalancing,
    )
    state_components_after = (state.backlog, state.idle, state.loaded, state.rebalancing)
    if any(
        not np.array_equal(before, after)
        for before, after in zip(state_components_before, state_components_after)
    ):
        raise RuntimeError("execute_fluid_control_step mutated the input physical state")
    if task_queue.task_ids != queue_ids_before:
        raise RuntimeError("execute_fluid_control_step mutated the input task queue")
    for label, before, after in (
        ("arrivals_matrix", arrivals_before, np.asarray(arrivals_matrix)),
        ("tau", tau_before, np.asarray(tau)),
        ("e_reb_mask", mask_before, np.asarray(e_reb_mask)),
        ("target_proportions", proportions_before, np.asarray(target_proportions)),
    ):
        if not np.array_equal(before, after, equal_nan=True):
            raise RuntimeError(f"execute_fluid_control_step mutated input {label}")

    normalized_proportions.setflags(write=False)
    target_idle.setflags(write=False)
    return ControlStepResult(
        next_state=fluid_result.next_state,
        next_task_queue=next_queue,
        matching_plan=matching_plan,
        post_matching_snapshot=post_matching_snapshot.copy(),
        target_proportions=normalized_proportions,
        target_idle=target_idle,
        rebalancing_plan=rebalancing_plan,
        fluid_step_diagnostics=fluid_result.diagnostics,
    )
