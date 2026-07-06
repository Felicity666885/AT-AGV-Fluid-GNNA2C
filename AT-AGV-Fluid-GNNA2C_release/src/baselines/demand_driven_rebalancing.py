"""Causal demand-driven proportional rebalancing heuristic.

The heuristic is deliberately transparent and deterministic.  It consumes
only the post-FCFS state at the current decision epoch; no arrival history,
future arrivals, learned policy, value function, or checkpoint is accepted.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray

from src.control.matching_solver import DEFAULT_INTEGER_TOLERANCE
from src.control.target_rounding import largest_remainder_round
from src.envs.fluid_state import FluidState


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class DemandDrivenRebalancingPlan:
    """One deterministic period action and its auditable intermediate values."""

    flow_matrix: IntArray
    remaining_queue_by_origin: FloatArray
    dispatchable_idle: IntArray
    movable_idle: IntArray
    target_inflow: IntArray
    rebalancing_budget: int
    total_rebalanced: int
    objective_travel_minutes: float
    reserve: int
    unfilled_target: int


def solve_demand_driven_rebalancing(
    post_matching_state: FluidState,
    tau: FloatArray,
    e_reb_mask: BoolArray,
    *,
    reserve: int = 1,
    tolerance: float = DEFAULT_INTEGER_TOLERANCE,
) -> DemandDrivenRebalancingPlan:
    """Return the fixed demand-driven integer rebalancing action ``X``.

    ``post_matching_state`` is :math:`\tilde{s}_t`.  For region ``i``, at most
    ``max(floor(idle_i + tolerance) - reserve, 0)`` vehicles are movable.
    The period budget is the smaller of total movable vehicles and the ceiling
    of remaining task backlog.  Destinations receive a largest-remainder
    allocation proportional to current outgoing backlog, then source vehicles
    are selected by increasing ``tau[i, j]`` with ascending region-index ties.

    The function has no argument through which current-interval or future
    arrivals can enter, and it never mutates the supplied state or arrays.
    """

    if isinstance(reserve, bool) or not isinstance(reserve, (int, np.integer)):
        raise TypeError("reserve must be a nonnegative integer")
    reserve_value = int(reserve)
    if reserve_value < 0:
        raise ValueError("reserve must be nonnegative")
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and nonnegative")

    post_matching_state.validate(tolerance)
    tau_values = np.array(tau, dtype=np.float64, copy=True)
    mask = np.array(e_reb_mask, dtype=bool, copy=True)
    shape = (FluidState.NUM_REGIONS, FluidState.NUM_REGIONS)
    if tau_values.shape != shape or mask.shape != shape:
        raise ValueError("tau and e_reb_mask must both have shape (16, 16)")
    if np.any(np.diag(mask)):
        raise ValueError("e_reb_mask diagonal must be false")

    queue = np.asarray(
        post_matching_state.backlog.sum(axis=1, dtype=np.float64), dtype=np.float64
    )
    if queue.shape != (FluidState.NUM_REGIONS,) or not np.all(np.isfinite(queue)):
        raise ValueError("post-matching queue by origin must be a finite 16-vector")
    if np.any(queue < -tolerance):
        raise ValueError("post-matching queue cannot be negative")
    queue = np.where(queue < 0.0, 0.0, queue)

    idle = np.asarray(post_matching_state.idle, dtype=np.float64)
    dispatchable = np.floor(idle + tolerance).astype(np.int64)
    if np.any(dispatchable < 0) or np.any(dispatchable > idle + tolerance):
        raise RuntimeError("integer dispatchable idle calculation is inconsistent")
    movable = np.maximum(dispatchable - reserve_value, 0).astype(np.int64)
    budget = min(int(movable.sum()), int(math.ceil(float(queue.sum()))))

    flow = np.zeros(shape, dtype=np.int64)
    target = np.zeros(FluidState.NUM_REGIONS, dtype=np.int64)
    remaining = movable.copy()
    if float(queue.sum()) > 0.0 and budget > 0:
        target = largest_remainder_round(queue, budget)
        destination_order = sorted(
            range(FluidState.NUM_REGIONS), key=lambda region: (-queue[region], region)
        )
        for destination in destination_order:
            needed = int(target[destination])
            while needed > 0:
                sources = [
                    origin
                    for origin in range(FluidState.NUM_REGIONS)
                    if origin != destination
                    and remaining[origin] > 0
                    and mask[origin, destination]
                    and math.isfinite(float(tau_values[origin, destination]))
                    and tau_values[origin, destination] > 0.0
                ]
                if not sources:
                    break
                source = min(
                    sources,
                    key=lambda origin: (float(tau_values[origin, destination]), origin),
                )
                dispatched = min(needed, int(remaining[source]))
                flow[source, destination] += dispatched
                remaining[source] -= dispatched
                needed -= dispatched

    if np.any(np.diag(flow)) or np.any(flow < 0):
        raise RuntimeError("demand-driven action contains an invalid flow")
    if np.any((flow > 0) & ~mask):
        raise RuntimeError("demand-driven action uses an OD outside E_reb")
    if np.any(flow.sum(axis=1) > movable):
        raise RuntimeError("demand-driven action exceeds movable idle vehicles")
    total_rebalanced = int(flow.sum())
    if total_rebalanced > budget:
        raise RuntimeError("demand-driven action exceeds the period budget")
    if np.any(dispatchable - flow.sum(axis=1) < np.minimum(dispatchable, reserve_value)):
        raise RuntimeError("demand-driven action violates the fixed reserve")

    positive = flow > 0
    objective = float(np.sum(flow[positive] * tau_values[positive])) if np.any(positive) else 0.0
    if not math.isfinite(objective):
        raise RuntimeError("demand-driven travel objective is nonfinite")
    unfilled = int(target.sum() - flow.sum())

    outputs = (flow, queue, dispatchable, movable, target)
    for values in outputs:
        values.setflags(write=False)
    return DemandDrivenRebalancingPlan(
        flow_matrix=flow,
        remaining_queue_by_origin=queue,
        dispatchable_idle=dispatchable,
        movable_idle=movable,
        target_inflow=target,
        rebalancing_budget=budget,
        total_rebalanced=total_rebalanced,
        objective_travel_minutes=objective,
        reserve=reserve_value,
        unfilled_target=unfilled,
    )

