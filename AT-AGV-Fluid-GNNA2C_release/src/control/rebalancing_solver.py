"""Exact integer minimum-travel-time rebalancing on feasible direct ODs."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np
from numpy.typing import NDArray
import ortools
from ortools.linear_solver import pywraplp

from src.envs.fluid_state import FluidState


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class RebalancingPlan:
    """Optimal period dispatch quantities and integer transportation diagnostics."""

    flow_matrix: IntArray
    available_idle: IntArray
    target_idle: IntArray
    surplus: IntArray
    deficit: IntArray
    objective_travel_minutes: float
    total_rebalanced: int
    solver_status: str
    solver_backend: str
    solve_time_seconds: float


def _integer_vector(values: IntArray, label: str) -> IntArray:
    array = np.asarray(values)
    if array.shape != (FluidState.NUM_REGIONS,):
        raise ValueError(f"{label} must have shape ({FluidState.NUM_REGIONS},), got {array.shape}")
    if np.issubdtype(array.dtype, np.bool_) or not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{label} must use an integer dtype")
    result = np.array(array, dtype=np.int64, copy=True)
    if np.any(result < 0):
        raise ValueError(f"{label} must be nonnegative")
    return result


def _create_solver() -> tuple[pywraplp.Solver, str]:
    solver = pywraplp.Solver.CreateSolver("SCIP")
    if solver is not None:
        solver.SetNumThreads(1)
        return solver, f"OR-Tools SCIP {ortools.__version__}"
    solver = pywraplp.Solver.CreateSolver("CBC_MIXED_INTEGER_PROGRAMMING")
    if solver is not None:
        solver.SetNumThreads(1)
        return solver, f"OR-Tools CBC {ortools.__version__}"
    raise RuntimeError("no OR-Tools SCIP or CBC integer-programming backend is available")


def solve_integer_rebalancing(
    available_idle: IntArray,
    target_idle: IntArray,
    tau: FloatArray,
    e_reb_mask: BoolArray,
) -> RebalancingPlan:
    """Solve the integer surplus-to-deficit transportation problem.

    Inputs and ``X`` are vehicle counts for one decision period; objective
    coefficients are V2.1 OD travel minutes.  Variables exist only from current
    surplus nodes to current deficit nodes, so same-period transshipment is
    structurally impossible.  Solver failure raises instead of invoking a
    heuristic fallback.
    """

    available = _integer_vector(available_idle, "available_idle")
    target = _integer_vector(target_idle, "target_idle")
    if int(available.sum()) != int(target.sum()):
        raise ValueError("available_idle and target_idle must have the same total")

    tau_array = np.array(tau, dtype=np.float64, copy=True)
    mask = np.array(e_reb_mask, dtype=bool, copy=True)
    expected_shape = (FluidState.NUM_REGIONS, FluidState.NUM_REGIONS)
    if tau_array.shape != expected_shape:
        raise ValueError(f"tau must have shape {expected_shape}, got {tau_array.shape}")
    if mask.shape != expected_shape:
        raise ValueError(f"e_reb_mask must have shape {expected_shape}, got {mask.shape}")
    if np.any(np.diag(mask)):
        raise ValueError("e_reb_mask diagonal must be false")

    difference = available - target
    surplus = np.where(difference > 0, difference, 0).astype(np.int64)
    deficit = np.where(difference < 0, -difference, 0).astype(np.int64)
    if int(surplus.sum()) != int(deficit.sum()):
        raise RuntimeError("surplus and deficit totals are inconsistent")

    flow = np.zeros(expected_shape, dtype=np.int64)
    if int(surplus.sum()) == 0:
        for array in (flow, available, target, surplus, deficit):
            array.setflags(write=False)
        return RebalancingPlan(
            flow_matrix=flow,
            available_idle=available,
            target_idle=target,
            surplus=surplus,
            deficit=deficit,
            objective_travel_minutes=0.0,
            total_rebalanced=0,
            solver_status="OPTIMAL_NO_REBALANCING",
            solver_backend=f"OR-Tools {ortools.__version__} (solver not required)",
            solve_time_seconds=0.0,
        )

    surplus_nodes = [i for i in range(FluidState.NUM_REGIONS) if surplus[i] > 0]
    deficit_nodes = [j for j in range(FluidState.NUM_REGIONS) if deficit[j] > 0]
    candidate_edges: list[tuple[int, int]] = []
    for i in surplus_nodes:
        for j in deficit_nodes:
            if mask[i, j]:
                if not math.isfinite(float(tau_array[i, j])) or tau_array[i, j] <= 0.0:
                    raise ValueError(f"feasible rebalancing OD ({i}, {j}) has invalid tau")
                candidate_edges.append((i, j))
    for i in surplus_nodes:
        if not any(edge[0] == i for edge in candidate_edges):
            raise ValueError(f"rebalancing is infeasible: surplus origin {i} has no deficit edge")
    for j in deficit_nodes:
        if not any(edge[1] == j for edge in candidate_edges):
            raise ValueError(f"rebalancing is infeasible: deficit destination {j} has no surplus edge")

    solver, backend = _create_solver()
    variables: dict[tuple[int, int], pywraplp.Variable] = {}
    for i, j in candidate_edges:
        variables[i, j] = solver.IntVar(0.0, float(surplus[i]), f"x_{i}_{j}")
    for i in surplus_nodes:
        solver.Add(sum(var for (origin, _), var in variables.items() if origin == i) == int(surplus[i]))
    for j in deficit_nodes:
        solver.Add(
            sum(var for (_, destination), var in variables.items() if destination == j)
            == int(deficit[j])
        )
    objective = solver.Objective()
    for (i, j), variable in variables.items():
        objective.SetCoefficient(variable, float(tau_array[i, j]))
    objective.SetMinimization()

    started = time.perf_counter()
    status = solver.Solve()
    elapsed = time.perf_counter() - started
    if status != pywraplp.Solver.OPTIMAL:
        names = {
            pywraplp.Solver.FEASIBLE: "FEASIBLE_NOT_PROVEN_OPTIMAL",
            pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
            pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
            pywraplp.Solver.ABNORMAL: "ABNORMAL",
            pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
        }
        raise RuntimeError(f"integer rebalancing solver failed with status {names.get(status, status)}")

    for (i, j), variable in variables.items():
        value = variable.solution_value()
        rounded = int(round(value))
        if abs(value - rounded) > 1e-7 or rounded < 0:
            raise RuntimeError(f"solver returned a noninteger value {value} for X[{i},{j}]")
        flow[i, j] = rounded

    realized = available - flow.sum(axis=1) + flow.sum(axis=0)
    if not np.array_equal(realized, target):
        raise RuntimeError("optimal flow does not exactly realize target_idle")
    positive = np.argwhere(flow > 0)
    for i, j in positive:
        if surplus[i] <= 0 or deficit[j] <= 0:
            raise RuntimeError("flow violates direct surplus-to-deficit structure")
    objective_value = float(np.sum(flow.astype(np.float64) * tau_array, where=np.isfinite(tau_array)))
    if not math.isfinite(objective_value):
        raise RuntimeError("rebalancing objective is not finite")

    for array in (flow, available, target, surplus, deficit):
        array.setflags(write=False)
    return RebalancingPlan(
        flow_matrix=flow,
        available_idle=available,
        target_idle=target,
        surplus=surplus,
        deficit=deficit,
        objective_travel_minutes=objective_value,
        total_rebalanced=int(flow.sum()),
        solver_status="OPTIMAL",
        solver_backend=backend,
        solve_time_seconds=elapsed,
    )

