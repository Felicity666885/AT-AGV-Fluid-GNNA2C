"""Analytic one-period updates for the deterministic AGV fluid model."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray

from src.envs.fluid_state import FluidState


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

INPUT_TOLERANCE = 1e-10
CONSERVATION_ATOL = 1e-8
CONSERVATION_RTOL = 1e-12


@dataclass(frozen=True)
class FluidStepDiagnostics:
    """Mass-balance diagnostics for one analytic update."""

    completed_loaded: FloatArray
    completed_rebalancing: FloatArray
    dispatched_matching: float
    dispatched_rebalancing: float
    arrivals_added: float
    vehicle_mass_before: float
    vehicle_mass_after: float
    conservation_error: float
    minimum_state_value: float


@dataclass(frozen=True)
class FluidStepResult:
    """Next physical state and diagnostics produced by :func:`fluid_step`."""

    next_state: FluidState
    diagnostics: FluidStepDiagnostics


def _matrix(values: NDArray[np.floating] | list[list[float]], label: str) -> FloatArray:
    result = np.array(values, dtype=np.float64, copy=True)
    expected = (FluidState.NUM_REGIONS, FluidState.NUM_REGIONS)
    if result.shape != expected:
        raise ValueError(f"{label} must have shape {expected}, got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must contain only finite values")
    if float(np.min(result)) < -INPUT_TOLERANCE:
        raise ValueError(f"{label} must be nonnegative")
    result[(result < 0.0) & (result >= -INPUT_TOLERANCE)] = 0.0
    return result


def _zero_small_negatives(values: FloatArray, label: str) -> FloatArray:
    result = np.array(values, dtype=np.float64, copy=True)
    minimum = float(np.min(result))
    if minimum < -INPUT_TOLERANCE:
        raise ValueError(f"analytic update produced materially negative {label}: {minimum}")
    result[(result < 0.0) & (result >= -INPUT_TOLERANCE)] = 0.0
    return result


def _normalized_state(state: FluidState) -> FluidState:
    state.validate(INPUT_TOLERANCE)
    normalized = state.copy()
    for values in (
        normalized.backlog,
        normalized.idle,
        normalized.loaded,
        normalized.rebalancing,
    ):
        values[(values < 0.0) & (values >= -INPUT_TOLERANCE)] = 0.0
    for values in (normalized.backlog, normalized.loaded, normalized.rebalancing):
        np.fill_diagonal(values, 0.0)
    return normalized


def _check_zero_diagonal(values: FloatArray, label: str) -> None:
    if np.any(np.abs(np.diag(values)) > INPUT_TOLERANCE):
        raise ValueError(f"{label} diagonal must be zero")


def build_post_matching_snapshot(state: FluidState, matching: FloatArray) -> FluidState:
    """Build temporary post-matching state ``tilde_s_t`` without mutation.

    ``matching`` is the period quantity :math:`M_{ij}(t)`, not a rate.  This
    snapshot subtracts it from backlog and origin idle mass solely for future
    policy/rebalancing inputs.  It does not create loaded flow and must never be
    passed as the physical initial state to :func:`fluid_step`; that update must
    start from the original ``s_t`` and apply ``M`` exactly once.
    """

    physical = _normalized_state(state)
    matched = _matrix(matching, "matching")
    _check_zero_diagonal(matched, "matching")
    if np.any(matched - physical.backlog > INPUT_TOLERANCE):
        raise ValueError("matching cannot exceed current backlog")
    dispatched = matched.sum(axis=1, dtype=np.float64)
    if np.any(dispatched - physical.idle > INPUT_TOLERANCE):
        raise ValueError("matching dispatched from an origin cannot exceed its idle vehicles")

    backlog_tilde = _zero_small_negatives(physical.backlog - matched, "post-matching backlog")
    idle_tilde = _zero_small_negatives(physical.idle - dispatched, "post-matching idle")
    np.fill_diagonal(backlog_tilde, 0.0)
    snapshot = FluidState(
        backlog=backlog_tilde,
        idle=idle_tilde,
        loaded=physical.loaded,
        rebalancing=physical.rebalancing,
    )
    snapshot.validate(INPUT_TOLERANCE)
    return snapshot


def fluid_step(
    state: FluidState,
    arrivals: FloatArray,
    matching: FloatArray,
    rebalancing_flow: FloatArray,
    tau: FloatArray,
    delta_t: float,
    e_reb_mask: BoolArray,
    fleet_size: float | None = None,
) -> FluidStepResult:
    """Advance the physical state by one decision period analytically.

    All times are minutes. ``matching`` (:math:`M`) and ``rebalancing_flow``
    (:math:`X`) are vehicle quantities dispatched during the full period, while
    :math:`m=M/Delta_t` and :math:`x=X/Delta_t` are their piecewise-constant
    rates.  For each valid OD, ``rho = exp(-delta_t/tau)`` and the in-transit
    states use the exact constant-input solution.  Completion quantities are
    obtained from mass balance so idle + loaded + rebalancing mass is conserved.

    New ``arrivals`` belong to ``[t,t+Delta_t)`` and are added only to the next
    backlog: ``Q_next = Q - M + A``.  The function never reads files and never
    mutates any input array or ``state``.
    """

    physical = _normalized_state(state)
    arrivals_array = _matrix(arrivals, "arrivals")
    matching_array = _matrix(matching, "matching")
    rebalancing_array = _matrix(rebalancing_flow, "rebalancing_flow")
    for label, values in (
        ("arrivals", arrivals_array),
        ("matching", matching_array),
        ("rebalancing_flow", rebalancing_array),
    ):
        _check_zero_diagonal(values, label)

    tau_array = np.array(tau, dtype=np.float64, copy=True)
    expected_shape = (FluidState.NUM_REGIONS, FluidState.NUM_REGIONS)
    if tau_array.shape != expected_shape:
        raise ValueError(f"tau must have shape {expected_shape}, got {tau_array.shape}")
    if not math.isfinite(delta_t) or delta_t <= 0.0:
        raise ValueError("delta_t must be finite and positive")
    reb_mask = np.array(e_reb_mask, dtype=bool, copy=True)
    if reb_mask.shape != expected_shape:
        raise ValueError(f"e_reb_mask must have shape {expected_shape}, got {reb_mask.shape}")
    if np.any(np.diag(reb_mask)):
        raise ValueError("e_reb_mask diagonal must be false")
    if np.any((rebalancing_array > INPUT_TOLERANCE) & ~reb_mask):
        raise ValueError("rebalancing_flow contains an OD outside E_reb")
    if np.any(matching_array - physical.backlog > INPUT_TOLERANCE):
        raise ValueError("matching cannot exceed current backlog")

    total_dispatch = matching_array.sum(axis=1) + rebalancing_array.sum(axis=1)
    if np.any(total_dispatch - physical.idle > INPUT_TOLERANCE):
        raise ValueError("matching plus rebalancing dispatch cannot exceed origin idle vehicles")

    active_tau = (
        (physical.loaded > 0.0)
        | (physical.rebalancing > 0.0)
        | (matching_array > 0.0)
        | (rebalancing_array > 0.0)
    )
    valid_tau = np.isfinite(tau_array) & (tau_array > 0.0)
    if np.any(active_tau & ~valid_tau):
        invalid = np.argwhere(active_tau & ~valid_tau)[0]
        raise ValueError(f"active OD ({invalid[0]}, {invalid[1]}) has invalid tau")

    rho = np.zeros(expected_shape, dtype=np.float64)
    rho[valid_tau] = np.exp(-delta_t / tau_array[valid_tau])
    loaded_next = np.zeros(expected_shape, dtype=np.float64)
    rebalancing_next = np.zeros(expected_shape, dtype=np.float64)
    loaded_next[valid_tau] = (
        rho[valid_tau] * physical.loaded[valid_tau]
        + (matching_array[valid_tau] / delta_t)
        * tau_array[valid_tau]
        * (1.0 - rho[valid_tau])
    )
    rebalancing_next[valid_tau] = (
        rho[valid_tau] * physical.rebalancing[valid_tau]
        + (rebalancing_array[valid_tau] / delta_t)
        * tau_array[valid_tau]
        * (1.0 - rho[valid_tau])
    )

    completed_loaded = _zero_small_negatives(
        physical.loaded + matching_array - loaded_next, "loaded completions"
    )
    completed_rebalancing = _zero_small_negatives(
        physical.rebalancing + rebalancing_array - rebalancing_next,
        "rebalancing completions",
    )
    backlog_next = _zero_small_negatives(
        physical.backlog - matching_array + arrivals_array, "backlog"
    )
    idle_next = _zero_small_negatives(
        physical.idle
        - matching_array.sum(axis=1)
        - rebalancing_array.sum(axis=1)
        + completed_loaded.sum(axis=0)
        + completed_rebalancing.sum(axis=0),
        "idle",
    )
    loaded_next = _zero_small_negatives(loaded_next, "loaded")
    rebalancing_next = _zero_small_negatives(rebalancing_next, "rebalancing")
    for values in (backlog_next, loaded_next, rebalancing_next, completed_loaded, completed_rebalancing):
        np.fill_diagonal(values, 0.0)

    next_state = FluidState(backlog_next, idle_next, loaded_next, rebalancing_next)
    next_state.validate(INPUT_TOLERANCE)
    mass_before = physical.total_vehicle_mass()
    mass_after = next_state.total_vehicle_mass()
    target_mass = mass_before if fleet_size is None else float(fleet_size)
    if not math.isfinite(target_mass) or target_mass < 0.0:
        raise ValueError("fleet_size must be finite and nonnegative when supplied")
    conservation_error = mass_after - target_mass
    allowed_error = max(CONSERVATION_ATOL, CONSERVATION_RTOL * max(1.0, abs(target_mass)))
    if abs(conservation_error) > allowed_error:
        raise ValueError(
            f"vehicle conservation failed: mass_after={mass_after}, target={target_mass}, "
            f"error={conservation_error}"
        )

    diagnostics = FluidStepDiagnostics(
        completed_loaded=completed_loaded.copy(),
        completed_rebalancing=completed_rebalancing.copy(),
        dispatched_matching=float(matching_array.sum(dtype=np.float64)),
        dispatched_rebalancing=float(rebalancing_array.sum(dtype=np.float64)),
        arrivals_added=float(arrivals_array.sum(dtype=np.float64)),
        vehicle_mass_before=mass_before,
        vehicle_mass_after=mass_after,
        conservation_error=conservation_error,
        minimum_state_value=next_state.minimum_value(),
    )
    return FluidStepResult(next_state=next_state, diagnostics=diagnostics)

