"""Pure one-period reward decomposition for the fluid control chain."""

from __future__ import annotations

from dataclasses import dataclass, fields
import math

import numpy as np
from numpy.typing import NDArray

from src.envs.fluid_state import FluidState


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class RewardConfig:
    """External nonnegative reward/cost coefficients and positive output scale."""

    service_reward_per_task: float
    loaded_cost_per_vehicle_minute: float
    rebalancing_cost_per_vehicle_minute: float
    waiting_cost_per_task_minute: float
    reward_scale: float

    def __post_init__(self) -> None:
        for field in fields(self):
            value = float(getattr(self, field.name))
            if not math.isfinite(value):
                raise ValueError(f"{field.name} must be finite")
            if field.name == "reward_scale":
                if value <= 0.0:
                    raise ValueError("reward_scale must be strictly positive")
            elif value < 0.0:
                raise ValueError(f"{field.name} must be nonnegative")
            object.__setattr__(self, field.name, value)


@dataclass(frozen=True)
class RewardBreakdown:
    """Finite unscaled components and scaled scalar reward for one period."""

    service_component: float
    loaded_travel_cost: float
    rebalancing_travel_cost: float
    waiting_cost: float
    unscaled_reward: float
    scaled_reward: float
    matched_tasks: float
    rebalanced_vehicles: float
    waiting_tasks_after_matching: float


def _control_matrix(values: FloatArray, label: str) -> FloatArray:
    array = np.array(values, dtype=np.float64, copy=True)
    if array.shape != (FluidState.NUM_REGIONS, FluidState.NUM_REGIONS):
        raise ValueError(f"{label} must have shape (16, 16)")
    if not np.all(np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError(f"{label} must be finite and nonnegative")
    if np.any(np.diag(array) != 0.0):
        raise ValueError(f"{label} diagonal must be zero")
    return array


def calculate_reward(
    matching: FloatArray,
    rebalancing_flow: FloatArray,
    post_matching_state: FluidState,
    tau: FloatArray,
    delta_t: float,
    config: RewardConfig,
) -> RewardBreakdown:
    """Calculate the causal period reward as a plain Python-float breakdown.

    ``M`` and ``X`` are dispatched vehicle counts.  Travel cost is OD travel
    minutes times dispatched count.  Waiting cost uses only ``tilde_Q`` before
    current-interval arrivals; no future or next-state input is accepted.
    """

    matched = _control_matrix(matching, "matching")
    rebalanced = _control_matrix(rebalancing_flow, "rebalancing_flow")
    post_matching_state.validate()
    if not math.isfinite(delta_t) or delta_t <= 0.0:
        raise ValueError("delta_t must be finite and positive")
    tau_array = np.array(tau, dtype=np.float64, copy=True)
    if tau_array.shape != (FluidState.NUM_REGIONS, FluidState.NUM_REGIONS):
        raise ValueError("tau must have shape (16, 16)")
    active = (matched > 0.0) | (rebalanced > 0.0)
    if np.any(active & (~np.isfinite(tau_array) | (tau_array <= 0.0))):
        index = np.argwhere(active & (~np.isfinite(tau_array) | (tau_array <= 0.0)))[0]
        raise ValueError(f"active OD ({index[0]}, {index[1]}) has invalid tau")

    matched_tasks = float(matched.sum(dtype=np.float64))
    rebalanced_vehicles = float(rebalanced.sum(dtype=np.float64))
    waiting_tasks = float(post_matching_state.backlog.sum(dtype=np.float64))
    loaded_minutes = float(np.sum(tau_array[matched > 0.0] * matched[matched > 0.0]))
    rebalancing_minutes = float(
        np.sum(tau_array[rebalanced > 0.0] * rebalanced[rebalanced > 0.0])
    )
    service_component = config.service_reward_per_task * matched_tasks
    loaded_cost = config.loaded_cost_per_vehicle_minute * loaded_minutes
    rebalancing_cost = config.rebalancing_cost_per_vehicle_minute * rebalancing_minutes
    waiting_cost = config.waiting_cost_per_task_minute * delta_t * waiting_tasks
    unscaled = service_component - loaded_cost - rebalancing_cost - waiting_cost
    scaled = unscaled / config.reward_scale
    values = (
        service_component,
        loaded_cost,
        rebalancing_cost,
        waiting_cost,
        unscaled,
        scaled,
    )
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError("reward calculation produced a nonfinite component")
    return RewardBreakdown(
        service_component=service_component,
        loaded_travel_cost=loaded_cost,
        rebalancing_travel_cost=rebalancing_cost,
        waiting_cost=waiting_cost,
        unscaled_reward=unscaled,
        scaled_reward=scaled,
        matched_tasks=matched_tasks,
        rebalanced_vehicles=rebalanced_vehicles,
        waiting_tasks_after_matching=waiting_tasks,
    )

