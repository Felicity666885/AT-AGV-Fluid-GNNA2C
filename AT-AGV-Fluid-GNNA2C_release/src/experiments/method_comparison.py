"""Unified deterministic evaluation for the Chapter-5 method comparison."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import numpy as np
import torch

from src.baselines.demand_driven_rebalancing import (
    DemandDrivenRebalancingPlan,
    solve_demand_driven_rebalancing,
)
from src.control.fluid_control_step import execute_fluid_control_step_from_precomputed_matching
from src.control.target_rounding import uniform_target
from src.data.scenario_loader import FluidScenario
from src.envs.fluid_dynamics import fluid_step
from src.experiments.experiment_metrics import (
    TaskWaitingRecord,
    aggregate_reward,
    build_waiting_records,
    waiting_summary,
)
from src.policy.actor import DirichletActor
from src.policy.critic import GNNCritic
from src.policy.directed_graph import IncomingNormalizedGraph
from src.policy.node_features import FeatureScaleConfig
from src.policy.policy_forward import forward_policy
from src.training.episode_state import (
    initialize_episode_state,
    is_drain_mode,
    natural_termination_reached,
)
from src.training.pilot_trainer import execute_drain_step
from src.training.prepared_decision import prepare_decision
from src.training.reward import RewardBreakdown, RewardConfig, calculate_reward


MethodMode = Literal["uniform", "demand_driven", "gnn_a2c"]


@dataclass(frozen=True)
class UnifiedEvaluationResult:
    """One complete deterministic episode under the shared environment."""

    method_mode: MethodMode
    metrics: dict[str, object]
    demand_plans: tuple[DemandDrivenRebalancingPlan, ...]


def _execute_demand_step(
    *,
    state,
    task_queue,
    prepared,
    arrivals,
    released_tasks,
    tau,
    delta_t,
    e_reb_mask,
    fleet_size,
) -> tuple[object, object, DemandDrivenRebalancingPlan, object]:
    """Apply a precomputed FCFS plan and direct heuristic ``X`` from ``s_t``."""

    task_queue.validate_against_backlog(state.backlog)
    plan = solve_demand_driven_rebalancing(
        prepared.post_matching_snapshot, tau, e_reb_mask, reserve=1
    )
    fluid_result = fluid_step(
        state=state,
        arrivals=arrivals,
        matching=prepared.matching_plan.matching,
        rebalancing_flow=plan.flow_matrix,
        tau=tau,
        delta_t=delta_t,
        e_reb_mask=e_reb_mask,
        fleet_size=fleet_size,
    )
    next_queue = task_queue.remove_selected_tasks(
        prepared.matching_plan.selected_task_ids
    ).add_released_tasks(released_tasks)
    next_queue.validate_against_backlog(fluid_result.next_state.backlog)
    return fluid_result.next_state, next_queue, plan, fluid_result.diagnostics


def run_unified_deterministic_evaluation(
    *,
    method_mode: MethodMode,
    scenario: FluidScenario,
    graph: IncomingNormalizedGraph,
    feature_scaler: FeatureScaleConfig,
    reward_config: RewardConfig,
    initial_idle_distribution: np.ndarray,
    fleet_size: float,
    arrival_window_steps: int,
    transit_clear_tolerance: float,
    max_decision_steps: int,
    run_id: str,
    seed: int | None = None,
    actor: DirichletActor | None = None,
    critic: GNNCritic | None = None,
) -> UnifiedEvaluationResult:
    """Run one method under identical FCFS, arrivals, dynamics, reward, and stop rules.

    For ``gnn_a2c``, Actor/Critic forward passes are deterministic and wrapped
    in ``torch.no_grad``.  This function contains no optimizer and cannot train.
    """

    if method_mode not in {"uniform", "demand_driven", "gnn_a2c"}:
        raise ValueError("unknown method_mode")
    if method_mode == "gnn_a2c":
        if actor is None or critic is None:
            raise ValueError("gnn_a2c evaluation requires actor and critic")
        actor.eval()
        critic.eval()
    elif actor is not None or critic is not None:
        raise ValueError("deterministic baselines must not receive neural networks")

    episode = initialize_episode_state(fleet_size, initial_idle_distribution, run_id)
    task_lookup = {task.task_id: task for task in scenario.task_records}
    rewards: list[RewardBreakdown] = []
    waiting_records: list[TaskWaitingRecord] = []
    demand_plans: list[DemandDrivenRebalancingPlan] = []
    max_vehicle_error = 0.0
    max_task_flow_error = 0.0
    max_ledger_error = 0.0
    total_rebalanced = 0
    matched_by_horizon: int | None = None
    backlog_at_horizon: float | None = None
    queue_cleared_step: int | None = None

    while not episode.terminated:
        if episode.decision_step >= max_decision_steps:
            raise RuntimeError(
                f"{method_mode} did not naturally terminate within {max_decision_steps} steps"
            )
        before = episode.physical_state.copy()
        vehicle_before = before.total_vehicle_mass()

        if is_drain_mode(episode, scenario.arrivals.number_of_steps):
            execute_drain_step(episode, scenario)
            max_vehicle_error = max(
                max_vehicle_error,
                abs(vehicle_before - fleet_size),
                abs(episode.physical_state.total_vehicle_mass() - fleet_size),
            )
            max_task_flow_error = max(
                max_task_flow_error,
                float(np.max(np.abs(episode.physical_state.backlog - before.backlog))),
            )
            if natural_termination_reached(
                episode, scenario.arrivals.number_of_steps, transit_clear_tolerance
            ):
                episode.terminated = True
            continue

        prepared = prepare_decision(
            episode.physical_state,
            episode.task_queue,
            episode.past_arrivals,
            arrival_window_steps,
            feature_scaler,
            graph,
            scenario.decision_period_min,
        )
        step = episode.decision_step
        arrivals = scenario.arrivals.arrivals_at(step)
        released = scenario.tasks_released_at(step)
        matching = np.asarray(prepared.matching_plan.matching, dtype=np.int64)

        if method_mode == "demand_driven":
            next_state, next_queue, demand_plan, diagnostics = _execute_demand_step(
                state=episode.physical_state,
                task_queue=episode.task_queue,
                prepared=prepared,
                arrivals=arrivals,
                released_tasks=released,
                tau=scenario.tau_minutes,
                delta_t=scenario.decision_period_min,
                e_reb_mask=scenario.e_reb_mask,
                fleet_size=fleet_size,
            )
            rebalancing = demand_plan.flow_matrix
            demand_plans.append(demand_plan)
        else:
            if method_mode == "uniform":
                target = uniform_target()
            else:
                with torch.no_grad():
                    policy = forward_policy(
                        actor, critic, prepared.policy_observation, deterministic=True
                    )
                target = policy.actor_output.action_proportions.detach().cpu().numpy().copy()
            control = execute_fluid_control_step_from_precomputed_matching(
                episode.physical_state,
                episode.task_queue,
                prepared.matching_plan,
                prepared.post_matching_snapshot,
                arrivals,
                released,
                target,
                scenario.tau_minutes,
                scenario.decision_period_min,
                scenario.e_reb_mask,
                fleet_size,
            )
            next_state = control.next_state
            next_queue = control.next_task_queue
            rebalancing = control.rebalancing_plan.flow_matrix
            diagnostics = control.fluid_step_diagnostics

        reward = calculate_reward(
            matching,
            rebalancing,
            prepared.post_matching_snapshot,
            scenario.tau_minutes,
            scenario.decision_period_min,
            reward_config,
        )
        rewards.append(reward)
        selected = prepared.matching_plan.selected_task_ids
        episode.record_matched_tasks(selected)
        waiting_records.extend(
            build_waiting_records(
                selected,
                task_lookup,
                step,
                scenario.decision_period_min,
                method_mode,
                0 if seed is None else seed,
            )
        )
        expected_backlog = before.backlog - matching + arrivals
        max_task_flow_error = max(
            max_task_flow_error,
            float(np.max(np.abs(next_state.backlog - expected_backlog))),
        )
        ledger_error = float(
            np.max(
                np.abs(next_queue.count_matrix().astype(np.float64) - next_state.backlog)
            )
        )
        max_ledger_error = max(max_ledger_error, ledger_error)
        max_vehicle_error = max(
            max_vehicle_error,
            abs(vehicle_before - fleet_size),
            abs(next_state.total_vehicle_mass() - fleet_size),
            abs(float(diagnostics.conservation_error)),
        )
        total_rebalanced += int(np.asarray(rebalancing).sum())

        episode.physical_state = next_state
        episode.task_queue = next_queue
        episode.append_past_arrival(arrivals)
        episode.decision_step += 1

        if step == scenario.arrivals.number_of_steps - 1:
            matched_by_horizon = len(episode.matched_task_ids)
            backlog_at_horizon = float(episode.physical_state.backlog.sum())
        if (
            queue_cleared_step is None
            and episode.decision_step >= scenario.arrivals.number_of_steps
            and episode.task_queue.total_waiting_tasks() == 0
        ):
            queue_cleared_step = step
        if natural_termination_reached(
            episode, scenario.arrivals.number_of_steps, transit_clear_tolerance
        ):
            episode.terminated = True

    if len(episode.matched_task_ids) != scenario.task_count:
        raise RuntimeError(f"{method_mode} did not uniquely match all scenario tasks")
    if episode.task_queue.total_waiting_tasks() != 0 or float(episode.physical_state.backlog.sum()) > 1e-10:
        raise RuntimeError(f"{method_mode} terminated with nonempty task backlog")
    if max_vehicle_error > 1e-8 or max_task_flow_error > 1e-10 or max_ledger_error > 1e-10:
        raise RuntimeError(
            f"{method_mode} conservation failure: vehicle={max_vehicle_error}, "
            f"task={max_task_flow_error}, ledger={max_ledger_error}"
        )

    reward_totals = aggregate_reward(rewards, method_mode, 0 if seed is None else seed)
    waits = waiting_summary(waiting_records)
    horizon_matched = (
        len(episode.matched_task_ids) if matched_by_horizon is None else matched_by_horizon
    )
    horizon_backlog = (
        float(episode.physical_state.backlog.sum())
        if backlog_at_horizon is None
        else backlog_at_horizon
    )
    metrics: dict[str, object] = {
        "run_id": run_id,
        "seed": "" if seed is None else seed,
        "cumulative_scaled_reward": float(reward_totals["scaled_reward"]),
        "service_rate_by_release_horizon": horizon_matched / scenario.task_count,
        "mean_waiting_time": waits["mean_waiting_time"],
        "median_waiting_time": waits["median_waiting_time"],
        "p95_waiting_time": waits["p95_waiting_time"],
        "total_waiting_cost": float(reward_totals["waiting_cost"]),
        "total_loaded_travel_cost": float(reward_totals["loaded_travel_cost"]),
        "total_rebalancing_cost": float(reward_totals["rebalancing_travel_cost"]),
        "total_rebalanced_vehicles": total_rebalanced,
        "backlog_at_release_horizon": horizon_backlog,
        "matched_by_release_horizon": horizon_matched,
        "total_episode_steps": episode.decision_step,
        "queue_cleared_step": "" if queue_cleared_step is None else queue_cleared_step,
        "maximum_vehicle_conservation_error": max_vehicle_error,
        "maximum_task_flow_conservation_error": max_task_flow_error,
        "terminated": episode.terminated,
        "truncated": False,
        "total_matched": len(episode.matched_task_ids),
    }
    if not all(
        math.isfinite(float(metrics[key]))
        for key in (
            "cumulative_scaled_reward",
            "service_rate_by_release_horizon",
            "mean_waiting_time",
            "median_waiting_time",
            "p95_waiting_time",
            "total_waiting_cost",
            "total_loaded_travel_cost",
            "total_rebalancing_cost",
            "maximum_vehicle_conservation_error",
            "maximum_task_flow_conservation_error",
        )
    ):
        raise RuntimeError(f"{method_mode} produced a nonfinite evaluation metric")
    return UnifiedEvaluationResult(method_mode, metrics, tuple(demand_plans))
