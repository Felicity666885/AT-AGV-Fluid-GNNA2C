"""Complete Stage-5A episode runner without changing the reviewed control core."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np
import torch

from src.control.fluid_control_step import execute_fluid_control_step_from_precomputed_matching
from src.control.target_rounding import hold_current_target, uniform_target
from src.data.scenario_loader import FluidScenario
from src.policy.actor import DirichletActor
from src.policy.critic import GNNCritic
from src.policy.directed_graph import IncomingNormalizedGraph
from src.policy.node_features import FeatureScaleConfig
from src.policy.policy_forward import forward_policy
from src.training.a2c_update import A2CUpdateResult, A2CUpdater
from src.training.episode_state import (
    EpisodeState,
    initialize_episode_state,
    is_drain_mode,
    natural_termination_reached,
)
from src.training.pilot_trainer import execute_drain_step
from src.training.prepared_decision import PreparedDecision, prepare_decision
from src.training.reward import RewardBreakdown, RewardConfig, calculate_reward
from src.training.rollout_buffer import NStepRolloutBuffer, RolloutTransition

from src.experiments.experiment_metrics import (
    FeatureAccumulator,
    TaskWaitingRecord,
    aggregate_reward,
    build_waiting_records,
    waiting_summary,
)


@dataclass(frozen=True)
class FullEpisodeConfig:
    arrival_window_steps: int = 4
    transit_clear_tolerance: float = 1e-6
    max_decision_steps: int = 1000
    max_updates: int = 1000
    max_runtime_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.arrival_window_steps <= 0 or self.max_decision_steps <= 0 or self.max_updates <= 0:
            raise ValueError("episode integer limits must be positive")
        if not math.isfinite(self.transit_clear_tolerance) or self.transit_clear_tolerance < 0:
            raise ValueError("transit_clear_tolerance must be finite and nonnegative")
        if self.max_runtime_seconds is not None and (
            not math.isfinite(self.max_runtime_seconds) or self.max_runtime_seconds <= 0
        ):
            raise ValueError("max_runtime_seconds must be finite and positive when supplied")


@dataclass(frozen=True)
class FullEpisodeResult:
    run_type: str
    episode_number: int
    status: str
    episode: EpisodeState
    summary: dict[str, object]
    step_rows: tuple[dict[str, object], ...]
    update_rows: tuple[dict[str, object], ...]
    waiting_records: tuple[TaskWaitingRecord, ...]
    feature_rows: tuple[dict[str, object], ...]
    solver_rows: tuple[dict[str, object], ...]
    reward_row: dict[str, object]


def _bootstrap(critic: GNNCritic, prepared: PreparedDecision) -> float:
    device = next(critic.parameters()).device
    with torch.no_grad():
        value = critic(
            prepared.policy_observation.node_feature_tensor.to(device=device, dtype=torch.float32),
            prepared.policy_observation.graph.to(device),
        )
    if value.ndim != 0 or not torch.isfinite(value):
        raise RuntimeError("bootstrap Critic value must be a finite scalar")
    return float(value.detach().cpu())


def _prepare(
    episode: EpisodeState,
    scenario: FluidScenario,
    scaler: FeatureScaleConfig,
    graph: IncomingNormalizedGraph,
    window: int,
) -> tuple[PreparedDecision, bool]:
    if episode.prepared_decision is not None:
        cached = episode.prepared_decision
        episode.prepared_decision = None
        return cached, True
    return prepare_decision(
        episode.physical_state,
        episode.task_queue,
        episode.past_arrivals,
        window,
        scaler,
        graph,
        scenario.decision_period_min,
    ), False


def _prepare_next(
    episode: EpisodeState,
    scenario: FluidScenario,
    scaler: FeatureScaleConfig,
    graph: IncomingNormalizedGraph,
    window: int,
) -> PreparedDecision:
    if episode.prepared_decision is None:
        episode.prepared_decision = prepare_decision(
            episode.physical_state,
            episode.task_queue,
            episode.past_arrivals,
            window,
            scaler,
            graph,
            scenario.decision_period_min,
        )
    return episode.prepared_decision


def _update_row(result: A2CUpdateResult, episode: int, update_index: int, elapsed: float) -> dict[str, object]:
    return {
        "episode": episode,
        "update_index": update_index,
        "rollout_length": result.rollout_length,
        "bootstrap_value": result.bootstrap_value,
        "actor_loss": result.actor_loss,
        "critic_loss": result.critic_loss,
        "entropy_mean": result.entropy_mean,
        "return_mean": result.return_mean,
        "advantage_mean": result.advantage_mean,
        "actor_grad_norm_before_clip": result.actor_grad_norm_before_clip,
        "critic_grad_norm_before_clip": result.critic_grad_norm_before_clip,
        "actor_parameters_changed": result.actor_parameters_changed,
        "critic_parameters_changed": result.critic_parameters_changed,
        "update_time_seconds": elapsed,
    }


def run_full_episode(
    *,
    scenario: FluidScenario,
    graph: IncomingNormalizedGraph,
    feature_scaler: FeatureScaleConfig,
    reward_config: RewardConfig,
    initial_idle_distribution: np.ndarray,
    fleet_size: float,
    config: FullEpisodeConfig,
    run_type: str,
    episode_number: int,
    actor: DirichletActor | None = None,
    critic: GNNCritic | None = None,
    updater: A2CUpdater | None = None,
) -> FullEpisodeResult:
    """Run one complete fixed-policy or stochastic-training episode.

    ``served demand`` is matching/transport-start count, not microscopic unload
    completion.  Fixed policies perform no network forward or optimizer update.
    """

    if run_type not in {
        "hold_current", "uniform", "single_episode_dry_run", "training",
        "deterministic_evaluation",
    }:
        raise ValueError("unknown run_type")
    training = run_type in {"single_episode_dry_run", "training"}
    deterministic_evaluation = run_type == "deterministic_evaluation"
    if training:
        if actor is None or critic is None or updater is None:
            raise ValueError("training runs require actor, critic, and updater")
        if updater.actor is not actor or updater.critic is not critic:
            raise ValueError("updater must own the exact actor and critic used by the runner")
        actor.train()
        critic.train()
        buffer = NStepRolloutBuffer(updater.config.n_steps)
    else:
        if updater is not None:
            raise ValueError("fixed baselines must not receive an updater")
        if deterministic_evaluation:
            if actor is None or critic is None:
                raise ValueError("deterministic evaluation requires actor and critic")
            actor.eval()
            critic.eval()
        buffer = None

    episode = initialize_episode_state(fleet_size, initial_idle_distribution, episode_number)
    task_lookup = {task.task_id: task for task in scenario.task_records}
    feature_accumulator = FeatureAccumulator(feature_scaler)
    rewards: list[RewardBreakdown] = []
    waiting_records: list[TaskWaitingRecord] = []
    step_rows: list[dict[str, object]] = []
    update_rows: list[dict[str, object]] = []
    solver_rows: list[dict[str, object]] = []
    state_totals: list[tuple[float, float, float]] = []
    waiting_queues: list[float] = []
    total_loaded_vehicle_minutes = 0.0
    total_rebalancing_vehicle_minutes = 0.0
    total_rebalanced = 0
    max_conservation_error = 0.0
    max_queue_difference = 0.0
    prepare_calls = 0
    cached_uses = 0
    policy_steps = 0
    drain_steps = 0
    matched_by_horizon: int | None = None
    backlog_at_horizon: float | None = None
    first_post_release_matched: int | None = None
    queue_clear_step: int | None = None
    run_started = time.perf_counter()

    while not episode.terminated and not episode.truncated:
        if config.max_runtime_seconds is not None and time.perf_counter() - run_started >= config.max_runtime_seconds:
            episode.truncated = True
            break
        if episode.decision_step >= config.max_decision_steps:
            episode.truncated = True
            break
        if is_drain_mode(episode, scenario.arrivals.number_of_steps):
            drain_started = time.perf_counter()
            execute_drain_step(episode, scenario)
            drain_steps += 1
            state_totals.append((
                float(episode.physical_state.idle.sum()),
                float(episode.physical_state.loaded.sum()),
                float(episode.physical_state.rebalancing.sum()),
            ))
            waiting_queues.append(float(episode.physical_state.backlog.sum()))
            step_rows.append({
                "run_type": run_type, "episode": episode_number,
                "step": episode.decision_step - 1, "mode": "drain", "arrivals": 0,
                "waiting_before": 0, "matched": 0, "waiting_after": 0,
                "scaled_reward": 0.0, "service_component": 0.0,
                "loaded_travel_cost": 0.0, "rebalancing_travel_cost": 0.0,
                "waiting_cost": 0.0, "alpha_min": math.nan, "alpha_max": math.nan,
                "action_sum": math.nan, "state_value": math.nan, "rebalanced": 0,
                "solver_status": "NOT_CALLED_DRAIN", "solver_time_seconds": 0.0,
                "idle_total": state_totals[-1][0], "loaded_total": state_totals[-1][1],
                "rebalancing_total": state_totals[-1][2], "conservation_error": 0.0,
                "queue_backlog_difference": 0.0, "used_cached_prepared_decision": False,
                "step_time_seconds": time.perf_counter() - drain_started,
            })
            if natural_termination_reached(episode, scenario.arrivals.number_of_steps, config.transit_clear_tolerance):
                episode.terminated = True
                if training and buffer is not None and len(buffer):
                    update_started = time.perf_counter()
                    update = updater.update(buffer, 0.0, allow_partial=True)
                    update_rows.append(_update_row(update, episode_number, len(update_rows) + 1, time.perf_counter() - update_started))
            continue

        step_started = time.perf_counter()
        prepared, used_cache = _prepare(
            episode, scenario, feature_scaler, graph, config.arrival_window_steps
        )
        if used_cache:
            cached_uses += 1
        else:
            prepare_calls += 1
        feature_accumulator.add(
            prepared.policy_observation.raw_node_features,
            prepared.policy_observation.normalized_node_features,
        )
        if training or deterministic_evaluation:
            if deterministic_evaluation:
                with torch.no_grad():
                    policy = forward_policy(actor, critic, prepared.policy_observation, deterministic=True)
            else:
                policy = forward_policy(actor, critic, prepared.policy_observation, deterministic=False)
            action_tensor = policy.actor_output.action_proportions
            if training and (action_tensor.requires_grad or action_tensor.grad_fn is not None):
                raise RuntimeError("environment action unexpectedly carries a gradient")
            target = action_tensor.detach().cpu().numpy().copy()
            alpha_min = float(policy.actor_output.concentrations.detach().min().cpu())
            alpha_max = float(policy.actor_output.concentrations.detach().max().cpu())
            action_sum = float(action_tensor.sum().cpu())
            value = float(policy.state_value.detach().cpu())
        else:
            target = (
                hold_current_target(prepared.dispatchable_idle_after_matching)
                if run_type == "hold_current"
                else uniform_target()
            )
            alpha_min = alpha_max = value = math.nan
            action_sum = float(target.sum())

        step = episode.decision_step
        arrivals = scenario.arrivals.arrivals_at(step)
        released = scenario.tasks_released_at(step)
        waiting_before = episode.task_queue.total_waiting_tasks()
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
            episode.fleet_size,
        )
        reward = calculate_reward(
            prepared.matching_plan.matching,
            control.rebalancing_plan.flow_matrix,
            prepared.post_matching_snapshot,
            scenario.tau_minutes,
            scenario.decision_period_min,
            reward_config,
        )
        rewards.append(reward)
        selected = prepared.matching_plan.selected_task_ids
        episode.record_matched_tasks(selected)
        waiting_records.extend(build_waiting_records(
            selected, task_lookup, step, scenario.decision_period_min, run_type, episode_number
        ))
        total_loaded_vehicle_minutes += float(np.sum(
            prepared.matching_plan.matching.astype(np.float64) * scenario.tau_minutes,
            where=np.isfinite(scenario.tau_minutes),
        ))
        total_rebalancing_vehicle_minutes += control.rebalancing_plan.objective_travel_minutes
        total_rebalanced += control.rebalancing_plan.total_rebalanced
        episode.physical_state = control.next_state
        episode.task_queue = control.next_task_queue
        episode.append_past_arrival(arrivals)
        episode.decision_step += 1
        episode.prepared_decision = None
        policy_steps += 1
        terminated = natural_termination_reached(
            episode, scenario.arrivals.number_of_steps, config.transit_clear_tolerance
        )
        episode.terminated = terminated

        if training and buffer is not None:
            buffer.add(RolloutTransition(
                log_prob=policy.actor_output.log_prob,
                entropy=policy.actor_output.entropy,
                state_value=policy.state_value,
                reward=reward.scaled_reward,
                terminated=terminated,
                action_proportions=action_tensor,
                episode_id=episode.episode_id,
                diagnostics={"step": step},
            ))

        queue_difference = float(np.max(np.abs(
            episode.task_queue.count_matrix().astype(np.float64) - episode.physical_state.backlog
        )))
        conservation_error = abs(control.fluid_step_diagnostics.conservation_error)
        max_queue_difference = max(max_queue_difference, queue_difference)
        max_conservation_error = max(max_conservation_error, conservation_error)
        totals = (
            float(episode.physical_state.idle.sum()),
            float(episode.physical_state.loaded.sum()),
            float(episode.physical_state.rebalancing.sum()),
        )
        state_totals.append(totals)
        waiting_after = float(episode.physical_state.backlog.sum())
        waiting_queues.append(waiting_after)
        solver_rows.append({
            "run_type": run_type, "episode": episode_number, "step": step,
            "solver_status": control.rebalancing_plan.solver_status,
            "solver_backend": control.rebalancing_plan.solver_backend,
            "solve_time_seconds": control.rebalancing_plan.solve_time_seconds,
            "total_rebalanced": control.rebalancing_plan.total_rebalanced,
            "objective_travel_minutes": control.rebalancing_plan.objective_travel_minutes,
        })
        step_rows.append({
            "run_type": run_type, "episode": episode_number, "step": step, "mode": "policy",
            "arrivals": int(arrivals.sum()), "waiting_before": waiting_before,
            "matched": prepared.matching_plan.matched_task_count, "waiting_after": waiting_after,
            "scaled_reward": reward.scaled_reward, "service_component": reward.service_component,
            "loaded_travel_cost": reward.loaded_travel_cost,
            "rebalancing_travel_cost": reward.rebalancing_travel_cost,
            "waiting_cost": reward.waiting_cost, "alpha_min": alpha_min, "alpha_max": alpha_max,
            "action_sum": action_sum, "state_value": value,
            "rebalanced": control.rebalancing_plan.total_rebalanced,
            "solver_status": control.rebalancing_plan.solver_status,
            "solver_time_seconds": control.rebalancing_plan.solve_time_seconds,
            "idle_total": totals[0], "loaded_total": totals[1], "rebalancing_total": totals[2],
            "conservation_error": conservation_error,
            "queue_backlog_difference": queue_difference,
            "used_cached_prepared_decision": used_cache,
            "step_time_seconds": time.perf_counter() - step_started,
        })

        if step == scenario.arrivals.number_of_steps - 1:
            matched_by_horizon = len(episode.matched_task_ids)
            backlog_at_horizon = waiting_after
        if step == scenario.arrivals.number_of_steps:
            first_post_release_matched = len(episode.matched_task_ids)
        if queue_clear_step is None and episode.decision_step >= scenario.arrivals.number_of_steps and episode.task_queue.total_waiting_tasks() == 0:
            queue_clear_step = step

        if training and buffer is not None and buffer.ready:
            next_is_drain = is_drain_mode(episode, scenario.arrivals.number_of_steps)
            if terminated or next_is_drain:
                bootstrap = 0.0
                episode.prepared_decision = None
            else:
                next_prepared = _prepare_next(
                    episode, scenario, feature_scaler, graph, config.arrival_window_steps
                )
                prepare_calls += 1
                bootstrap = _bootstrap(critic, next_prepared)
            update_started = time.perf_counter()
            update = updater.update(buffer, bootstrap)
            update_rows.append(_update_row(update, episode_number, len(update_rows) + 1, time.perf_counter() - update_started))
            if len(update_rows) >= config.max_updates and not episode.terminated:
                episode.truncated = True

    if training and buffer is not None and episode.truncated and len(buffer) and len(update_rows) < config.max_updates:
        if is_drain_mode(episode, scenario.arrivals.number_of_steps):
            bootstrap = 0.0
            episode.prepared_decision = None
        else:
            next_prepared = _prepare_next(
                episode, scenario, feature_scaler, graph, config.arrival_window_steps
            )
            prepare_calls += 1
            bootstrap = _bootstrap(critic, next_prepared)
        update_started = time.perf_counter()
        update = updater.update(buffer, bootstrap, allow_partial=True)
        update_rows.append(_update_row(update, episode_number, len(update_rows) + 1, time.perf_counter() - update_started))

    runtime = time.perf_counter() - run_started
    reward_row = aggregate_reward(rewards, run_type, episode_number)
    waiting_stats = waiting_summary(waiting_records)
    states = np.asarray(state_totals, dtype=np.float64)
    solve_times = np.asarray([row["solve_time_seconds"] for row in solver_rows], dtype=np.float64)
    true_scip = [row for row in solver_rows if row["solver_status"] != "OPTIMAL_NO_REBALANCING"]
    update_times = [float(row["update_time_seconds"]) for row in update_rows]
    alpha_values = [
        float(row[key])
        for row in step_rows
        if row["mode"] == "policy"
        for key in ("alpha_min", "alpha_max")
        if math.isfinite(float(row[key]))
    ]
    action_errors = [abs(float(row["action_sum"]) - 1.0) for row in step_rows if row["mode"] == "policy"]
    runtime_limit_hit = config.max_runtime_seconds is not None and runtime >= config.max_runtime_seconds
    status = "completed" if episode.terminated and not episode.truncated else (
        "not_cleared_within_cap" if not training else (
            "runtime_review_required" if runtime_limit_hit else "blocked"
        )
    )
    summary: dict[str, object] = {
        "run_type": run_type, "episode": episode_number, "status": status,
        "terminated": episode.terminated, "truncated": episode.truncated,
        "fleet_size": fleet_size, "total_tasks": scenario.task_count,
        "total_matched": len(episode.matched_task_ids),
        "matched_by_release_horizon": matched_by_horizon if matched_by_horizon is not None else len(episode.matched_task_ids),
        "service_rate_by_release_horizon": (matched_by_horizon if matched_by_horizon is not None else len(episode.matched_task_ids)) / scenario.task_count,
        "backlog_at_release_horizon": backlog_at_horizon if backlog_at_horizon is not None else float(episode.physical_state.backlog.sum()),
        "first_post_release_matched": first_post_release_matched if first_post_release_matched is not None else len(episode.matched_task_ids),
        "queue_clear_step": queue_clear_step if queue_clear_step is not None else "",
        "policy_steps": policy_steps, "drain_steps": drain_steps,
        "total_episode_steps": episode.decision_step,
        "total_episode_minutes": episode.decision_step * scenario.decision_period_min,
        **waiting_stats,
        "average_waiting_queue": float(np.mean(waiting_queues)) if waiting_queues else 0.0,
        "maximum_waiting_queue": float(np.max(waiting_queues)) if waiting_queues else 0.0,
        "average_idle_vehicles": float(np.mean(states[:, 0])) if states.size else fleet_size,
        "average_idle_ratio": float(np.mean(states[:, 0]) / fleet_size) if states.size else 1.0,
        "average_loaded_vehicles": float(np.mean(states[:, 1])) if states.size else 0.0,
        "average_loaded_ratio": float(np.mean(states[:, 1]) / fleet_size) if states.size else 0.0,
        "average_rebalancing_vehicles": float(np.mean(states[:, 2])) if states.size else 0.0,
        "average_rebalancing_ratio": float(np.mean(states[:, 2]) / fleet_size) if states.size else 0.0,
        "maximum_loaded_vehicles": float(np.max(states[:, 1])) if states.size else 0.0,
        "maximum_rebalancing_vehicles": float(np.max(states[:, 2])) if states.size else 0.0,
        "total_rebalanced_vehicles": total_rebalanced,
        "total_rebalancing_vehicle_minutes": total_rebalancing_vehicle_minutes,
        "total_loaded_vehicle_minutes": total_loaded_vehicle_minutes,
        "max_vehicle_conservation_error": max_conservation_error,
        "max_queue_backlog_difference": max_queue_difference,
        "solver_calls": len(true_scip),
        "no_rebalancing_count": sum(row["solver_status"] == "OPTIMAL_NO_REBALANCING" for row in solver_rows),
        "solver_total_time_seconds": float(solve_times.sum()) if solve_times.size else 0.0,
        "solver_mean_time_seconds": float(solve_times.mean()) if solve_times.size else 0.0,
        "solver_p95_time_seconds": float(np.percentile(solve_times, 95)) if solve_times.size else 0.0,
        "solver_max_time_seconds": float(solve_times.max()) if solve_times.size else 0.0,
        "solver_optimal_ratio": sum(str(row["solver_status"]).startswith("OPTIMAL") for row in solver_rows) / len(solver_rows) if solver_rows else 1.0,
        "update_count": len(update_rows),
        "actor_loss_mean": float(np.mean([row["actor_loss"] for row in update_rows])) if update_rows else math.nan,
        "critic_loss_mean": float(np.mean([row["critic_loss"] for row in update_rows])) if update_rows else math.nan,
        "actor_entropy_mean": float(np.mean([row["entropy_mean"] for row in update_rows])) if update_rows else math.nan,
        "return_mean": float(np.mean([row["return_mean"] for row in update_rows])) if update_rows else math.nan,
        "advantage_mean": float(np.mean([row["advantage_mean"] for row in update_rows])) if update_rows else math.nan,
        "actor_grad_norm_mean": float(np.mean([row["actor_grad_norm_before_clip"] for row in update_rows])) if update_rows else math.nan,
        "actor_grad_norm_max": float(np.max([row["actor_grad_norm_before_clip"] for row in update_rows])) if update_rows else math.nan,
        "critic_grad_norm_mean": float(np.mean([row["critic_grad_norm_before_clip"] for row in update_rows])) if update_rows else math.nan,
        "critic_grad_norm_max": float(np.max([row["critic_grad_norm_before_clip"] for row in update_rows])) if update_rows else math.nan,
        "alpha_min": min(alpha_values) if alpha_values else math.nan,
        "alpha_max": max(alpha_values) if alpha_values else math.nan,
        "action_sum_max_error": max(action_errors) if action_errors else 0.0,
        "nonfinite_detected": False,
        "episode_runtime_seconds": runtime,
        "average_policy_step_seconds": runtime / policy_steps if policy_steps else math.nan,
        "average_update_seconds": float(np.mean(update_times)) if update_times else math.nan,
        "prepared_decision_calls": prepare_calls,
        "cached_prepared_decision_uses": cached_uses,
        **{key: value for key, value in reward_row.items() if key not in {"run_type", "episode"}},
    }
    return FullEpisodeResult(
        run_type, episode_number, status, episode, summary, tuple(step_rows), tuple(update_rows),
        tuple(waiting_records), tuple(feature_accumulator.rows(run_type, episode_number)),
        tuple(solver_rows), reward_row,
    )
