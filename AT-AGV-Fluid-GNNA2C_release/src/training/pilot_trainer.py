"""Strictly bounded pilot A2C and deterministic evaluation interfaces."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch

from src.control.fluid_control_step import (
    execute_fluid_control_step_from_precomputed_matching,
)
from src.data.scenario_loader import FluidScenario
from src.envs.fluid_dynamics import fluid_step
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
from src.training.prepared_decision import PreparedDecision, prepare_decision
from src.training.reward import RewardBreakdown, RewardConfig, calculate_reward
from src.training.rollout_buffer import NStepRolloutBuffer, RolloutTransition


@dataclass(frozen=True)
class PilotConfig:
    """Externally supplied bounds and causal/drain settings for a pilot run."""

    arrival_window_steps: int
    max_decision_steps: int
    max_updates: int
    transit_clear_tolerance: float

    def __post_init__(self) -> None:
        for label in ("arrival_window_steps", "max_decision_steps", "max_updates"):
            value = getattr(self, label)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{label} must be a positive integer")
        if not math.isfinite(self.transit_clear_tolerance) or self.transit_clear_tolerance < 0:
            raise ValueError("transit_clear_tolerance must be finite and nonnegative")


@dataclass(frozen=True)
class PilotStepRecord:
    """One environment-step diagnostic; no tensors or retained graphs."""

    step: int
    waiting_before: int
    matched: int
    reward: RewardBreakdown
    alpha_min: float
    alpha_max: float
    action_sum: float
    state_value: float
    rebalanced: int
    solver_status: str
    idle_total: float
    loaded_total: float
    rebalancing_total: float
    conservation_error: float
    queue_backlog_difference: float
    used_cached_prepared_decision: bool


@dataclass(frozen=True)
class PilotRunResult:
    """Bounded pilot outcome and update diagnostics, not a performance result."""

    episode: EpisodeState
    steps: tuple[PilotStepRecord, ...]
    updates: tuple[A2CUpdateResult, ...]
    prepared_decision_calls: int
    drain_steps: int
    zero_bootstrap_due_to_termination_count: int = 0
    zero_bootstrap_due_to_drain_count: int = 0
    critic_bootstrap_count: int = 0


@dataclass(frozen=True)
class EvaluationResult:
    """Forward-only deterministic evaluation-interface diagnostics."""

    episode: EpisodeState
    steps_run: int
    prepared_decision_calls: int
    actor_parameters_unchanged: bool
    critic_parameters_unchanged: bool
    max_conservation_error: float
    max_queue_backlog_difference: float


def _critic_bootstrap(critic: GNNCritic, prepared: PreparedDecision) -> float:
    """Evaluate only the next ``tilde_s`` Critic under no_grad."""

    device = next(critic.parameters()).device
    features = prepared.policy_observation.node_feature_tensor.to(device=device, dtype=torch.float32)
    graph = prepared.policy_observation.graph.to(device)
    with torch.no_grad():
        value = critic(features, graph)
    if value.ndim != 0 or not torch.isfinite(value):
        raise RuntimeError("bootstrap Critic value must be a finite scalar")
    return float(value.cpu())


def _get_or_prepare(
    episode: EpisodeState,
    scenario: FluidScenario,
    feature_scaler: FeatureScaleConfig,
    graph: IncomingNormalizedGraph,
    arrival_window_steps: int,
) -> tuple[PreparedDecision, bool]:
    if episode.prepared_decision is not None:
        prepared = episode.prepared_decision
        episode.prepared_decision = None
        return prepared, True
    return (
        prepare_decision(
            episode.physical_state,
            episode.task_queue,
            episode.past_arrivals,
            arrival_window_steps,
            feature_scaler,
            graph,
            scenario.decision_period_min,
        ),
        False,
    )


def _prepare_next_for_bootstrap(
    episode: EpisodeState,
    scenario: FluidScenario,
    feature_scaler: FeatureScaleConfig,
    graph: IncomingNormalizedGraph,
    arrival_window_steps: int,
) -> PreparedDecision:
    if episode.prepared_decision is None:
        episode.prepared_decision = prepare_decision(
            episode.physical_state,
            episode.task_queue,
            episode.past_arrivals,
            arrival_window_steps,
            feature_scaler,
            graph,
            scenario.decision_period_min,
        )
    return episode.prepared_decision


def execute_drain_step(episode: EpisodeState, scenario: FluidScenario) -> None:
    """Advance one no-policy, zero-control drain period in place.

    No Actor, Critic, matching, rebalancing solver, reward, or rollout operation
    occurs here.  Existing transit fluid decays through the stage-1 dynamics.
    """

    if not is_drain_mode(episode, scenario.arrivals.number_of_steps):
        raise ValueError("execute_drain_step requires an episode already in drain mode")
    zero = np.zeros((16, 16), dtype=np.float64)
    fluid_result = fluid_step(
        episode.physical_state,
        zero,
        zero,
        zero,
        scenario.tau_minutes,
        scenario.decision_period_min,
        scenario.e_reb_mask,
        episode.fleet_size,
    )
    episode.physical_state = fluid_result.next_state
    episode.decision_step += 1
    episode.prepared_decision = None


def run_pilot_training(
    scenario: FluidScenario,
    actor: DirichletActor,
    critic: GNNCritic,
    graph: IncomingNormalizedGraph,
    feature_scaler: FeatureScaleConfig,
    reward_config: RewardConfig,
    updater: A2CUpdater,
    initial_idle_distribution: np.ndarray,
    fleet_size: float,
    pilot_config: PilotConfig,
    episode_id: int | str = "pilot",
) -> PilotRunResult:
    """Run a bounded stochastic pilot with cached next-decision bootstrap context."""

    if updater.actor is not actor:
        raise ValueError("updater.actor must be the exact Actor object passed to run_pilot_training")
    if updater.critic is not critic:
        raise ValueError("updater.critic must be the exact Critic object passed to run_pilot_training")
    episode = initialize_episode_state(fleet_size, initial_idle_distribution, episode_id)
    buffer = NStepRolloutBuffer(updater.config.n_steps)
    step_records: list[PilotStepRecord] = []
    update_records: list[A2CUpdateResult] = []
    prepare_calls = 0
    drain_steps = 0
    zero_bootstrap_termination = 0
    zero_bootstrap_drain = 0
    critic_bootstrap_count = 0
    actor.train()
    critic.train()

    while not episode.terminated and not episode.truncated:
        if episode.decision_step >= pilot_config.max_decision_steps:
            episode.truncated = True
            break
        if is_drain_mode(episode, scenario.arrivals.number_of_steps):
            execute_drain_step(episode, scenario)
            drain_steps += 1
            if natural_termination_reached(
                episode,
                scenario.arrivals.number_of_steps,
                pilot_config.transit_clear_tolerance,
            ):
                episode.terminated = True
                if len(buffer):
                    zero_bootstrap_drain += 1
                    update_records.append(updater.update(buffer, 0.0, allow_partial=True))
            continue

        prepared, used_cache = _get_or_prepare(
            episode,
            scenario,
            feature_scaler,
            graph,
            pilot_config.arrival_window_steps,
        )
        if not used_cache:
            prepare_calls += 1
        policy = forward_policy(actor, critic, prepared.policy_observation, deterministic=False)
        action = policy.actor_output.action_proportions
        if action.requires_grad or action.grad_fn is not None:
            raise RuntimeError("stochastic environment action must not carry gradients")
        action_numpy = action.detach().cpu().numpy().copy()
        step = episode.decision_step
        arrivals = scenario.arrivals.arrivals_at(step)
        released = scenario.tasks_released_at(step)
        waiting_before = episode.task_queue.total_waiting_tasks()
        control = execute_fluid_control_step_from_precomputed_matching(
            state=episode.physical_state,
            task_queue=episode.task_queue,
            matching_plan=prepared.matching_plan,
            post_matching_snapshot=prepared.post_matching_snapshot,
            arrivals_matrix=arrivals,
            released_tasks=released,
            target_proportions=action_numpy,
            tau=scenario.tau_minutes,
            delta_t=scenario.decision_period_min,
            e_reb_mask=scenario.e_reb_mask,
            fleet_size=episode.fleet_size,
        )
        reward = calculate_reward(
            prepared.matching_plan.matching,
            control.rebalancing_plan.flow_matrix,
            prepared.post_matching_snapshot,
            scenario.tau_minutes,
            scenario.decision_period_min,
            reward_config,
        )
        episode.record_matched_tasks(prepared.matching_plan.selected_task_ids)
        episode.physical_state = control.next_state
        episode.task_queue = control.next_task_queue
        episode.append_past_arrival(arrivals)
        episode.decision_step += 1
        episode.prepared_decision = None
        terminated = natural_termination_reached(
            episode,
            scenario.arrivals.number_of_steps,
            pilot_config.transit_clear_tolerance,
        )
        episode.terminated = terminated
        buffer.add(
            RolloutTransition(
                log_prob=policy.actor_output.log_prob,
                entropy=policy.actor_output.entropy,
                state_value=policy.state_value,
                reward=reward.scaled_reward,
                terminated=terminated,
                action_proportions=action,
                episode_id=episode.episode_id,
                diagnostics={"step": step},
            )
        )
        queue_difference = float(
            np.max(
                np.abs(
                    episode.task_queue.count_matrix().astype(np.float64)
                    - episode.physical_state.backlog
                )
            )
        )
        step_records.append(
            PilotStepRecord(
                step=step,
                waiting_before=waiting_before,
                matched=prepared.matching_plan.matched_task_count,
                reward=reward,
                alpha_min=float(policy.actor_output.concentrations.detach().min().cpu()),
                alpha_max=float(policy.actor_output.concentrations.detach().max().cpu()),
                action_sum=float(action.sum().cpu()),
                state_value=float(policy.state_value.detach().cpu()),
                rebalanced=control.rebalancing_plan.total_rebalanced,
                solver_status=control.rebalancing_plan.solver_status,
                idle_total=float(episode.physical_state.idle.sum()),
                loaded_total=float(episode.physical_state.loaded.sum()),
                rebalancing_total=float(episode.physical_state.rebalancing.sum()),
                conservation_error=control.fluid_step_diagnostics.conservation_error,
                queue_backlog_difference=queue_difference,
                used_cached_prepared_decision=used_cache,
            )
        )

        if buffer.ready:
            next_is_drain = is_drain_mode(episode, scenario.arrivals.number_of_steps)
            if terminated or next_is_drain:
                bootstrap = 0.0
                episode.prepared_decision = None
                if terminated:
                    zero_bootstrap_termination += 1
                else:
                    zero_bootstrap_drain += 1
            else:
                next_prepared = _prepare_next_for_bootstrap(
                    episode,
                    scenario,
                    feature_scaler,
                    graph,
                    pilot_config.arrival_window_steps,
                )
                prepare_calls += 1
                bootstrap = _critic_bootstrap(critic, next_prepared)
                critic_bootstrap_count += 1
            update_records.append(updater.update(buffer, bootstrap))
            if len(update_records) >= pilot_config.max_updates and not episode.terminated:
                episode.truncated = True

    if episode.truncated and len(buffer) and len(update_records) < pilot_config.max_updates:
        next_is_drain = is_drain_mode(episode, scenario.arrivals.number_of_steps)
        if next_is_drain:
            bootstrap = 0.0
            episode.prepared_decision = None
            zero_bootstrap_drain += 1
        else:
            next_prepared = _prepare_next_for_bootstrap(
                episode,
                scenario,
                feature_scaler,
                graph,
                pilot_config.arrival_window_steps,
            )
            prepare_calls += 1
            bootstrap = _critic_bootstrap(critic, next_prepared)
            critic_bootstrap_count += 1
        update_records.append(
            updater.update(buffer, bootstrap, allow_partial=True)
        )
    return PilotRunResult(
        episode=episode,
        steps=tuple(step_records),
        updates=tuple(update_records),
        prepared_decision_calls=prepare_calls,
        drain_steps=drain_steps,
        zero_bootstrap_due_to_termination_count=zero_bootstrap_termination,
        zero_bootstrap_due_to_drain_count=zero_bootstrap_drain,
        critic_bootstrap_count=critic_bootstrap_count,
    )


def run_deterministic_evaluation(
    scenario: FluidScenario,
    actor: DirichletActor,
    critic: GNNCritic,
    graph: IncomingNormalizedGraph,
    feature_scaler: FeatureScaleConfig,
    initial_idle_distribution: np.ndarray,
    fleet_size: float,
    arrival_window_steps: int,
    max_steps: int,
) -> EvaluationResult:
    """Run a short no-grad deterministic interface check with no optimizer step."""

    episode = initialize_episode_state(fleet_size, initial_idle_distribution, "evaluation")
    actor.eval()
    critic.eval()
    actor_before = [parameter.detach().clone() for parameter in actor.parameters()]
    critic_before = [parameter.detach().clone() for parameter in critic.parameters()]
    prepare_calls = 0
    maximum_error = 0.0
    maximum_queue_difference = 0.0
    with torch.no_grad():
        for _ in range(max_steps):
            prepared = prepare_decision(
                episode.physical_state,
                episode.task_queue,
                episode.past_arrivals,
                arrival_window_steps,
                feature_scaler,
                graph,
                scenario.decision_period_min,
            )
            prepare_calls += 1
            first = forward_policy(actor, critic, prepared.policy_observation, deterministic=True)
            second = forward_policy(actor, critic, prepared.policy_observation, deterministic=True)
            if not torch.equal(
                first.actor_output.action_proportions, second.actor_output.action_proportions
            ):
                raise RuntimeError("deterministic Actor action changed for identical input")
            action = first.actor_output.action_proportions.cpu().numpy().copy()
            step = episode.decision_step
            control = execute_fluid_control_step_from_precomputed_matching(
                episode.physical_state,
                episode.task_queue,
                prepared.matching_plan,
                prepared.post_matching_snapshot,
                scenario.arrivals.arrivals_at(step),
                scenario.tasks_released_at(step),
                action,
                scenario.tau_minutes,
                scenario.decision_period_min,
                scenario.e_reb_mask,
                episode.fleet_size,
            )
            episode.record_matched_tasks(prepared.matching_plan.selected_task_ids)
            episode.physical_state = control.next_state
            episode.task_queue = control.next_task_queue
            episode.append_past_arrival(scenario.arrivals.arrivals_at(step))
            episode.decision_step += 1
            maximum_error = max(
                maximum_error, abs(control.fluid_step_diagnostics.conservation_error)
            )
            maximum_queue_difference = max(
                maximum_queue_difference,
                float(
                    np.max(
                        np.abs(
                            episode.task_queue.count_matrix().astype(np.float64)
                            - episode.physical_state.backlog
                        )
                    )
                ),
            )
    actor_unchanged = all(
        torch.equal(before, after) for before, after in zip(actor_before, actor.parameters())
    )
    critic_unchanged = all(
        torch.equal(before, after) for before, after in zip(critic_before, critic.parameters())
    )
    return EvaluationResult(
        episode,
        max_steps,
        prepare_calls,
        actor_unchanged,
        critic_unchanged,
        maximum_error,
        maximum_queue_difference,
    )
