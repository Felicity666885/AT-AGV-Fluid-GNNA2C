"""One-FCFS-call preparation of the post-matching policy decision context."""

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
from src.control.task_queue import TaskQueueLedger
from src.envs.fluid_dynamics import build_post_matching_snapshot
from src.envs.fluid_state import FluidState
from src.policy.directed_graph import IncomingNormalizedGraph
from src.policy.node_features import FeatureScaleConfig, estimate_causal_origin_arrival_rate
from src.policy.policy_forward import PolicyObservation, prepare_policy_observation


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class PreparedDecision:
    """Cached deterministic matching and the exact ``tilde_s_t`` observation."""

    matching_plan: MatchingPlan
    post_matching_snapshot: FluidState
    causal_arrival_rate: FloatArray
    policy_observation: PolicyObservation
    waiting_after_matching: float
    dispatchable_idle_after_matching: IntArray


def prepare_decision(
    state: FluidState,
    task_queue: TaskQueueLedger,
    past_arrivals: Sequence[FloatArray] | FloatArray,
    arrival_window_steps: int,
    feature_scaler: FeatureScaleConfig,
    graph: IncomingNormalizedGraph,
    delta_t: float,
    tolerance: float = DEFAULT_INTEGER_TOLERANCE,
) -> PreparedDecision:
    """Call FCFS once, build ``tilde_s_t``, causal rates, and policy observation.

    ``past_arrivals`` must end at ``A(t-1)``.  The function does not accept or
    read ``A(t)``, run either network, solve rebalancing, or advance dynamics.
    """

    state.validate(tolerance)
    task_queue.validate_against_backlog(state.backlog, tolerance)
    matching_plan = solve_fcfs_matching(state, task_queue, tolerance)
    snapshot = build_post_matching_snapshot(state, matching_plan.matching)
    causal_rate = estimate_causal_origin_arrival_rate(
        past_arrivals, delta_t, arrival_window_steps
    )
    observation = prepare_policy_observation(
        snapshot, matching_plan.matching, causal_rate, feature_scaler, graph
    )
    dispatchable = dispatchable_integer_idle(snapshot.idle, tolerance)
    causal_rate = np.array(causal_rate, dtype=np.float64, copy=True)
    dispatchable = np.array(dispatchable, dtype=np.int64, copy=True)
    causal_rate.setflags(write=False)
    dispatchable.setflags(write=False)
    return PreparedDecision(
        matching_plan=matching_plan,
        post_matching_snapshot=snapshot,
        causal_arrival_rate=causal_rate,
        policy_observation=observation,
        waiting_after_matching=float(snapshot.backlog.sum(dtype=np.float64)),
        dispatchable_idle_after_matching=dispatchable,
    )

