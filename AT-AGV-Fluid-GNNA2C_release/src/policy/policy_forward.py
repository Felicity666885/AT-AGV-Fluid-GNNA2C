"""Pure observation preparation and Actor-Critic forward-only integration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
import torch

from src.envs.fluid_state import FluidState
from src.policy.actor import ActorOutput, DirichletActor
from src.policy.critic import GNNCritic
from src.policy.directed_graph import IncomingNormalizedGraph
from src.policy.node_features import (
    FeatureScaleConfig,
    NODE_FEATURE_NAMES,
    build_raw_node_features,
    normalize_node_features,
)


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PolicyObservation:
    """NumPy diagnostics plus float32 Torch features and the fixed directed graph."""

    raw_node_features: FloatArray
    normalized_node_features: FloatArray
    node_feature_tensor: torch.Tensor
    edge_index: torch.Tensor
    edge_weight: torch.Tensor
    graph: IncomingNormalizedGraph
    feature_names: tuple[str, ...]

    def __post_init__(self) -> None:
        raw = np.array(self.raw_node_features, dtype=np.float64, copy=True)
        normalized = np.array(self.normalized_node_features, dtype=np.float64, copy=True)
        if raw.shape != (16, 6) or normalized.shape != (16, 6):
            raise ValueError("policy observation features must have shape (16, 6)")
        if not np.all(np.isfinite(raw)) or not np.all(np.isfinite(normalized)):
            raise ValueError("policy observation features must be finite")
        if np.any(raw < 0.0) or np.any(normalized < 0.0):
            raise ValueError("policy observation features must be nonnegative")
        if tuple(self.feature_names) != NODE_FEATURE_NAMES:
            raise ValueError("policy observation feature_names have an invalid order")
        raw.setflags(write=False)
        normalized.setflags(write=False)
        tensor = self.node_feature_tensor.detach().clone().to(dtype=torch.float32, device="cpu")
        if tensor.shape != (16, 6) or not torch.isfinite(tensor).all():
            raise ValueError("node_feature_tensor must be finite float32 with shape (16, 6)")
        object.__setattr__(self, "raw_node_features", raw)
        object.__setattr__(self, "normalized_node_features", normalized)
        object.__setattr__(self, "node_feature_tensor", tensor)
        object.__setattr__(self, "edge_index", self.edge_index.detach().clone().to(torch.long))
        object.__setattr__(self, "edge_weight", self.edge_weight.detach().clone().to(torch.float32))


@dataclass(frozen=True)
class ObservationDiagnostics:
    """Finite extrema used to audit each forward observation."""

    raw_min: float
    raw_max: float
    normalized_min: float
    normalized_max: float


@dataclass(frozen=True)
class PolicyForwardResult:
    """Actor output, Critic value, and non-training observation diagnostics."""

    actor_output: ActorOutput
    state_value: torch.Tensor
    observation_diagnostics: ObservationDiagnostics


def prepare_policy_observation(
    post_matching_state: FluidState,
    matching: FloatArray,
    causal_arrival_rate: FloatArray,
    feature_scaler: FeatureScaleConfig,
    graph: IncomingNormalizedGraph,
) -> PolicyObservation:
    """Build a copied post-matching observation without reading files or future demand."""

    raw = build_raw_node_features(post_matching_state, matching, causal_arrival_rate)
    normalized = normalize_node_features(raw, feature_scaler)
    tensor = torch.as_tensor(np.array(normalized, copy=True), dtype=torch.float32)
    return PolicyObservation(
        raw_node_features=raw.values,
        normalized_node_features=normalized,
        node_feature_tensor=tensor,
        edge_index=graph.edge_index,
        edge_weight=graph.edge_weight,
        graph=graph,
        feature_names=NODE_FEATURE_NAMES,
    )


def forward_policy(
    actor: DirichletActor,
    critic: GNNCritic,
    observation: PolicyObservation,
    deterministic: bool,
) -> PolicyForwardResult:
    """Run Actor and Critic on the same observation with no update or hidden cache."""

    actor_parameter_ids = {id(parameter) for parameter in actor.parameters()}
    critic_parameter_ids = {id(parameter) for parameter in critic.parameters()}
    if actor_parameter_ids & critic_parameter_ids:
        raise ValueError("Actor and Critic must not share Parameter objects")
    actor_device = next(actor.parameters()).device
    critic_device = next(critic.parameters()).device
    if actor_device != critic_device:
        raise ValueError("Actor and Critic must be on the same explicit device")
    features = observation.node_feature_tensor.to(device=actor_device, dtype=torch.float32)
    graph = observation.graph.to(actor_device)
    actor_output = actor(features, graph, deterministic=deterministic)
    state_value = critic(features, graph)
    diagnostics = ObservationDiagnostics(
        raw_min=float(np.min(observation.raw_node_features)),
        raw_max=float(np.max(observation.raw_node_features)),
        normalized_min=float(np.min(observation.normalized_node_features)),
        normalized_max=float(np.max(observation.normalized_node_features)),
    )
    return PolicyForwardResult(actor_output, state_value, diagnostics)
