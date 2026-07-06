"""Independent directed-GNN Actor with a 16-dimensional Dirichlet policy."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.distributions import Dirichlet
from torch.nn import functional as functional

from src.policy.directed_gnn import DirectedGNNEncoder
from src.policy.directed_graph import IncomingNormalizedGraph


@dataclass(frozen=True)
class ActorOutput:
    """Dirichlet concentrations, simplex action, density terms, and node embeddings."""

    concentrations: torch.Tensor
    action_proportions: torch.Tensor
    log_prob: torch.Tensor
    entropy: torch.Tensor
    deterministic: bool
    node_embeddings: torch.Tensor


class DirichletActor(nn.Module):
    """Map ``(16,6)`` or ``(B,16,6)`` observations to a Dirichlet policy.

    One shared scalar head is applied to each node embedding.  Concentrations
    are ``softplus(logit) + alpha_min``.  Deterministic actions are distribution
    means; stochastic actions use non-reparameterized ``Dirichlet.sample()``.
    The sampled action has no gradient path back to concentrations, while its
    ``log_prob`` and entropy remain differentiable through the distribution
    parameters.  This is the likelihood-ratio interface needed because integer
    rounding, rebalancing ILP, and fluid environment transitions are external
    and non-differentiable.
    """

    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 64,
        embedding_dim: int = 64,
        num_layers: int = 2,
        alpha_min: float = 1e-4,
    ) -> None:
        super().__init__()
        if not math.isfinite(alpha_min) or alpha_min <= 0.0:
            raise ValueError("alpha_min must be finite and strictly positive")
        self.alpha_min = float(alpha_min)
        self.encoder = DirectedGNNEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=embedding_dim,
            num_layers=num_layers,
        )
        self.actor_head = nn.Linear(embedding_dim, 1)
        nn.init.xavier_uniform_(self.actor_head.weight)
        nn.init.zeros_(self.actor_head.bias)

    def forward(
        self,
        x: torch.Tensor,
        graph: IncomingNormalizedGraph,
        deterministic: bool = False,
    ) -> ActorOutput:
        """Return Actor output for one state or a batch, without any parameter update."""

        embeddings = self.encoder(x, graph)
        concentrations = functional.softplus(self.actor_head(embeddings).squeeze(-1)) + self.alpha_min
        if not torch.isfinite(concentrations).all() or not torch.all(concentrations > 0.0):
            raise RuntimeError("Dirichlet concentrations must be finite and strictly positive")
        distribution = Dirichlet(concentrations)
        if deterministic:
            action = concentrations / concentrations.sum(dim=-1, keepdim=True)
        else:
            # The environment action is fixed for score-function policy gradients.
            # Future A2C uses -log_prob * detached_advantage; no gradient traverses
            # the downstream rounding, integer program, or fluid transition.
            action = distribution.sample()
            if action.requires_grad or action.grad_fn is not None:
                raise RuntimeError("stochastic Dirichlet action must be gradient-detached")
        if not torch.isfinite(action).all() or torch.any(action < 0.0):
            raise RuntimeError("Dirichlet action must be finite and nonnegative")
        sums = action.sum(dim=-1)
        if not torch.allclose(sums, torch.ones_like(sums), atol=1e-6, rtol=1e-6):
            raise RuntimeError("Dirichlet action does not lie on the probability simplex")
        log_prob = distribution.log_prob(action)
        entropy = distribution.entropy()
        if not torch.isfinite(log_prob).all() or not torch.isfinite(entropy).all():
            raise RuntimeError("Dirichlet log_prob or entropy is nonfinite")
        return ActorOutput(
            concentrations=concentrations,
            action_proportions=action,
            log_prob=log_prob,
            entropy=entropy,
            deterministic=bool(deterministic),
            node_embeddings=embeddings,
        )
