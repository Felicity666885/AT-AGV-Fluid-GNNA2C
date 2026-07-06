"""Independent directed-GNN state-value forward network."""

from __future__ import annotations

import torch
from torch import nn

from src.policy.directed_gnn import DirectedGNNEncoder
from src.policy.directed_graph import IncomingNormalizedGraph


class GNNCritic(nn.Module):
    """Estimate ``V(tilde_s_t)`` from ``(16,6)`` or ``(B,16,6)`` input.

    The independent GNN embeddings are globally mean-pooled across 16 nodes.
    A configurable one-hidden-layer MLP outputs a scalar for one state (0-D
    tensor) or a ``(B,)`` tensor for a batch.
    """

    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 64,
        embedding_dim: int = 64,
        num_layers: int = 2,
        value_hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        if value_hidden_dim <= 0:
            raise ValueError("value_hidden_dim must be positive")
        self.encoder = DirectedGNNEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=embedding_dim,
            num_layers=num_layers,
        )
        self.value_mlp = nn.Sequential(
            nn.Linear(embedding_dim, value_hidden_dim),
            nn.ReLU(),
            nn.Linear(value_hidden_dim, 1),
        )
        for module in self.value_mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def encode_and_pool(
        self, x: torch.Tensor, graph: IncomingNormalizedGraph
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return node embeddings and their global mean representation."""

        embeddings = self.encoder(x, graph)
        pooled = embeddings.mean(dim=-2)
        return embeddings, pooled

    def forward(self, x: torch.Tensor, graph: IncomingNormalizedGraph) -> torch.Tensor:
        """Return one finite value scalar or a finite ``(B,)`` value vector."""

        _, pooled = self.encode_and_pool(x, graph)
        value = self.value_mlp(pooled).squeeze(-1)
        if not torch.isfinite(value).all():
            raise RuntimeError("GNNCritic produced a nonfinite value")
        return value

