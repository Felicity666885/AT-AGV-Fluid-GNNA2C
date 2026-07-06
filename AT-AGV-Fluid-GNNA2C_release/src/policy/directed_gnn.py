"""Native-PyTorch directed incoming-message graph convolutions and encoder."""

from __future__ import annotations

import torch
from torch import nn

from src.policy.directed_graph import IncomingNormalizedGraph


class DirectedGraphConv(nn.Module):
    """Directed graph convolution for ``(N,F)`` or ``(B,N,F)`` input.

    For every destination ``v``, the output before activation is
    ``W_self h_v + sum_(u->v) a_uv W_neigh h_u``.  Self and neighbor transforms
    use independent parameters; only incoming edges are aggregated.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activation: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim and output_dim must be positive")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.self_linear = nn.Linear(self.input_dim, self.output_dim)
        self.neighbor_linear = nn.Linear(self.input_dim, self.output_dim)
        self.activation = nn.Identity() if activation is None else activation
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Use explicit Xavier-uniform weights and zero biases."""

        nn.init.xavier_uniform_(self.self_linear.weight)
        nn.init.xavier_uniform_(self.neighbor_linear.weight)
        nn.init.zeros_(self.self_linear.bias)
        nn.init.zeros_(self.neighbor_linear.bias)

    def forward(self, x: torch.Tensor, graph: IncomingNormalizedGraph) -> torch.Tensor:
        """Return ``(N,O)`` or ``(B,N,O)`` on the input tensor's device."""

        if x.ndim not in (2, 3):
            raise ValueError("x must have shape (N,F) or (B,N,F)")
        if x.shape[-2] != graph.num_nodes or x.shape[-1] != self.input_dim:
            raise ValueError(
                f"x must end in ({graph.num_nodes}, {self.input_dim}), got {tuple(x.shape)}"
            )
        if not torch.is_floating_point(x) or not torch.isfinite(x).all():
            raise ValueError("x must be a finite floating-point tensor")
        edge_index = graph.edge_index.to(device=x.device, dtype=torch.long)
        edge_weight = graph.edge_weight.to(device=x.device, dtype=x.dtype)
        source, destination = edge_index
        self_part = self.self_linear(x)
        neighbor_embeddings = self.neighbor_linear(x)
        if x.ndim == 2:
            messages = neighbor_embeddings.index_select(0, source) * edge_weight.unsqueeze(-1)
            aggregated = torch.zeros_like(self_part)
            aggregated.index_add_(0, destination, messages)
        else:
            messages = neighbor_embeddings.index_select(1, source) * edge_weight.view(1, -1, 1)
            aggregated = torch.zeros_like(self_part)
            aggregated.index_add_(1, destination, messages)
        output = self.activation(self_part + aggregated)
        if not torch.isfinite(output).all():
            raise RuntimeError("DirectedGraphConv produced a nonfinite output")
        return output


class DirectedGNNEncoder(nn.Module):
    """Configurable ReLU stack mapping ``(...,16,input_dim)`` to output_dim."""

    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 64,
        output_dim: int = 64,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0 or output_dim <= 0:
            raise ValueError("all encoder dimensions must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.num_layers = int(num_layers)
        dimensions = [self.input_dim]
        if self.num_layers > 1:
            dimensions.extend([self.hidden_dim] * (self.num_layers - 1))
        dimensions.append(self.output_dim)
        self.layers = nn.ModuleList(
            DirectedGraphConv(dimensions[i], dimensions[i + 1], nn.ReLU())
            for i in range(self.num_layers)
        )

    def forward(self, x: torch.Tensor, graph: IncomingNormalizedGraph) -> torch.Tensor:
        """Encode one state ``(16,F)`` or a batch ``(B,16,F)``."""

        hidden = x
        for layer in self.layers:
            hidden = layer(hidden, graph)
        return hidden
