"""Validation and destination-indegree normalization for directed E_G edges."""

from __future__ import annotations

from dataclasses import dataclass
import operator

import numpy as np
import torch


@dataclass(frozen=True)
class IncomingNormalizedGraph:
    """Directed edges and incoming-normalized weights shared by a batch."""

    edge_index: torch.Tensor
    edge_weight: torch.Tensor
    num_nodes: int

    def __post_init__(self) -> None:
        edge_index = self.edge_index.detach().clone().to(dtype=torch.long, device="cpu")
        edge_weight = self.edge_weight.detach().clone().to(dtype=torch.float32, device="cpu")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape (2, E)")
        if edge_weight.shape != (edge_index.shape[1],):
            raise ValueError("edge_weight must have shape (E,)")
        if not torch.isfinite(edge_weight).all() or not torch.all(edge_weight > 0):
            raise ValueError("edge weights must be finite and strictly positive")
        object.__setattr__(self, "edge_index", edge_index)
        object.__setattr__(self, "edge_weight", edge_weight)

    def to(self, device: torch.device | str) -> "IncomingNormalizedGraph":
        """Return a graph copy on an explicit device."""

        result = object.__new__(IncomingNormalizedGraph)
        object.__setattr__(
            result, "edge_index", self.edge_index.to(device=device, dtype=torch.long).clone()
        )
        object.__setattr__(
            result, "edge_weight", self.edge_weight.to(device=device, dtype=torch.float32).clone()
        )
        object.__setattr__(result, "num_nodes", self.num_nodes)
        return result


def build_incoming_normalized_graph(
    e_g_edge_index: np.ndarray | torch.Tensor,
    num_nodes: int = 16,
) -> IncomingNormalizedGraph:
    """Keep each ``u->v`` direction and assign weight ``1 / indegree(v)``.

    No self-loops or reverse edges are added.  A node with zero indegree simply
    has no neighbor message and retains its representation through self-linear
    terms in the graph-convolution layer.
    """

    if isinstance(num_nodes, bool):
        raise TypeError("num_nodes must be a positive integer")
    try:
        node_count = operator.index(num_nodes)
    except TypeError as exc:
        raise TypeError("num_nodes must be a positive integer") from exc
    if node_count <= 0:
        raise ValueError("num_nodes must be positive")
    if isinstance(e_g_edge_index, torch.Tensor):
        edge_index = e_g_edge_index.detach().clone().to(dtype=torch.long, device="cpu")
    else:
        raw = np.array(e_g_edge_index, copy=True)
        if not np.issubdtype(raw.dtype, np.integer):
            raise ValueError("edge_index must contain integer node indices")
        edge_index = torch.as_tensor(raw, dtype=torch.long).clone()
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must have shape (2, E), got {tuple(edge_index.shape)}")
    if edge_index.shape[1] == 0:
        raise ValueError("edge_index must contain at least one directed edge")
    if torch.any(edge_index < 0) or torch.any(edge_index >= node_count):
        raise ValueError("edge_index contains an out-of-range node")
    source, destination = edge_index
    if torch.any(source == destination):
        raise ValueError("E_G must not contain self-loops")
    pairs = [tuple(pair) for pair in edge_index.t().tolist()]
    if len(set(pairs)) != len(pairs):
        raise ValueError("E_G must not contain duplicate directed edges")

    indegree = torch.bincount(destination, minlength=node_count).to(dtype=torch.float32)
    edge_weight = torch.reciprocal(indegree[destination])
    return IncomingNormalizedGraph(edge_index=edge_index, edge_weight=edge_weight, num_nodes=node_count)
