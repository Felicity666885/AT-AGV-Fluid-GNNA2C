"""Episode-safe n-step rollout storage retaining only policy/value graphs."""

from __future__ import annotations

from dataclasses import dataclass
import copy
import math
import operator
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import torch


@dataclass(frozen=True)
class RolloutTransition:
    """One scalar Actor-Critic transition and detached environment action."""

    log_prob: torch.Tensor
    entropy: torch.Tensor
    state_value: torch.Tensor
    reward: float
    terminated: bool
    action_proportions: torch.Tensor
    episode_id: int | str
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        for label, tensor in (
            ("log_prob", self.log_prob),
            ("entropy", self.entropy),
            ("state_value", self.state_value),
        ):
            if not isinstance(tensor, torch.Tensor) or tensor.ndim != 0:
                raise ValueError(f"{label} must be a scalar torch.Tensor")
            if not torch.isfinite(tensor):
                raise ValueError(f"{label} must be finite")
        reward = float(self.reward)
        if not math.isfinite(reward):
            raise ValueError("reward must be finite")
        action = self.action_proportions.detach().clone()
        if action.ndim != 1 or not torch.isfinite(action).all() or torch.any(action < 0.0):
            raise ValueError("action_proportions must be a finite nonnegative vector")
        if action.requires_grad or action.grad_fn is not None:
            raise RuntimeError("stored environment action must not retain a gradient graph")
        if not torch.allclose(action.sum(), torch.tensor(1.0, device=action.device), atol=1e-6):
            raise ValueError("action_proportions must lie on the probability simplex")
        object.__setattr__(self, "reward", reward)
        object.__setattr__(self, "terminated", bool(self.terminated))
        object.__setattr__(self, "action_proportions", action)
        object.__setattr__(self, "diagnostics", MappingProxyType(copy.deepcopy(dict(self.diagnostics))))


@dataclass(frozen=True)
class ReturnAdvantageBatch:
    """Stacked rollout tensors; returns are detached, advantages retain value graph."""

    returns: torch.Tensor
    advantages: torch.Tensor
    values: torch.Tensor
    log_probs: torch.Tensor
    entropies: torch.Tensor


class NStepRolloutBuffer:
    """Bounded rollout from exactly one episode, cleared after each update."""

    def __init__(self, n_steps: int) -> None:
        if isinstance(n_steps, bool):
            raise TypeError("n_steps must be a positive integer")
        try:
            count = operator.index(n_steps)
        except TypeError as exc:
            raise TypeError("n_steps must be a positive integer") from exc
        if count <= 0:
            raise ValueError("n_steps must be positive")
        self.n_steps = count
        self._transitions: list[RolloutTransition] = []
        self._episode_id: int | str | None = None

    def __len__(self) -> int:
        return len(self._transitions)

    @property
    def transitions(self) -> tuple[RolloutTransition, ...]:
        return tuple(self._transitions)

    @property
    def is_full(self) -> bool:
        return len(self) == self.n_steps

    @property
    def ready(self) -> bool:
        return bool(self._transitions) and (self.is_full or self._transitions[-1].terminated)

    def add(self, transition: RolloutTransition) -> None:
        """Append one transition without mixing episodes or exceeding capacity."""

        if not isinstance(transition, RolloutTransition):
            raise TypeError("transition must be RolloutTransition")
        if self.is_full:
            raise RuntimeError("rollout buffer is full and must be updated/cleared")
        if self._transitions and self._transitions[-1].terminated:
            raise RuntimeError("cannot append after a naturally terminated transition")
        if self._episode_id is None:
            self._episode_id = transition.episode_id
        elif transition.episode_id != self._episode_id:
            raise RuntimeError("cannot mix different episodes in one rollout")
        self._transitions.append(transition)

    def clear(self) -> None:
        """Release retained computation graphs after one update."""

        self._transitions.clear()
        self._episode_id = None


def compute_returns_and_advantages(
    transitions: Sequence[RolloutTransition],
    gamma: float,
    bootstrap_value: float | torch.Tensor,
) -> ReturnAdvantageBatch:
    """Compute reverse n-step returns and ``R_t - V_t`` without clipping.

    Bootstrap is detached.  A terminated transition resets its return to its
    immediate reward, so later values and bootstrap cannot propagate across a
    natural episode boundary.
    """

    if not math.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be finite and in [0, 1]")
    items = tuple(transitions)
    if not items:
        raise ValueError("at least one rollout transition is required")
    device = items[0].state_value.device
    dtype = items[0].state_value.dtype
    for item in items:
        if item.state_value.device != device or item.state_value.dtype != dtype:
            raise ValueError("all rollout tensors must share value device and dtype")
    bootstrap = torch.as_tensor(bootstrap_value, dtype=dtype, device=device).detach()
    if bootstrap.ndim != 0 or not torch.isfinite(bootstrap):
        raise ValueError("bootstrap_value must be a finite scalar")
    if items[-1].terminated:
        bootstrap = torch.zeros((), dtype=dtype, device=device)

    running = bootstrap
    reversed_returns: list[torch.Tensor] = []
    for item in reversed(items):
        reward = torch.tensor(item.reward, dtype=dtype, device=device)
        if item.terminated:
            running = reward
        else:
            running = reward + gamma * running
        reversed_returns.append(running)
    returns = torch.stack(list(reversed(reversed_returns))).detach()
    values = torch.stack([item.state_value for item in items])
    log_probs = torch.stack([item.log_prob for item in items])
    entropies = torch.stack([item.entropy for item in items])
    advantages = returns - values
    for label, tensor in (
        ("returns", returns),
        ("advantages", advantages),
        ("values", values),
        ("log_probs", log_probs),
        ("entropies", entropies),
    ):
        if not torch.isfinite(tensor).all():
            raise RuntimeError(f"computed {label} is nonfinite")
    return ReturnAdvantageBatch(returns, advantages, values, log_probs, entropies)

