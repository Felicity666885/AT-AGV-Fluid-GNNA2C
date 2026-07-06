"""Independent n-step A2C losses, Adam optimizers, and clipped updates."""

from __future__ import annotations

from dataclasses import dataclass
import math
import operator

import torch
from torch import nn
from torch.nn import functional as functional

from src.training.rollout_buffer import (
    NStepRolloutBuffer,
    ReturnAdvantageBatch,
    compute_returns_and_advantages,
)


@dataclass(frozen=True)
class A2CLossConfig:
    """Externally supplied n-step discount, entropy, and gradient limits."""

    gamma: float
    n_steps: int
    entropy_coef: float
    max_grad_norm: float

    def __post_init__(self) -> None:
        gamma = float(self.gamma)
        entropy = float(self.entropy_coef)
        maximum = float(self.max_grad_norm)
        if not math.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be finite and in [0, 1]")
        if isinstance(self.n_steps, bool):
            raise TypeError("n_steps must be a positive integer")
        try:
            steps = operator.index(self.n_steps)
        except TypeError as exc:
            raise TypeError("n_steps must be a positive integer") from exc
        if steps <= 0:
            raise ValueError("n_steps must be positive")
        if not math.isfinite(entropy) or entropy < 0.0:
            raise ValueError("entropy_coef must be finite and nonnegative")
        if not math.isfinite(maximum) or maximum <= 0.0:
            raise ValueError("max_grad_norm must be finite and positive")
        object.__setattr__(self, "gamma", gamma)
        object.__setattr__(self, "n_steps", steps)
        object.__setattr__(self, "entropy_coef", entropy)
        object.__setattr__(self, "max_grad_norm", maximum)


@dataclass(frozen=True)
class A2CLosses:
    """Finite scalar Actor/Critic losses and their rollout tensor batch."""

    actor_loss: torch.Tensor
    critic_loss: torch.Tensor
    batch: ReturnAdvantageBatch


@dataclass(frozen=True)
class A2CUpdateResult:
    """Scalar update diagnostics including pre-clip gradient norms."""

    actor_loss: float
    critic_loss: float
    entropy_mean: float
    return_mean: float
    advantage_mean: float
    actor_grad_norm_before_clip: float
    critic_grad_norm_before_clip: float
    rollout_length: int
    bootstrap_value: float
    actor_parameters_changed: bool
    critic_parameters_changed: bool


def compute_a2c_losses(
    buffer: NStepRolloutBuffer,
    bootstrap_value: float | torch.Tensor,
    config: A2CLossConfig,
) -> A2CLosses:
    """Build likelihood-ratio Actor loss and SmoothL1 Critic loss."""

    batch = compute_returns_and_advantages(
        buffer.transitions, config.gamma, bootstrap_value
    )
    actor_loss = -(batch.log_probs * batch.advantages.detach()).mean()
    actor_loss = actor_loss - config.entropy_coef * batch.entropies.mean()
    critic_loss = functional.smooth_l1_loss(batch.values, batch.returns.detach())
    if actor_loss.ndim != 0 or critic_loss.ndim != 0:
        raise RuntimeError("A2C losses must be scalar tensors")
    if not torch.isfinite(actor_loss) or not torch.isfinite(critic_loss):
        raise RuntimeError("A2C loss is nonfinite")
    return A2CLosses(actor_loss, critic_loss, batch)


class A2CUpdater:
    """Own independent Adam optimizers and consume each rollout exactly once."""

    def __init__(
        self,
        actor: nn.Module,
        critic: nn.Module,
        config: A2CLossConfig,
        actor_lr: float,
        critic_lr: float,
    ) -> None:
        for label, value in (("actor_lr", actor_lr), ("critic_lr", critic_lr)):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{label} must be finite and positive")
        actor_ids = {id(parameter) for parameter in actor.parameters()}
        critic_ids = {id(parameter) for parameter in critic.parameters()}
        if not actor_ids or not critic_ids or actor_ids & critic_ids:
            raise ValueError("Actor and Critic must have nonempty disjoint parameter sets")
        self.actor = actor
        self.critic = critic
        self.config = config
        self.actor_optimizer = torch.optim.Adam(actor.parameters(), lr=float(actor_lr))
        self.critic_optimizer = torch.optim.Adam(critic.parameters(), lr=float(critic_lr))
        actor_optimizer_ids = {
            id(parameter)
            for group in self.actor_optimizer.param_groups
            for parameter in group["params"]
        }
        critic_optimizer_ids = {
            id(parameter)
            for group in self.critic_optimizer.param_groups
            for parameter in group["params"]
        }
        if actor_optimizer_ids & critic_optimizer_ids:
            raise RuntimeError("Actor and Critic Adam parameter sets overlap")

    @staticmethod
    def _validate_gradients(module: nn.Module, label: str) -> None:
        gradients = [parameter.grad for parameter in module.parameters() if parameter.grad is not None]
        if not gradients:
            raise RuntimeError(f"{label} has no gradients")
        if not all(torch.isfinite(gradient).all() for gradient in gradients):
            raise RuntimeError(f"{label} contains nonfinite gradients")

    def update(
        self,
        buffer: NStepRolloutBuffer,
        bootstrap_value: float | torch.Tensor,
        allow_partial: bool = False,
    ) -> A2CUpdateResult:
        """Consume one ready rollout, clip gradients separately, step, and clear."""

        if len(buffer) == 0:
            raise RuntimeError("cannot update from an empty or already consumed rollout")
        if buffer.n_steps != self.config.n_steps:
            raise ValueError("buffer n_steps does not match A2CLossConfig")
        if not buffer.ready and not allow_partial:
            raise RuntimeError("rollout is neither full nor naturally terminated")
        rollout_length = len(buffer)
        bootstrap = float(torch.as_tensor(bootstrap_value).detach().cpu().item())
        losses = compute_a2c_losses(buffer, bootstrap_value, self.config)
        actor_before = [parameter.detach().clone() for parameter in self.actor.parameters()]
        critic_before = [parameter.detach().clone() for parameter in self.critic.parameters()]

        self.actor_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        losses.actor_loss.backward()
        losses.critic_loss.backward()
        self._validate_gradients(self.actor, "Actor")
        self._validate_gradients(self.critic, "Critic")
        actor_norm = torch.nn.utils.clip_grad_norm_(
            self.actor.parameters(), self.config.max_grad_norm
        )
        critic_norm = torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(), self.config.max_grad_norm
        )
        if not torch.isfinite(actor_norm) or not torch.isfinite(critic_norm):
            raise RuntimeError("gradient norm is nonfinite")
        self._validate_gradients(self.actor, "clipped Actor")
        self._validate_gradients(self.critic, "clipped Critic")
        self.actor_optimizer.step()
        self.critic_optimizer.step()

        actor_changed = any(
            not torch.equal(before, after)
            for before, after in zip(actor_before, self.actor.parameters())
        )
        critic_changed = any(
            not torch.equal(before, after)
            for before, after in zip(critic_before, self.critic.parameters())
        )
        for label, module in (("Actor", self.actor), ("Critic", self.critic)):
            if not all(torch.isfinite(parameter).all() for parameter in module.parameters()):
                raise RuntimeError(f"{label} parameter became nonfinite")
        result = A2CUpdateResult(
            actor_loss=float(losses.actor_loss.detach().cpu()),
            critic_loss=float(losses.critic_loss.detach().cpu()),
            entropy_mean=float(losses.batch.entropies.detach().mean().cpu()),
            return_mean=float(losses.batch.returns.mean().cpu()),
            advantage_mean=float(losses.batch.advantages.detach().mean().cpu()),
            actor_grad_norm_before_clip=float(actor_norm.detach().cpu()),
            critic_grad_norm_before_clip=float(critic_norm.detach().cpu()),
            rollout_length=rollout_length,
            bootstrap_value=bootstrap,
            actor_parameters_changed=actor_changed,
            critic_parameters_changed=critic_changed,
        )
        if not all(
            math.isfinite(value)
            for value in (
                result.actor_loss,
                result.critic_loss,
                result.entropy_mean,
                result.return_mean,
                result.advantage_mean,
                result.actor_grad_norm_before_clip,
                result.critic_grad_norm_before_clip,
            )
        ):
            raise RuntimeError("update diagnostics contain a nonfinite value")
        buffer.clear()
        return result

