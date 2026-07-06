"""Causal six-column node features for the post-matching fluid snapshot."""

from __future__ import annotations

from dataclasses import dataclass, fields
import math
import operator
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from src.envs.fluid_state import FluidState


FloatArray = NDArray[np.float64]

NODE_FEATURE_NAMES = (
    "idle_after_matching",
    "remaining_outgoing_backlog",
    "matched_inbound",
    "loaded_inbound",
    "rebalancing_inbound",
    "causal_outgoing_arrival_rate",
)


@dataclass(frozen=True)
class RawNodeFeatures:
    """A copied ``(16, 6)`` float64 feature matrix and fixed column metadata."""

    values: FloatArray
    feature_names: tuple[str, ...] = NODE_FEATURE_NAMES

    def __post_init__(self) -> None:
        values = np.array(self.values, dtype=np.float64, copy=True)
        if values.shape != (FluidState.NUM_REGIONS, len(NODE_FEATURE_NAMES)):
            raise ValueError(f"raw node features must have shape (16, 6), got {values.shape}")
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("raw node features must be finite and nonnegative")
        if tuple(self.feature_names) != NODE_FEATURE_NAMES:
            raise ValueError("node feature column order does not match NODE_FEATURE_NAMES")
        values.setflags(write=False)
        object.__setattr__(self, "values", values)


@dataclass(frozen=True)
class FeatureScaleConfig:
    """Fixed positive scales for the six node-feature columns.

    These values are external configuration.  They are never estimated from a
    current state, updated online, or inferred from future episode data.
    """

    idle_scale: float
    backlog_scale: float
    matched_inbound_scale: float
    loaded_inbound_scale: float
    rebalancing_inbound_scale: float
    arrival_rate_scale: float

    def __post_init__(self) -> None:
        for field in fields(self):
            value = float(getattr(self, field.name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field.name} must be finite and strictly positive")
            object.__setattr__(self, field.name, value)

    def as_array(self) -> FloatArray:
        """Return scales in exactly ``NODE_FEATURE_NAMES`` column order."""

        return np.asarray(
            (
                self.idle_scale,
                self.backlog_scale,
                self.matched_inbound_scale,
                self.loaded_inbound_scale,
                self.rebalancing_inbound_scale,
                self.arrival_rate_scale,
            ),
            dtype=np.float64,
        )


def estimate_causal_origin_arrival_rate(
    past_arrivals: Sequence[FloatArray] | FloatArray,
    delta_t: float,
    window_steps: int,
) -> FloatArray:
    """Estimate origin release rates from completed historical periods only.

    ``past_arrivals`` must contain exactly ``A(0), ..., A(t-1)`` and must not
    contain ``A(t)``.  The result is tasks per minute for each origin, computed
    over the last ``window_steps`` available periods.  No files are read.
    """

    if not math.isfinite(delta_t) or delta_t <= 0.0:
        raise ValueError("delta_t must be finite and positive")
    if isinstance(window_steps, bool):
        raise TypeError("window_steps must be a positive integer")
    try:
        window = operator.index(window_steps)
    except TypeError as exc:
        raise TypeError("window_steps must be a positive integer") from exc
    if window <= 0:
        raise ValueError("window_steps must be positive")

    if isinstance(past_arrivals, np.ndarray):
        history = np.array(past_arrivals, dtype=np.float64, copy=True)
        if history.size == 0:
            return np.zeros(FluidState.NUM_REGIONS, dtype=np.float64)
        if history.ndim == 2:
            history = history[np.newaxis, :, :]
    else:
        copied = [np.array(arrival, dtype=np.float64, copy=True) for arrival in past_arrivals]
        if not copied:
            return np.zeros(FluidState.NUM_REGIONS, dtype=np.float64)
        history = np.stack(copied, axis=0)
    expected_tail = (FluidState.NUM_REGIONS, FluidState.NUM_REGIONS)
    if history.ndim != 3 or history.shape[1:] != expected_tail:
        raise ValueError(f"past_arrivals must have shape (T, 16, 16), got {history.shape}")
    if not np.all(np.isfinite(history)) or np.any(history < 0.0):
        raise ValueError("past_arrivals must be finite and nonnegative")
    if np.any(np.diagonal(history, axis1=1, axis2=2) != 0.0):
        raise ValueError("past_arrivals must have zero self-loop entries")

    used = history[-min(window, history.shape[0]) :]
    rates = used.sum(axis=(0, 2), dtype=np.float64) / (used.shape[0] * delta_t)
    return np.asarray(rates, dtype=np.float64)


def build_raw_node_features(
    post_matching_state: FluidState,
    matching: FloatArray,
    causal_arrival_rate: FloatArray,
) -> RawNodeFeatures:
    """Construct six features directly from ``tilde_s_t``, ``M(t)``, and history.

    The function does not reapply matching, add current/future arrivals, or
    mutate any input.  ``matching`` is a period task quantity, while arrival
    rates are tasks per minute.
    """

    post_matching_state.validate()
    matching_array = np.array(matching, dtype=np.float64, copy=True)
    if matching_array.shape != (FluidState.NUM_REGIONS, FluidState.NUM_REGIONS):
        raise ValueError(f"matching must have shape (16, 16), got {matching_array.shape}")
    if not np.all(np.isfinite(matching_array)) or np.any(matching_array < 0.0):
        raise ValueError("matching must be finite and nonnegative")
    if np.any(np.diag(matching_array) != 0.0):
        raise ValueError("matching diagonal must be zero")
    arrival_rate = np.array(causal_arrival_rate, dtype=np.float64, copy=True)
    if arrival_rate.shape != (FluidState.NUM_REGIONS,):
        raise ValueError(f"causal_arrival_rate must have shape (16,), got {arrival_rate.shape}")
    if not np.all(np.isfinite(arrival_rate)) or np.any(arrival_rate < 0.0):
        raise ValueError("causal_arrival_rate must be finite and nonnegative")

    values = np.column_stack(
        (
            post_matching_state.idle,
            post_matching_state.backlog.sum(axis=1, dtype=np.float64),
            matching_array.sum(axis=0, dtype=np.float64),
            post_matching_state.loaded.sum(axis=0, dtype=np.float64),
            post_matching_state.rebalancing.sum(axis=0, dtype=np.float64),
            arrival_rate,
        )
    )
    return RawNodeFeatures(values=values)


def normalize_node_features(
    raw_features: RawNodeFeatures | FloatArray,
    scales: FeatureScaleConfig,
) -> FloatArray:
    """Divide each feature column by one fixed configured scale without clipping."""

    source = raw_features.values if isinstance(raw_features, RawNodeFeatures) else raw_features
    values = np.array(source, dtype=np.float64, copy=True)
    if values.shape != (FluidState.NUM_REGIONS, len(NODE_FEATURE_NAMES)):
        raise ValueError(f"raw_features must have shape (16, 6), got {values.shape}")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("raw_features must be finite and nonnegative")
    normalized = values / scales.as_array()[np.newaxis, :]
    if not np.all(np.isfinite(normalized)) or np.any(normalized < 0.0):
        raise ValueError("normalized node features must be finite and nonnegative")
    return normalized

