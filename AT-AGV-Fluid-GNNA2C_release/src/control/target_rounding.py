"""Vehicle-total-preserving target rounding and fixed test target policies."""

from __future__ import annotations

import operator

import numpy as np
from numpy.typing import NDArray

from src.envs.fluid_state import FluidState


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def normalize_proportions(proportions: FloatArray) -> FloatArray:
    """Validate and normalize a nonnegative 16-region target vector."""

    values = np.array(proportions, dtype=np.float64, copy=True)
    if values.shape != (FluidState.NUM_REGIONS,):
        raise ValueError(
            f"target proportions must have shape ({FluidState.NUM_REGIONS},), got {values.shape}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("target proportions must contain only finite values")
    if np.any(values < 0.0):
        raise ValueError("target proportions must be nonnegative")
    total = float(values.sum(dtype=np.float64))
    if total <= 0.0:
        raise ValueError("target proportions must have a positive sum")
    return values / total


def largest_remainder_round(proportions: FloatArray, total: int) -> IntArray:
    """Round normalized target shares to nonnegative integers summing to total.

    Fractional ties are resolved by ascending region index.  ``total=0``
    returns an all-zero vector after shape, finite, and nonnegative checks; no
    vehicle can be created by ordinary rounding.
    """

    if isinstance(total, bool):
        raise TypeError("total must be a nonnegative integer")
    try:
        integer_total = operator.index(total)
    except TypeError as exc:
        raise TypeError("total must be a nonnegative integer") from exc
    if integer_total < 0:
        raise ValueError("total must be nonnegative")

    values = np.array(proportions, dtype=np.float64, copy=True)
    if values.shape != (FluidState.NUM_REGIONS,):
        raise ValueError(
            f"target proportions must have shape ({FluidState.NUM_REGIONS},), got {values.shape}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("target proportions must contain only finite values")
    if np.any(values < 0.0):
        raise ValueError("target proportions must be nonnegative")
    if integer_total == 0:
        return np.zeros(FluidState.NUM_REGIONS, dtype=np.int64)

    normalized = normalize_proportions(values)
    raw = normalized * integer_total
    rounded = np.floor(raw).astype(np.int64)
    remaining = integer_total - int(rounded.sum())
    fractions = raw - rounded
    order = sorted(range(FluidState.NUM_REGIONS), key=lambda i: (-fractions[i], i))
    for region in order[:remaining]:
        rounded[region] += 1
    if np.any(rounded < 0) or int(rounded.sum()) != integer_total:
        raise RuntimeError("largest-remainder rounding failed to preserve the requested total")
    return rounded


def hold_current_target(available_idle: IntArray) -> FloatArray:
    """Test policy returning the current integer idle distribution as shares."""

    available = np.asarray(available_idle)
    if available.shape != (FluidState.NUM_REGIONS,):
        raise ValueError(f"available_idle must have shape ({FluidState.NUM_REGIONS},)")
    if not np.issubdtype(available.dtype, np.integer) or np.any(available < 0):
        raise ValueError("available_idle must contain nonnegative integers")
    total = int(available.sum())
    if total == 0:
        return uniform_target()
    return available.astype(np.float64) / total


def uniform_target() -> FloatArray:
    """Test policy returning equal target shares for all 16 regions."""

    return np.full(FluidState.NUM_REGIONS, 1.0 / FluidState.NUM_REGIONS, dtype=np.float64)


def fixed_target(proportions: FloatArray) -> FloatArray:
    """Validate and normalize caller-provided fixed test target shares."""

    return normalize_proportions(proportions)

