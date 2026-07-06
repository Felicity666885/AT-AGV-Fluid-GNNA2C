"""Physical state container for the 16-region deterministic fluid model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass
class FluidState:
    """Physical state ``s_t = {Q, Z_idle, Z_loaded, Y}`` in vehicle/task units.

    ``backlog[i,j]`` is :math:`Q_{ij}`; ``idle[i]`` is
    :math:`Z_i^{idle}`; ``loaded[i,j]`` is :math:`Z_{ij}^{loaded}`; and
    ``rebalancing[i,j]`` is :math:`Y_{ij}`.  Fluid quantities are float64 and
    may be fractional.  Self-loop entries of all OD matrices must remain zero.
    """

    backlog: FloatArray
    idle: FloatArray
    loaded: FloatArray
    rebalancing: FloatArray

    NUM_REGIONS: ClassVar[int] = 16
    DEFAULT_TOLERANCE: ClassVar[float] = 1e-10

    def __post_init__(self) -> None:
        self.backlog = np.array(self.backlog, dtype=np.float64, copy=True)
        self.idle = np.array(self.idle, dtype=np.float64, copy=True)
        self.loaded = np.array(self.loaded, dtype=np.float64, copy=True)
        self.rebalancing = np.array(self.rebalancing, dtype=np.float64, copy=True)

    def copy(self) -> "FluidState":
        """Return a deep copy whose arrays do not alias the current state."""

        return FluidState(self.backlog, self.idle, self.loaded, self.rebalancing)

    def total_vehicle_mass(self) -> float:
        """Return idle plus loaded and rebalancing in-transit vehicle mass."""

        return float(
            self.idle.sum(dtype=np.float64)
            + self.loaded.sum(dtype=np.float64)
            + self.rebalancing.sum(dtype=np.float64)
        )

    def minimum_value(self) -> float:
        """Return the minimum value among all four state components."""

        return float(
            min(
                np.min(self.backlog),
                np.min(self.idle),
                np.min(self.loaded),
                np.min(self.rebalancing),
            )
        )

    def validate(self, tolerance: float = DEFAULT_TOLERANCE) -> None:
        """Raise ``ValueError`` for wrong shape, nonfinite, negative, or diagonal state."""

        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("tolerance must be finite and nonnegative")
        matrix_shape = (self.NUM_REGIONS, self.NUM_REGIONS)
        if self.backlog.shape != matrix_shape:
            raise ValueError(f"backlog must have shape {matrix_shape}, got {self.backlog.shape}")
        if self.idle.shape != (self.NUM_REGIONS,):
            raise ValueError(f"idle must have shape ({self.NUM_REGIONS},), got {self.idle.shape}")
        if self.loaded.shape != matrix_shape:
            raise ValueError(f"loaded must have shape {matrix_shape}, got {self.loaded.shape}")
        if self.rebalancing.shape != matrix_shape:
            raise ValueError(
                f"rebalancing must have shape {matrix_shape}, got {self.rebalancing.shape}"
            )
        for label, values in (
            ("backlog", self.backlog),
            ("idle", self.idle),
            ("loaded", self.loaded),
            ("rebalancing", self.rebalancing),
        ):
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{label} must contain only finite values")
            minimum = float(np.min(values))
            if minimum < -tolerance:
                raise ValueError(f"{label} contains a negative value {minimum}")
        for label, values in (
            ("backlog", self.backlog),
            ("loaded", self.loaded),
            ("rebalancing", self.rebalancing),
        ):
            if np.any(np.abs(np.diag(values)) > tolerance):
                raise ValueError(f"{label} diagonal must be zero")

