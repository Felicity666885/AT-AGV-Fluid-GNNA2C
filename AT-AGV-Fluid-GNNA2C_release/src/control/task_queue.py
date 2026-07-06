"""Immutable task-level FCFS ledger synchronized with aggregate backlog Q."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

from src.data.scenario_loader import TaskRecord
from src.envs.fluid_state import FluidState


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def task_order_key(task: TaskRecord) -> tuple[float, int]:
    """Return the deterministic FCFS key: release minute, then source index."""

    return task.release_minute, task.source_index


@dataclass(frozen=True)
class TaskQueueLedger:
    """Task identities and release order supporting aggregate physical backlog.

    The ledger is auxiliary bookkeeping, not an additional physical state.  It
    must agree exactly (within an explicit floating tolerance) with
    ``FluidState.backlog``.  All update methods return a new ledger.
    """

    _tasks: tuple[TaskRecord, ...] = ()

    def __post_init__(self) -> None:
        tasks = tuple(sorted(tuple(self._tasks), key=task_order_key))
        if any(not isinstance(task, TaskRecord) for task in tasks):
            raise TypeError("TaskQueueLedger accepts only TaskRecord objects")
        task_ids = [task.task_id for task in tasks]
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("TaskQueueLedger contains duplicate task_id values")
        source_indices = [task.source_index for task in tasks]
        if len(set(source_indices)) != len(source_indices):
            raise ValueError("TaskQueueLedger contains duplicate source_index values")
        for task in tasks:
            if not np.isfinite(task.release_minute) or task.release_minute < 0.0:
                raise ValueError("task release_minute must be finite and nonnegative")
            if task.release_step < 0:
                raise ValueError("task release_step must be nonnegative")
            if not (0 <= task.origin < FluidState.NUM_REGIONS):
                raise ValueError("task origin is outside the 16-region system")
            if not (0 <= task.destination < FluidState.NUM_REGIONS):
                raise ValueError("task destination is outside the 16-region system")
            if task.origin == task.destination:
                raise ValueError("self-loop tasks are not allowed")
        object.__setattr__(self, "_tasks", tasks)

    @property
    def tasks(self) -> tuple[TaskRecord, ...]:
        """Return the immutable task tuple in deterministic FCFS order."""

        return self._tasks

    @property
    def task_ids(self) -> tuple[int, ...]:
        """Return task ids in ledger order."""

        return tuple(task.task_id for task in self._tasks)

    def copy(self) -> "TaskQueueLedger":
        """Return an independent ledger with the same immutable records."""

        return TaskQueueLedger(tuple(self._tasks))

    def count_matrix(self) -> IntArray:
        """Aggregate waiting tasks to a new 16x16 integer OD count matrix."""

        counts = np.zeros((FluidState.NUM_REGIONS, FluidState.NUM_REGIONS), dtype=np.int64)
        for task in self._tasks:
            counts[task.origin, task.destination] += 1
        return counts

    def total_waiting_tasks(self) -> int:
        """Return the number of task identities currently waiting."""

        return len(self._tasks)

    def tasks_at_origin(self, origin: int) -> tuple[TaskRecord, ...]:
        """Return an FCFS-ordered immutable tuple for one origin region."""

        if isinstance(origin, bool) or not isinstance(origin, (int, np.integer)):
            raise TypeError("origin must be an integer")
        if not 0 <= int(origin) < FluidState.NUM_REGIONS:
            raise ValueError("origin is outside the 16-region system")
        return tuple(task for task in self._tasks if task.origin == int(origin))

    def add_released_tasks(self, tasks: Sequence[TaskRecord]) -> "TaskQueueLedger":
        """Return a ledger containing the old queue and newly released records."""

        additions = tuple(tasks)
        return TaskQueueLedger(self._tasks + additions)

    def remove_selected_tasks(self, task_ids: Iterable[int]) -> "TaskQueueLedger":
        """Return a ledger without exactly the selected unique task ids."""

        selected_tuple = tuple(int(task_id) for task_id in task_ids)
        if len(set(selected_tuple)) != len(selected_tuple):
            raise ValueError("selected task ids must be unique")
        selected = set(selected_tuple)
        existing = set(self.task_ids)
        missing = selected - existing
        if missing:
            raise ValueError(f"cannot remove unknown task ids: {sorted(missing)[:5]}")
        return TaskQueueLedger(tuple(task for task in self._tasks if task.task_id not in selected))

    def validate_against_backlog(
        self, backlog: FloatArray, tolerance: float = 1e-10
    ) -> None:
        """Raise if task identity counts do not equal aggregate physical backlog."""

        values = np.asarray(backlog, dtype=np.float64)
        expected_shape = (FluidState.NUM_REGIONS, FluidState.NUM_REGIONS)
        if values.shape != expected_shape:
            raise ValueError(f"backlog must have shape {expected_shape}, got {values.shape}")
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("tolerance must be finite and nonnegative")
        if not np.all(np.isfinite(values)):
            raise ValueError("backlog must contain only finite values")
        counts = self.count_matrix().astype(np.float64)
        difference = values - counts
        if np.any(np.abs(difference) > tolerance):
            index = np.argwhere(np.abs(difference) > tolerance)[0]
            i, j = int(index[0]), int(index[1])
            raise ValueError(
                f"task queue/backlog mismatch at ({i}, {j}): "
                f"ledger={counts[i, j]}, backlog={values[i, j]}"
            )

