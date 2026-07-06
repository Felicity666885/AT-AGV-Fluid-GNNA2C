"""Strict loader for the reviewed MSAIV610A V2.1 fluid scenario.

The loader performs validation only.  It does not choose a fleet size, invent
travel times, or read demand from any vessel other than the one in the file.
Travel-time units and the decision period are minutes.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]

EXPECTED_STATUS = "ready_for_fluid_environment_development"
EXPECTED_VESSEL = "MSAIV610A"
EXPECTED_REGIONS = 16
EXPECTED_E_G = 34
EXPECTED_E_REB = 240
EXPECTED_TASKS = 3207
EXPECTED_DECISION_PERIOD_MIN = 15.0


@dataclass(frozen=True)
class Region:
    """One scenario region with its integer index, display name, and type."""

    id: int
    name: str
    type: str


@dataclass(frozen=True)
class TaskRecord:
    """One immutable MSAIV610A task ordered by release time and source row."""

    task_id: int
    source_index: int
    release_minute: float
    release_step: int
    origin: int
    destination: int


@dataclass(frozen=True)
class TaskArrivalSeries:
    """Dense non-anticipative task-arrival tensor indexed by decision step.

    ``arrivals_at(t)`` is the demand released during ``[t, t + delta_t)``.
    It is added after the decision at step ``t`` and therefore cannot be
    matched until the next decision epoch.
    """

    _values: FloatArray
    decision_period_min: float
    last_release_step: int

    def __post_init__(self) -> None:
        values = np.array(self._values, dtype=np.float64, copy=True)
        if values.ndim != 3 or values.shape[1:] != (EXPECTED_REGIONS, EXPECTED_REGIONS):
            raise ValueError(f"arrival tensor must have shape (steps, 16, 16), got {values.shape}")
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("arrival tensor must contain finite nonnegative values")
        if np.any(np.diagonal(values, axis1=1, axis2=2) != 0.0):
            raise ValueError("self-loop task arrivals are not allowed")
        values.setflags(write=False)
        object.__setattr__(self, "_values", values)

    @property
    def number_of_steps(self) -> int:
        """Number of decision intervals containing the release horizon."""

        return int(self._values.shape[0])

    @property
    def total_arrivals(self) -> float:
        """Total number of tasks released by the selected vessel."""

        return float(self._values.sum(dtype=np.float64))

    def arrivals_at(self, step: int) -> FloatArray:
        """Return a copy of arrivals for ``step``; post-horizon steps are zero."""

        if isinstance(step, bool) or not isinstance(step, (int, np.integer)):
            raise TypeError("step must be an integer")
        if step < 0:
            raise ValueError("step must be nonnegative")
        if step >= self.number_of_steps:
            return np.zeros((EXPECTED_REGIONS, EXPECTED_REGIONS), dtype=np.float64)
        return np.array(self._values[step], dtype=np.float64, copy=True)


@dataclass(frozen=True)
class FluidScenario:
    """Validated immutable metadata and NumPy views of a fluid scenario."""

    source_path: Path
    status: str
    vessel: str
    regions: tuple[Region, ...]
    e_g_edges: IntArray
    e_g_edge_index: IntArray
    e_reb_edges: IntArray
    e_reb_mask: BoolArray
    tau_minutes: FloatArray
    nonself_od_mask: BoolArray
    region_id_to_name: Mapping[int, str]
    region_name_to_id: Mapping[str, int]
    arrivals: TaskArrivalSeries
    task_records: tuple[TaskRecord, ...]
    tasks_by_release_step: Mapping[int, tuple[TaskRecord, ...]]
    task_count: int
    decision_period_min: float
    fleet_size: None

    def tasks_released_at(self, step: int) -> tuple[TaskRecord, ...]:
        """Return immutable task records released during decision interval ``step``."""

        if isinstance(step, bool) or not isinstance(step, (int, np.integer)):
            raise TypeError("step must be an integer")
        if step < 0:
            raise ValueError("step must be nonnegative")
        return self.tasks_by_release_step.get(int(step), ())


def _load_unique_edges(raw: Any, expected_count: int, label: str) -> IntArray:
    if not isinstance(raw, list) or len(raw) != expected_count:
        raise ValueError(f"{label} must contain exactly {expected_count} edges")
    edges: list[tuple[int, int]] = []
    for position, item in enumerate(raw):
        if (
            not isinstance(item, list)
            or len(item) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in item)
        ):
            raise ValueError(f"{label}[{position}] must be an integer [origin, destination] pair")
        origin, destination = item
        if not (0 <= origin < EXPECTED_REGIONS and 0 <= destination < EXPECTED_REGIONS):
            raise ValueError(f"{label}[{position}] contains an out-of-range region index")
        if origin == destination:
            raise ValueError(f"{label} must not contain self-loops")
        edges.append((origin, destination))
    if len(set(edges)) != expected_count:
        raise ValueError(f"{label} contains duplicate edges")
    return np.asarray(edges, dtype=np.int64)


def _build_tau(raw: Any) -> FloatArray:
    if not isinstance(raw, list) or len(raw) != EXPECTED_E_REB:
        raise ValueError("tau must contain exactly 240 OD records")
    tau = np.full((EXPECTED_REGIONS, EXPECTED_REGIONS), np.nan, dtype=np.float64)
    seen: set[tuple[int, int]] = set()
    for position, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"tau[{position}] must be an object")
        try:
            origin = item["origin"]
            destination = item["destination"]
            value = float(item["tau_minutes"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"tau[{position}] has invalid required fields") from exc
        if isinstance(origin, bool) or isinstance(destination, bool):
            raise ValueError(f"tau[{position}] uses a Boolean region index")
        if not isinstance(origin, int) or not isinstance(destination, int):
            raise ValueError(f"tau[{position}] region indices must be integers")
        if not (0 <= origin < EXPECTED_REGIONS and 0 <= destination < EXPECTED_REGIONS):
            raise ValueError(f"tau[{position}] contains an out-of-range region index")
        if origin == destination:
            raise ValueError("tau must contain only non-self OD records")
        if (origin, destination) in seen:
            raise ValueError(f"duplicate tau OD ({origin}, {destination})")
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"tau for ({origin}, {destination}) must be finite and positive")
        seen.add((origin, destination))
        tau[origin, destination] = value
    return tau


def _build_arrivals(
    document: Mapping[str, Any], decision_period_min: float
) -> tuple[TaskArrivalSeries, tuple[TaskRecord, ...], Mapping[int, tuple[TaskRecord, ...]]]:
    sparse = document.get("task_arrivals")
    tasks = document.get("tasks")
    if not isinstance(sparse, list) or not isinstance(tasks, list):
        raise ValueError("scenario must contain both task_arrivals and tasks lists")
    if len(tasks) != EXPECTED_TASKS:
        raise ValueError(f"tasks must contain exactly {EXPECTED_TASKS} records")

    sparse_counts: dict[tuple[int, int, int], int] = {}
    for position, row in enumerate(sparse):
        if not isinstance(row, dict):
            raise ValueError(f"task_arrivals[{position}] must be an object")
        try:
            step = int(row["decision_step"])
            origin = int(row["origin"])
            destination = int(row["destination"])
            count = int(row["count"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"task_arrivals[{position}] has invalid required fields") from exc
        if step < 0 or count <= 0:
            raise ValueError("arrival steps must be nonnegative and sparse counts positive")
        if not (0 <= origin < EXPECTED_REGIONS and 0 <= destination < EXPECTED_REGIONS):
            raise ValueError("task_arrivals contains an out-of-range OD")
        if origin == destination:
            raise ValueError("self-loop task arrivals are not allowed")
        key = (step, origin, destination)
        if key in sparse_counts:
            raise ValueError(f"duplicate sparse arrival key {key}")
        sparse_counts[key] = count

    detail_counts: dict[tuple[int, int, int], int] = {}
    task_records: list[TaskRecord] = []
    seen_task_ids: set[int] = set()
    seen_source_indices: set[int] = set()
    for position, task in enumerate(tasks):
        if not isinstance(task, dict):
            raise ValueError(f"tasks[{position}] must be an object")
        if task.get("vessel") != EXPECTED_VESSEL:
            raise ValueError(f"tasks[{position}] does not belong to {EXPECTED_VESSEL}")
        try:
            step = int(task["decision_step"])
            origin = int(task["origin"])
            destination = int(task["destination"])
            release_minute = float(task["release_minute"])
            task_id = int(task["task_id"])
            # Public processed scenarios use a synthetic stable source index;
            # no original spreadsheet row identifiers are distributed.
            source_index = int(task["source_index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"tasks[{position}] has invalid release/OD fields") from exc
        if not math.isfinite(release_minute) or release_minute < 0.0:
            raise ValueError(f"tasks[{position}] has an invalid release_minute")
        expected_step = int(math.floor(release_minute / decision_period_min))
        if step != expected_step:
            raise ValueError(
                f"tasks[{position}] decision_step {step} is inconsistent with LIFTTIME-based "
                f"release_minute {release_minute}"
            )
        if not (0 <= origin < EXPECTED_REGIONS and 0 <= destination < EXPECTED_REGIONS):
            raise ValueError(f"tasks[{position}] contains an out-of-range OD")
        if origin == destination:
            raise ValueError("self-loop tasks are not allowed")
        if task_id in seen_task_ids:
            raise ValueError(f"duplicate task_id {task_id}")
        if source_index in seen_source_indices:
            raise ValueError(f"duplicate source_index {source_index}")
        seen_task_ids.add(task_id)
        seen_source_indices.add(source_index)
        task_records.append(
            TaskRecord(
                task_id=task_id,
                source_index=source_index,
                release_minute=release_minute,
                release_step=step,
                origin=origin,
                destination=destination,
            )
        )
        key = (step, origin, destination)
        detail_counts[key] = detail_counts.get(key, 0) + 1

    if sparse_counts != detail_counts:
        missing = sorted(set(detail_counts.items()) - set(sparse_counts.items()))[:5]
        extra = sorted(set(sparse_counts.items()) - set(detail_counts.items()))[:5]
        raise ValueError(f"task_arrivals and tasks disagree; detail_only={missing}, sparse_only={extra}")
    if sum(sparse_counts.values()) != EXPECTED_TASKS:
        raise ValueError("aggregated arrivals do not sum to 3207")

    last_step = max((key[0] for key in sparse_counts), default=-1)
    values = np.zeros((last_step + 1, EXPECTED_REGIONS, EXPECTED_REGIONS), dtype=np.float64)
    for (step, origin, destination), count in sparse_counts.items():
        values[step, origin, destination] = float(count)
    arrival_series = TaskArrivalSeries(values, decision_period_min, last_step)
    ordered_records = tuple(
        sorted(task_records, key=lambda task: (task.release_minute, task.source_index))
    )
    by_step_mutable: dict[int, list[TaskRecord]] = {}
    for task in ordered_records:
        by_step_mutable.setdefault(task.release_step, []).append(task)
    by_step = MappingProxyType(
        {step: tuple(records) for step, records in sorted(by_step_mutable.items())}
    )
    for step in range(arrival_series.number_of_steps):
        record_matrix = np.zeros((EXPECTED_REGIONS, EXPECTED_REGIONS), dtype=np.float64)
        for task in by_step.get(step, ()):
            record_matrix[task.origin, task.destination] += 1.0
        if not np.array_equal(record_matrix, arrival_series.arrivals_at(step)):
            raise ValueError(f"task records disagree with arrivals_at({step})")
    return arrival_series, ordered_records, by_step


def load_fluid_scenario(path: str | Path) -> FluidScenario:
    """Load and strictly validate the reviewed V2.1 scenario JSON.

    The returned ``tau_minutes`` matrix has ``NaN`` on its diagonal because
    self-loop transport is not an admissible OD.  Consumers must use
    ``nonself_od_mask`` rather than inventing diagonal travel times.
    """

    source_path = Path(path).expanduser().resolve()
    try:
        document = json.loads(source_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load valid scenario JSON from {source_path}") from exc
    if not isinstance(document, dict):
        raise ValueError("scenario root must be an object")
    if document.get("status") != EXPECTED_STATUS:
        raise ValueError(f"scenario status must be {EXPECTED_STATUS}")
    if document.get("vessel") != EXPECTED_VESSEL:
        raise ValueError(f"scenario vessel must be {EXPECTED_VESSEL}")
    if document.get("task_count") != EXPECTED_TASKS:
        raise ValueError(f"task_count must equal {EXPECTED_TASKS}")

    raw_regions = document.get("regions")
    if not isinstance(raw_regions, list) or len(raw_regions) != EXPECTED_REGIONS:
        raise ValueError("regions must contain exactly 16 records")
    regions: list[Region] = []
    for position, row in enumerate(raw_regions):
        if not isinstance(row, dict):
            raise ValueError(f"regions[{position}] must be an object")
        try:
            region = Region(id=int(row["id"]), name=str(row["name"]), type=str(row["type"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"regions[{position}] has invalid fields") from exc
        regions.append(region)
    if [region.id for region in regions] != list(range(EXPECTED_REGIONS)):
        raise ValueError("region ids must be ordered and contiguous from 0 through 15")
    names = [region.name for region in regions]
    if len(set(names)) != EXPECTED_REGIONS:
        raise ValueError("region names must be unique")

    e_g_edges = _load_unique_edges(document.get("E_G"), EXPECTED_E_G, "E_G")
    e_reb_edges = _load_unique_edges(document.get("E_reb"), EXPECTED_E_REB, "E_reb")
    all_nonself = {(i, j) for i in range(EXPECTED_REGIONS) for j in range(EXPECTED_REGIONS) if i != j}
    if set(map(tuple, e_reb_edges.tolist())) != all_nonself:
        raise ValueError("E_reb must contain all 240 non-self directed ODs")

    tau = _build_tau(document.get("tau"))
    tau_od = set(map(tuple, np.argwhere(np.isfinite(tau)).tolist()))
    if tau_od != set(map(tuple, e_reb_edges.tolist())):
        raise ValueError("tau OD set must exactly match E_reb")

    time = document.get("time")
    if not isinstance(time, dict):
        raise ValueError("time metadata must be an object")
    if time.get("release_field") != "LIFTTIME":
        raise ValueError("task release_field must be LIFTTIME")
    try:
        decision_period_min = float(time["decision_period_min"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("decision_period_min is missing or invalid") from exc
    if decision_period_min != EXPECTED_DECISION_PERIOD_MIN:
        raise ValueError("decision period must be 15 minutes")
    arrivals, task_records, tasks_by_release_step = _build_arrivals(
        document, decision_period_min
    )

    initial_fleet = document.get("initial_fleet")
    if not isinstance(initial_fleet, dict) or initial_fleet.get("fleet_size") is not None:
        raise ValueError("formal fleet_size K must remain unset in the V2.1 scenario")

    nonself_mask = ~np.eye(EXPECTED_REGIONS, dtype=bool)
    e_reb_mask = np.zeros((EXPECTED_REGIONS, EXPECTED_REGIONS), dtype=bool)
    e_reb_mask[e_reb_edges[:, 0], e_reb_edges[:, 1]] = True
    e_g_edge_index = e_g_edges.T.copy()
    for array in (e_g_edges, e_g_edge_index, e_reb_edges, e_reb_mask, tau, nonself_mask):
        array.setflags(write=False)

    return FluidScenario(
        source_path=source_path,
        status=EXPECTED_STATUS,
        vessel=EXPECTED_VESSEL,
        regions=tuple(regions),
        e_g_edges=e_g_edges,
        e_g_edge_index=e_g_edge_index,
        e_reb_edges=e_reb_edges,
        e_reb_mask=e_reb_mask,
        tau_minutes=tau,
        nonself_od_mask=nonself_mask,
        region_id_to_name={region.id: region.name for region in regions},
        region_name_to_id={region.name: region.id for region in regions},
        arrivals=arrivals,
        task_records=task_records,
        tasks_by_release_step=tasks_by_release_step,
        task_count=EXPECTED_TASKS,
        decision_period_min=decision_period_min,
        fleet_size=None,
    )
