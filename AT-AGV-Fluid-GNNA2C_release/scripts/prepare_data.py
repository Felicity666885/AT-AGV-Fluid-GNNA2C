"""Validate the public processed scenario and regenerate transparent CSV views."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENARIO = ROOT / "data" / "processed" / "scenario_MSAIV610A_processed.json"
REGION_ORDER = ("Q1", "Q2", "Q3", "Q4", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S12")
FORBIDDEN_KEYS = {
    "source_excel_row", "release_time", "historical_carrytime", "historical_puttime",
    "historical_waiting_time_min", "historical_travel_time_min", "workbook", "worksheet",
}


def write_csv(path: Path, rows: Iterable[Mapping[str, object]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def validate_public_scenario(document: Mapping[str, object]) -> None:
    text = json.dumps(document, ensure_ascii=False)
    if any(key in text for key in FORBIDDEN_KEYS):
        raise ValueError("processed scenario contains a forbidden private-data field")
    if document.get("vessel") != "MSAIV610A" or int(document.get("task_count", -1)) != 3207:
        raise ValueError("processed scenario identity mismatch")
    regions = document.get("regions")
    tasks = document.get("tasks")
    edges = document.get("E_G")
    tau = document.get("tau")
    if not isinstance(regions, list) or len(regions) != 16:
        raise ValueError("processed scenario must contain 16 regions")
    if [row["name"] for row in regions] != list(REGION_ORDER):
        raise ValueError("processed region order mismatch")
    if not isinstance(tasks, list) or len(tasks) != 3207:
        raise ValueError("processed scenario must contain 3207 tasks")
    if not isinstance(edges, list) or len({tuple(edge) for edge in edges}) != 34:
        raise ValueError("processed scenario must contain 34 unique directed E_G edges")
    if not isinstance(tau, list) or len(tau) != 240:
        raise ValueError("processed scenario must contain 240 non-self OD travel times")
    task_ids = [int(task["task_id"]) for task in tasks]
    source_indices = [int(task["source_index"]) for task in tasks]
    if task_ids != list(range(3207)) or source_indices != list(range(3207)):
        raise ValueError("task_id/source_index must be synthetic contiguous identifiers")
    for task in tasks:
        release = float(task["release_minute"])
        step = int(task["decision_step"])
        origin, destination = int(task["origin"]), int(task["destination"])
        if not math.isfinite(release) or release < 0 or step != int(release // 15):
            raise ValueError("invalid relative task release data")
        if not (0 <= origin < 16 and 0 <= destination < 16 and origin != destination):
            raise ValueError("invalid task OD")
    for row in tau:
        value = float(row["tau_minutes"])
        if not math.isfinite(value) or value <= 0:
            raise ValueError("travel time must be finite and positive")


def regenerate_csvs(scenario_path: Path) -> dict[str, int]:
    document = json.loads(scenario_path.read_text(encoding="utf-8"))
    validate_public_scenario(document)
    output = scenario_path.parent
    regions = document["regions"]
    names = {int(row["id"]): str(row["name"]) for row in regions}

    task_rows = [
        {
            "task_id": int(task["task_id"]),
            "release_minute": float(task["release_minute"]),
            "decision_step": int(task["decision_step"]),
            "origin": names[int(task["origin"])],
            "destination": names[int(task["destination"])],
        }
        for task in document["tasks"]
    ]
    write_csv(output / "MSAIV610A_tasks_processed.csv", task_rows, ("task_id", "release_minute", "decision_step", "origin", "destination"))

    arrival_rows = [
        {
            "decision_step": int(row["decision_step"]),
            "time_stamp_min": float(row["time_stamp_min"]),
            "origin": names[int(row["origin"])],
            "destination": names[int(row["destination"])],
            "count": int(row["count"]),
        }
        for row in document["task_arrivals"]
    ]
    write_csv(output / "task_arrivals_15min.csv", arrival_rows, ("decision_step", "time_stamp_min", "origin", "destination", "count"))

    edge_rows = [
        {"source": names[int(edge[0])], "destination": names[int(edge[1])]}
        for edge in document["E_G"]
    ]
    write_csv(output / "region_network_edges.csv", edge_rows, ("source", "destination"))

    tau_lookup = {(int(row["origin"]), int(row["destination"])): float(row["tau_minutes"]) for row in document["tau"]}
    matrix_rows = []
    for origin in range(16):
        row: dict[str, object] = {"origin": names[origin]}
        for destination in range(16):
            row[names[destination]] = "" if origin == destination else tau_lookup[(origin, destination)]
        matrix_rows.append(row)
    write_csv(output / "travel_time_matrix.csv", matrix_rows, ("origin", *REGION_ORDER))

    metadata_rows = [
        {"region_id": int(row["id"]), "region_name": str(row["name"]), "region_type": str(row["type"])}
        for row in regions
    ]
    write_csv(output / "region_metadata.csv", metadata_rows, ("region_id", "region_name", "region_type"))
    return {
        "tasks": len(task_rows),
        "arrival_rows": len(arrival_rows),
        "directed_edges": len(edge_rows),
        "travel_time_od": len(tau_lookup),
        "regions": len(metadata_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=Path, default=DEFAULT_SCENARIO)
    args = parser.parse_args()
    counts = regenerate_csvs(args.scenario.resolve())
    print(json.dumps(counts, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
