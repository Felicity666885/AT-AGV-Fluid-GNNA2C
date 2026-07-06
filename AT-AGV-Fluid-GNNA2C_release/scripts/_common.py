"""Shared relative-path configuration helpers for release scripts."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.scenario_loader import load_fluid_scenario
from src.experiments.final_experiment import (
    FinalExperimentConfig,
    canonical_json_bytes,
    sha256_bytes,
)
from src.policy.directed_graph import build_incoming_normalized_graph


def load_official_config(path: str | Path) -> FinalExperimentConfig:
    source = Path(path)
    if not source.is_absolute():
        source = (ROOT / source).resolve()
    raw: dict[str, Any] = yaml.safe_load(source.read_text(encoding="utf-8"))
    required = {
        "experiment_name", "scenario_path", "fleet_size",
        "initial_idle_distribution", "feature_scales", "reward_config",
        "a2c_config", "formal_seeds", "episodes", "max_decision_steps",
        "max_updates",
    }
    missing = sorted(required.difference(raw))
    if missing:
        raise ValueError(f"missing official config fields: {missing}")
    if raw["vessel"] != "MSAIV610A" or int(raw["task_count"]) != 3207:
        raise ValueError("official MSAIV610A identity mismatch")
    if int(raw["region_count"]) != 16 or int(raw["directed_edge_count"]) != 34:
        raise ValueError("official region topology mismatch")
    if int(raw["episodes"]) != 1000:
        raise ValueError("official release requires 1000 episodes")
    if list(map(int, raw["formal_seeds"])) != [20260801, 20260811, 20260821]:
        raise ValueError("official release seed list mismatch")
    if sum(map(float, raw["initial_idle_distribution"])) != float(raw["fleet_size"]):
        raise ValueError("initial idle distribution does not equal fleet size")
    raw["development_seeds"] = []
    digest = sha256_bytes(canonical_json_bytes(raw))
    return FinalExperimentConfig(source, raw, digest)


def load_scenario_and_graph(config: FinalExperimentConfig):
    scenario_path = (ROOT / str(config.raw["scenario_path"])).resolve()
    scenario = load_fluid_scenario(scenario_path)
    if scenario.task_count != int(config.raw["task_count"]):
        raise ValueError("scenario task count differs from official config")
    graph = build_incoming_normalized_graph(scenario.e_g_edge_index, num_nodes=16)
    return scenario, graph


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")

