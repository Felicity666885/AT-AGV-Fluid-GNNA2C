"""Reproduce the paper's Table 2 and Table 3 from released CSV metrics."""

from __future__ import annotations

import csv
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def mean(rows: list[dict[str, str]], key: str) -> float:
    return statistics.fmean(float(row[key]) for row in rows)


def table2() -> Path:
    source = ROOT / "results" / "metrics" / "formal_training_summary.csv"
    rows = read_csv(source)
    if len(rows) != 3000 or len({(r["seed"], r["episode"]) for r in rows}) != 3000:
        raise ValueError("formal training summary must contain 3 x 1000 unique rows")
    first = [r for r in rows if 801 <= int(r["episode"]) <= 900]
    second = [r for r in rows if 901 <= int(r["episode"]) <= 1000]
    specs = (
        ("Cumulative scaled reward", "cumulative_scaled_reward", "percent"),
        ("Release-horizon service rate", "service_rate_by_release_horizon", "points"),
        ("Mean waiting time (min)", "mean_waiting_time", "percent"),
    )
    output_rows = []
    for label, key, mode in specs:
        old, new = mean(first, key), mean(second, key)
        change = (new - old) * 100.0 if mode == "points" else (new - old) / abs(old) * 100.0
        output_rows.append({
            "metric": label,
            "episodes_801_900_mean": old,
            "episodes_901_1000_mean": new,
            "change": change,
            "change_unit": "percentage points" if mode == "points" else "%",
        })
    output = ROOT / "results" / "tables" / "table2_training_window_comparison.csv"
    write_csv(output, output_rows, list(output_rows[0]))
    return output


def table3() -> Path:
    source = ROOT / "results" / "metrics" / "method_comparison_raw_results.csv"
    rows = read_csv(source)
    by_method = {name: [r for r in rows if r["method"] == name] for name in ("Uniform", "Demand-driven", "GNN-A2C")}
    if [len(by_method[name]) for name in by_method] != [1, 1, 3]:
        raise ValueError("method comparison must contain deterministic 1, 1, and 3 samples")
    specs = (
        ("Release-horizon service rate (%)", "service_rate_by_release_horizon", 100.0),
        ("Mean waiting time (min)", "mean_waiting_time", 1.0),
        ("P95 waiting time (min)", "p95_waiting_time", 1.0),
        ("Total waiting cost", "total_waiting_cost", 1.0),
        ("Total rebalancing cost", "total_rebalancing_cost", 1.0),
        ("Simulation termination step", "total_episode_steps", 1.0),
    )
    output_rows = []
    for label, key, scale in specs:
        gnn = [float(r[key]) * scale for r in by_method["GNN-A2C"]]
        output_rows.append({
            "metric": label,
            "Uniform": float(by_method["Uniform"][0][key]) * scale,
            "Demand-driven": float(by_method["Demand-driven"][0][key]) * scale,
            "GNN-A2C_mean": statistics.fmean(gnn),
            "GNN-A2C_sample_std": statistics.stdev(gnn),
            "GNN-A2C_sample_size": 3,
        })
    output = ROOT / "results" / "tables" / "table3_method_comparison.csv"
    write_csv(output, output_rows, list(output_rows[0]))
    return output


def main() -> None:
    outputs = (table2(), table3())
    for path in outputs:
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
