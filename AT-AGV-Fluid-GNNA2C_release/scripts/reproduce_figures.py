"""Reproduce copyright-safe paper Figures 5--9 from released data."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


REGIONS = ("Q1", "Q2", "Q3", "Q4", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S12")
FIGURES = ROOT / "figures"
SOURCE = FIGURES / "source_data"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader(); writer.writerows(rows)


def save(fig, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig5_framework() -> None:
    fig, ax = plt.subplots(figsize=(11, 3.5), constrained_layout=True)
    ax.axis("off")
    labels = [
        "Physical state\n$Q, Z^{idle}, Z^{loaded}, Y$",
        "FCFS matching\n$M_t$",
        "GNN-A2C target\nproportions $q_t$",
        "Integer rounding +\nrebalancing optimization $X_t$",
        "Analytical fluid update\n$s_t \\rightarrow s_{t+1}$",
    ]
    xs = np.linspace(0.02, 0.82, len(labels))
    for i, (x, label) in enumerate(zip(xs, labels)):
        ax.add_patch(Rectangle((x, .35), .16, .30, facecolor="#eaf2f8", edgecolor="#1f4e79", linewidth=1.6))
        ax.text(x + .08, .50, label, ha="center", va="center", fontsize=9)
        if i < len(labels) - 1:
            ax.add_patch(FancyArrowPatch((x + .16, .50), (xs[i + 1], .50), arrowstyle="->", mutation_scale=14, color="#444"))
    ax.set_title("Dynamic scheduling and rebalancing framework", fontsize=13)
    save(fig, "fig5_task_matching_rebalancing.png")


def demand_counts() -> tuple[dict[str, int], dict[str, int]]:
    tasks = read_csv(ROOT / "data" / "processed" / "MSAIV610A_tasks_processed.csv")
    origins = {r: 0 for r in REGIONS}; destinations = {r: 0 for r in REGIONS}
    for row in tasks:
        origins[row["origin"]] += 1; destinations[row["destination"]] += 1
    if sum(origins.values()) != 3207 or sum(destinations.values()) != 3207:
        raise ValueError("task demand totals do not equal 3207")
    write_csv(SOURCE / "fig6_task_origin_destination_distribution.csv", [
        {"region": r, "origin_tasks": origins[r], "destination_tasks": destinations[r]} for r in REGIONS
    ], ["region", "origin_tasks", "destination_tasks"])
    return origins, destinations


def fig6_demand() -> None:
    layout = json.loads((ROOT / "data" / "processed" / "cict_16_region_layout.json").read_text(encoding="utf-8"))
    positions = {r["name"]: (float(r["center"][0]), float(r["center"][1]), r["type"]) for r in layout["regions"]}
    origins, destinations = demand_counts()
    vmax = max(max(origins.values()), max(destinations.values()))
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    mappable = None
    for ax, values, title in zip(axes, (origins, destinations), ("(a) Task origins", "(b) Task destinations")):
        ax.set_aspect("equal"); ax.axis("off"); ax.set_title(title)
        for name in REGIONS:
            x, y, kind = positions[name]
            color = plt.cm.YlOrRd(values[name] / vmax if vmax else 0)
            patch = Rectangle((x - .46, y - .40), .92, .80, facecolor=color, edgecolor="#17365d", linewidth=2 if kind == "QC" else 1.2)
            ax.add_patch(patch); ax.text(x, y, f"{name}\n{values[name]}", ha="center", va="center", fontsize=8)
        ax.set_xlim(-.6, 3.6); ax.set_ylim(-.55, 3.55)
        mappable = plt.cm.ScalarMappable(norm=plt.Normalize(0, vmax), cmap="YlOrRd")
    fig.colorbar(mappable, ax=axes, shrink=.75, label="Number of tasks")
    fig.suptitle("Abstract 16-region origin--destination demand distribution")
    save(fig, "fig6_task_origin_destination_distribution.png")


def fig7_arrivals() -> None:
    rows = read_csv(ROOT / "data" / "processed" / "task_arrivals_15min.csv")
    max_step = max(int(r["decision_step"]) for r in rows)
    matrix = np.zeros((16, max_step + 1), dtype=float)
    index = {name: i for i, name in enumerate(REGIONS)}
    for row in rows:
        matrix[index[row["origin"]], int(row["decision_step"])] += int(row["count"])
    peak = int(np.argmax(matrix.sum(axis=0)))
    source_rows = [{"decision_step": t, "region": r, "arrivals": matrix[i, t]} for i, r in enumerate(REGIONS) for t in range(max_step + 1)]
    write_csv(SOURCE / "fig7_task_arrival_distribution.csv", source_rows, ["decision_step", "region", "arrivals"])
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    image = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="YlOrRd", origin="upper")
    ax.axvline(peak, color="#1f77b4", linestyle="--", linewidth=1.4, label=f"Peak step {peak}")
    ax.set_yticks(range(16), REGIONS); ax.set_xlabel("Decision step (15 min)"); ax.set_ylabel("Origin region")
    ax.set_title("Task arrivals by decision step and origin region"); ax.legend(loc="upper right")
    fig.colorbar(image, ax=ax, label="Task arrivals")
    save(fig, "fig7_training_stability.png".replace("training_stability", "task_arrival_distribution"))


def moving_average(values: np.ndarray, window: int = 25) -> np.ndarray:
    result = np.full(values.shape, np.nan)
    for i in range(window - 1, len(values)):
        result[i] = values[i - window + 1:i + 1].mean()
    return result


def fig8_training() -> None:
    rows = read_csv(ROOT / "results" / "metrics" / "formal_training_summary.csv")
    episodes = np.arange(1, 1001)
    specs = (("cumulative_scaled_reward", "Cumulative scaled reward"), ("service_rate_by_release_horizon", "Release-horizon service rate"), ("mean_waiting_time", "Mean waiting time (min)"))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    source_rows = []
    for ax, (key, label) in zip(axes, specs):
        data = np.array([[float(r[key]) for r in rows if int(r["seed"]) == seed] for seed in (20260801, 20260811, 20260821)])
        means = data.mean(axis=0); stds = data.std(axis=0, ddof=1); smooth = moving_average(means)
        ax.plot(episodes, means, color="#4c78a8", linewidth=.7, alpha=.7, label="3-seed mean")
        ax.fill_between(episodes, means - stds, means + stds, color="#4c78a8", alpha=.18, label="$\\pm$1 sample SD")
        ax.plot(episodes, smooth, color="#d62728", linewidth=1.5, label="25-episode moving mean")
        ax.set_xlabel("Episode"); ax.set_ylabel(label); ax.grid(alpha=.2)
        for episode, mean_value, std_value, smooth_value in zip(episodes, means, stds, smooth):
            source_rows.append({"metric": key, "episode": episode, "mean": mean_value, "sample_std": std_value, "moving_mean_25": smooth_value})
    axes[0].legend(fontsize=8); fig.suptitle("Three-seed GNN-A2C training stability")
    write_csv(SOURCE / "fig8_training_stability.csv", source_rows, ["metric", "episode", "mean", "sample_std", "moving_mean_25"])
    save(fig, "fig8_training_stability.png")


def fig9_tradeoff() -> None:
    rows = read_csv(ROOT / "results" / "metrics" / "method_comparison_raw_results.csv")
    methods = ("Uniform", "Demand-driven", "GNN-A2C")
    source_rows = []
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), constrained_layout=True)
    colors = {"Uniform": "#7f7f7f", "Demand-driven": "#f2a541", "GNN-A2C": "#2a9d8f"}
    for method in methods:
        sample = [r for r in rows if r["method"] == method]
        cost = np.array([float(r["total_rebalancing_cost"]) for r in sample])
        service = np.array([float(r["service_rate_by_release_horizon"]) * 100 for r in sample])
        waiting = np.array([float(r["mean_waiting_time"]) for r in sample])
        x, ys, yw = cost.mean(), service.mean(), waiting.mean()
        xerr = cost.std(ddof=1) if len(cost) > 1 else None
        yserr = service.std(ddof=1) if len(service) > 1 else None
        ywerr = waiting.std(ddof=1) if len(waiting) > 1 else None
        axes[0].errorbar(x, ys, xerr=xerr, yerr=yserr, marker="o", capsize=4, color=colors[method], label=method)
        axes[1].errorbar(x, yw, xerr=xerr, yerr=ywerr, marker="o", capsize=4, color=colors[method], label=method)
        source_rows.append({"method": method, "sample_size": len(sample), "rebalancing_cost_mean": x, "service_rate_percent_mean": ys, "mean_waiting_time_mean": yw, "rebalancing_cost_sample_std": "" if xerr is None else xerr, "service_rate_sample_std": "" if yserr is None else yserr, "waiting_time_sample_std": "" if ywerr is None else ywerr})
    axes[0].set(xlabel="Total rebalancing cost", ylabel="Release-horizon service rate (%)")
    axes[1].set(xlabel="Total rebalancing cost", ylabel="Mean waiting time (min)")
    for ax in axes: ax.grid(alpha=.25); ax.margins(.15)
    axes[0].legend(); fig.suptitle("Cost--service trade-off")
    write_csv(SOURCE / "fig9_cost_service_tradeoff.csv", source_rows, list(source_rows[0]))
    save(fig, "fig9_cost_service_tradeoff.png")


def main() -> None:
    SOURCE.mkdir(parents=True, exist_ok=True)
    fig5_framework(); fig6_demand(); fig7_arrivals(); fig8_training(); fig9_tradeoff()
    for path in sorted(FIGURES.glob("fig*.png")):
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
