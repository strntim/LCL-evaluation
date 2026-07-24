#!/usr/bin/env python3

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from evaluation_data import (
    IMPLEMENTATION_NAMES,
    STAGES,
    STAGE_NAMES,
    load_loc,
    resolve_evaluation_dir,
    summarize_benchmark,
    summarize_scalability,
)


ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATIONS = tuple(IMPLEMENTATION_NAMES.values())
SIZES = ("1x", "5x", "10x")
SIZE_LABELS = {"1x": "1×", "5x": "5×", "10x": "10×"}
STAGE_HATCHES = {
    "prepare": "\\",
    "featurize": "//",
    "split": "xx",
    "gridsearch": "--",
    "automl": "||",
}
STAGE_COLORS = dict(zip(STAGES, plt.get_cmap("Set1")(np.linspace(0, 1, 9))))
plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 300})


def save_figure(fig: plt.Figure, directory: Path, filename: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / filename, bbox_inches="tight")
    plt.close(fig)


def style_labels(labels) -> None:
    for label in labels:
        label.set_path_effects(
            [path_effects.Stroke(linewidth=2, foreground="white"), path_effects.Normal()]
        )
        label.set_zorder(10)


def label_center(ax: plt.Axes, bars, fmt: str = "%.0f", fontsize: int = 8) -> None:
    labels = ax.bar_label(bars, label_type="center", fmt=fmt, fontsize=fontsize)
    style_labels(labels)


def label_loc(
    ax: plt.Axes,
    bars,
    fontsize: int = 7,
    deleted: bool = False,
) -> None:
    values = [bar.get_height() for bar in bars]
    centered = [f"{value:.0f}" if abs(value) >= 10 else "" for value in values]
    outside = [f"{value:.0f}" if 0 < abs(value) < 10 else "" for value in values]
    style_labels(
        ax.bar_label(
            bars,
            labels=centered,
            label_type="center",
            fontsize=fontsize,
        )
    )
    style_labels(
        ax.bar_label(
            bars,
            labels=outside,
            label_type="edge",
            fontsize=fontsize,
            padding=2,
        )
    )
    if deleted:
        zero_labels = [
            ax.annotate(
                "-0",
                xy=(bar.get_x() + bar.get_width() / 2, 0),
                xytext=(0, -2),
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=fontsize,
            )
            for bar, value in zip(bars, values)
            if value == 0
        ]
        style_labels(zero_labels)


def add_distribution(
    ax: plt.Axes,
    x: float,
    mean: float,
    q1: float,
    q3: float,
    maximum: float,
) -> None:
    ax.errorbar(
        x,
        mean,
        yerr=[[max(mean - q1, 0)], [max(q3 - mean, 0)]],
        fmt="none",
        ecolor="black",
        capsize=3,
        linewidth=1.2,
    )
    ax.vlines(x, mean, maximum, linestyles=":", colors="black", linewidth=1)
    ax.scatter(x, maximum, marker="_", s=40, color="black", zorder=3)


def plot_benchmark_memory(summary, figures: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    group_width = 0.9
    bar_width = group_width / len(STAGES)

    for implementation_index, implementation in enumerate(IMPLEMENTATIONS):
        for stage_index, stage in enumerate(STAGES):
            row = summary[
                (summary["implementation"] == implementation)
                & (summary["stage"] == stage)
            ].iloc[0]
            x = implementation_index - group_width / 2 + stage_index * bar_width + bar_width / 2
            mean = row["memory_mean_mb"]
            q1 = row["memory_q1_mb"]
            q3 = row["memory_q3_mb"]
            maximum = row["memory_max_mb"]

            bars = ax.bar(
                x,
                mean,
                width=bar_width,
                color=STAGE_COLORS[stage],
                hatch=STAGE_HATCHES[stage],
                edgecolor="black",
            )
            label_center(ax, bars)
            add_distribution(ax, x, mean, q1, q3, maximum)

    ax.set_xticks(range(len(IMPLEMENTATIONS)), IMPLEMENTATIONS)
    ax.set_ylabel("Memory usage [MB]")
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle=":", linewidth=0.5)
    ax.legend(
        handles=[
            Patch(
                facecolor=STAGE_COLORS[stage],
                edgecolor="black",
                hatch=STAGE_HATCHES[stage],
                label=STAGE_NAMES[stage],
            )
            for stage in STAGES
        ],
        title="Stage",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )
    ax.set_title("Memory usage statistics per pipeline stage")
    save_figure(fig, figures, "MemUsageGraph.png")


def plot_benchmark_time(summary, figures: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(9, 5.6), constrained_layout=True)
    axes = axes.flatten()
    bar_width = 0.36

    for stage, ax in zip(STAGES, axes[:-1]):
        rows = summary[summary["stage"] == stage].set_index("implementation")
        positions = np.arange(len(IMPLEMENTATIONS))
        cpu = rows.loc[list(IMPLEMENTATIONS), "core_seconds"].to_numpy()
        wall = rows.loc[list(IMPLEMENTATIONS), "wall_seconds"].to_numpy()
        cpu_bars = ax.bar(
            positions - bar_width / 2,
            cpu,
            width=bar_width,
            color=STAGE_COLORS[stage],
            edgecolor="black",
            hatch="xx",
        )
        wall_bars = ax.bar(
            positions + bar_width / 2,
            wall,
            width=bar_width,
            color=STAGE_COLORS[stage],
            edgecolor="black",
            hatch="//",
            alpha=0.65,
        )
        label_center(ax, cpu_bars, fmt="%.1f", fontsize=7)
        label_center(ax, wall_bars, fmt="%.1f", fontsize=7)
        for index, implementation in enumerate(IMPLEMENTATIONS):
            row = rows.loc[implementation]
            add_distribution(
                ax,
                positions[index] - bar_width / 2,
                row["core_seconds"],
                row["core_q1_seconds"],
                row["core_q3_seconds"],
                row["core_max_seconds"],
            )
            add_distribution(
                ax,
                positions[index] + bar_width / 2,
                row["wall_seconds"],
                row["wall_q1_seconds"],
                row["wall_q3_seconds"],
                row["wall_max_seconds"],
            )
        ax.set_xticks(positions, IMPLEMENTATIONS, fontsize=8)
        ax.set_title(STAGE_NAMES[stage])
        ax.set_axisbelow(True)
        ax.grid(axis="y", linestyle=":", linewidth=0.5)

    axes[-1].axis("off")
    axes[-1].legend(
        handles=[
            Patch(facecolor="white", edgecolor="black", hatch="xx", label="CPU time [core·s]"),
            Patch(facecolor="white", edgecolor="black", hatch="//", label="Wall time [s]"),
        ],
        loc="center",
    )
    fig.suptitle("Wall-clock and CPU time per pipeline stage")
    save_figure(fig, figures, "CPUWall_time.png")


def plot_loc(loc, figures: Path) -> None:
    implementations = (
        ("localize", "LOCALIZE", "green", "red", "--", "||"),
        ("notebook", "Jupyter", "lightgreen", "salmon", "//", "\\"),
        ("kedro", "Kedro", "cornflowerblue", "lightskyblue", "xx", ".."),
    )
    x = np.arange(len(loc))
    width = 0.25
    fig, ax = plt.subplots(figsize=(7.5, 4.6), constrained_layout=True)

    for index, (key, label, added_color, deleted_color, added_hatch, deleted_hatch) in enumerate(
        implementations
    ):
        position = x + (index - 1) * width
        added = ax.bar(
            position,
            loc[f"{key}_added"],
            width,
            color=added_color,
            edgecolor="black",
            hatch=added_hatch,
            label=f"{label} added",
        )
        deleted = ax.bar(
            position,
            -loc[f"{key}_deleted"],
            width,
            color=deleted_color,
            edgecolor="black",
            hatch=deleted_hatch,
            label=f"{label} deleted",
        )
        label_loc(ax, added)
        label_loc(ax, deleted, deleted=True)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, [f"Change {index}" for index in loc["change"]])
    ax.set_ylabel("Lines of code")
    ax.set_title("Lines added and removed per experiment change")
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle=":", linewidth=0.5)
    ax.legend(ncols=3, fontsize=7)
    save_figure(fig, figures, "LOCchanged.png")


def plot_scalability_memory(summary, figures: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.8), constrained_layout=True)
    group_width = 0.9
    bar_width = group_width / len(STAGES)

    for size_index, size in enumerate(SIZES):
        for stage_index, stage in enumerate(STAGES):
            row = summary[(summary["size"] == size) & (summary["stage"] == stage)].iloc[0]
            x = size_index - group_width / 2 + stage_index * bar_width + bar_width / 2
            mean = row["memory_mean_mb"]
            q1 = row["memory_q1_mb"]
            q3 = row["memory_q3_mb"]
            maximum = row["memory_max_mb"]
            bars = ax.bar(
                x,
                mean,
                width=bar_width,
                color=STAGE_COLORS[stage],
                hatch=STAGE_HATCHES[stage],
                edgecolor="black",
            )
            label_center(ax, bars)
            ax.errorbar(
                x,
                mean,
                yerr=[[max(mean - q1, 0)], [max(q3 - mean, 0)]],
                fmt="none",
                ecolor="black",
                capsize=3,
                linewidth=1.5,
            )
            ax.vlines(x, mean, maximum, linestyles=":", colors="black", linewidth=1)
            ax.scatter(x, maximum, marker="_", s=45, color="black", zorder=3)

    ax.set_xticks(range(len(SIZES)), [SIZE_LABELS[size] for size in SIZES])
    ax.set_ylabel("Memory usage [MB]")
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle=":", linewidth=0.5)
    ax.legend(
        handles=[
            Patch(
                facecolor=STAGE_COLORS[stage],
                edgecolor="black",
                hatch=STAGE_HATCHES[stage],
                label=STAGE_NAMES[stage],
            )
            for stage in STAGES
        ],
        title="Stage",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )
    ax.set_title("Memory usage by LOG-a-TEC dataset size")
    save_figure(fig, figures, "DSSC_MemUsageGraph.png")


def plot_scalability_time(summary, figures: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(8, 5.6), constrained_layout=True)
    axes = axes.flatten()
    bar_width = 0.36

    for stage, ax in zip(STAGES, axes[:-1]):
        rows = summary[summary["stage"] == stage].set_index("size")
        positions = np.arange(len(SIZES))
        cpu = rows.loc[list(SIZES), "core_seconds"].to_numpy()
        wall = rows.loc[list(SIZES), "wall_seconds"].to_numpy()
        cpu_bars = ax.bar(
            positions - bar_width / 2,
            cpu,
            width=bar_width,
            color=STAGE_COLORS[stage],
            edgecolor="black",
            hatch="xx",
        )
        wall_bars = ax.bar(
            positions + bar_width / 2,
            wall,
            width=bar_width,
            color=STAGE_COLORS[stage],
            edgecolor="black",
            hatch="//",
            alpha=0.65,
        )
        ax.bar_label(cpu_bars, fmt="%.1f", fontsize=7, padding=2)
        ax.bar_label(wall_bars, fmt="%.1f", fontsize=7, padding=2)
        ax.set_xticks(
            positions,
            [SIZE_LABELS[size] for size in SIZES],
            fontsize=8,
        )
        ax.set_title(STAGE_NAMES[stage])
        ax.set_axisbelow(True)
        ax.grid(axis="y", linestyle=":", linewidth=0.5)

    axes[-1].axis("off")
    axes[-1].legend(
        handles=[
            Patch(facecolor="white", edgecolor="black", hatch="xx", label="CPU time [core·s]"),
            Patch(facecolor="white", edgecolor="black", hatch="//", label="Wall time [s]"),
        ],
        loc="center",
    )
    fig.suptitle("Wall-clock and CPU time by LOG-a-TEC dataset size")
    save_figure(fig, figures, "DSSC_CPUWall_time.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-dir")
    parser.add_argument("--runs", type=int)
    parser.add_argument("--benchmark-only", action="store_true")
    args = parser.parse_args()
    evaluation_dir = resolve_evaluation_dir(ROOT, args.evaluation_dir)
    figures = evaluation_dir / "figures"

    benchmark = summarize_benchmark(evaluation_dir, runs=args.runs)
    loc = load_loc(evaluation_dir)

    plot_benchmark_memory(benchmark, figures)
    plot_benchmark_time(benchmark, figures)
    plot_loc(loc, figures)
    if not args.benchmark_only:
        scalability = summarize_scalability(evaluation_dir)
        plot_scalability_memory(scalability, figures)
        plot_scalability_time(scalability, figures)
    print(f"Figures saved to {figures}")


if __name__ == "__main__":
    main()
