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


def label_center(ax: plt.Axes, bars, fmt: str = "%.0f", fontsize: int = 8) -> None:
    labels = ax.bar_label(bars, label_type="center", fmt=fmt, fontsize=fontsize)
    for label in labels:
        label.set_path_effects(
            [path_effects.Stroke(linewidth=2, foreground="white"), path_effects.Normal()]
        )


def plot_benchmark_memory(summary, figures: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.8), constrained_layout=True)
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
    fig, axes = plt.subplots(2, 3, figsize=(8, 5.6), constrained_layout=True)
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
        ax.bar_label(cpu_bars, fmt="%.1f", fontsize=7, padding=2)
        ax.bar_label(wall_bars, fmt="%.1f", fontsize=7, padding=2)
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
    x = np.arange(len(loc))
    width = 0.36
    fig, ax = plt.subplots(figsize=(6.2, 4.3), constrained_layout=True)
    localize_added = ax.bar(
        x - width / 2,
        loc["localize_added"],
        width,
        color="green",
        edgecolor="black",
        hatch="--",
        label="LOCALIZE added",
    )
    localize_deleted = ax.bar(
        x - width / 2,
        -loc["localize_deleted"],
        width,
        color="red",
        edgecolor="black",
        hatch="||",
        label="LOCALIZE deleted",
    )
    notebook_added = ax.bar(
        x + width / 2,
        loc["notebook_added"],
        width,
        color="lightgreen",
        edgecolor="black",
        hatch="//",
        label="Notebook added",
    )
    notebook_deleted = ax.bar(
        x + width / 2,
        -loc["notebook_deleted"],
        width,
        color="salmon",
        edgecolor="black",
        hatch="\\",
        label="Notebook deleted",
    )
    for bars in (localize_added, localize_deleted, notebook_added, notebook_deleted):
        label_center(ax, bars)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, [f"Change {index}" for index in loc["change"]])
    ax.set_ylabel("Lines of code")
    ax.set_title("Lines added and removed per experiment change")
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle=":", linewidth=0.5)
    ax.legend(ncols=2, fontsize=8)
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

    ax.set_xticks(range(len(SIZES)), SIZES)
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
        ax.set_xticks(positions, SIZES, fontsize=8)
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
    args = parser.parse_args()
    evaluation_dir = resolve_evaluation_dir(ROOT, args.evaluation_dir)
    figures = evaluation_dir / "figures"

    benchmark = summarize_benchmark(evaluation_dir)
    scalability = summarize_scalability(evaluation_dir)
    loc = load_loc(evaluation_dir)

    plot_benchmark_memory(benchmark, figures)
    plot_benchmark_time(benchmark, figures)
    plot_loc(loc, figures)
    plot_scalability_memory(scalability, figures)
    plot_scalability_time(scalability, figures)
    print(f"Figures saved to {figures}")


if __name__ == "__main__":
    main()
