import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


STAGES = ("prepare", "featurize", "split", "gridsearch", "automl")
STAGE_NAMES = {
    "prepare": "Prepare",
    "featurize": "Featurize",
    "split": "Split",
    "gridsearch": "Grid-search",
    "automl": "AutoML",
}
IMPLEMENTATION_NAMES = {"localize": "LOCALIZE", "jupyter": "Notebook"}
EXPECTED_USAGE_COUNTS = {
    "localize": {
        "prepare": 2,
        "featurize": 2,
        "split": 4,
        "gridsearch": 8,
        "automl": 4,
    },
    "jupyter": {stage: 1 for stage in STAGES},
}
EXPECTED_REPORT_COUNT = 12
REQUIRED_USAGE_COLUMNS = {"t", "cpu_cores", "ram_mb", "memory_metric"}


def resolve_evaluation_dir(root: Path, value: str | Path | None = None) -> Path:
    evaluations = root / "results" / "evaluations"
    if value is None:
        latest = evaluations / "latest.txt"
        if not latest.is_file():
            raise FileNotFoundError(f"Evaluation selection not found: {latest}")
        value = latest.read_text(encoding="utf-8").strip()

    path = Path(value)
    if not path.is_absolute():
        path = evaluations / path
    path = path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Evaluation results not found: {path}")
    return path


def stage_usage_paths(run_dir: Path, stage: str) -> list[Path]:
    return sorted((run_dir / "usage").glob(f"{stage}*_usage.pkl"))


def report_paths(run_dir: Path) -> list[Path]:
    directory = run_dir / "reports"
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.rglob("*.pkl")
        if "result" in path.name.lower()
    )


def load_usage(path: Path) -> pd.DataFrame:
    frame = joblib.load(path)
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"Usage log is not a DataFrame: {path}")

    missing = REQUIRED_USAGE_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if len(frame) < 2:
        raise ValueError(f"Usage log has fewer than two samples: {path}")

    frame = frame.sort_values("t").reset_index(drop=True)
    if frame["t"].isna().any() or not frame["t"].is_monotonic_increasing:
        raise ValueError(f"Usage timestamps are invalid: {path}")
    return frame


def validate_run(run_dir: Path, implementation: str) -> None:
    if implementation not in EXPECTED_USAGE_COUNTS:
        raise ValueError(f"Unknown implementation: {implementation}")
    if not (run_dir / "complete.json").is_file():
        raise ValueError(f"Run is not marked complete: {run_dir}")

    for stage, expected in EXPECTED_USAGE_COUNTS[implementation].items():
        paths = stage_usage_paths(run_dir, stage)
        if len(paths) != expected:
            raise ValueError(
                f"Expected {expected} {stage} usage logs in {run_dir}, found {len(paths)}"
            )
        for path in paths:
            load_usage(path)

    reports = report_paths(run_dir)
    if len(reports) != EXPECTED_REPORT_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_REPORT_COUNT} result reports in {run_dir}, found {len(reports)}"
        )


def summarize_stage(run_dir: Path, stage: str) -> dict[str, object]:
    frames = [load_usage(path) for path in stage_usage_paths(run_dir, stage)]
    if not frames:
        raise FileNotFoundError(f"No {stage} usage logs found in {run_dir}")

    wall_seconds = sum(float(frame["t"].iloc[-1] - frame["t"].iloc[0]) for frame in frames)
    core_seconds = sum(
        float(np.trapz(frame["cpu_cores"].to_numpy(), frame["t"].to_numpy()))
        for frame in frames
    )
    memory = pd.concat([frame["ram_mb"] for frame in frames], ignore_index=True)
    metrics = {
        value
        for frame in frames
        for value in frame["memory_metric"].dropna().astype(str).unique()
    }
    if len(metrics) != 1:
        raise ValueError(f"Inconsistent memory metrics in {run_dir}: {sorted(metrics)}")

    return {
        "wall_seconds": wall_seconds,
        "core_seconds": core_seconds,
        "memory": memory,
        "memory_metric": metrics.pop(),
    }


def completed_runs(directory: Path, implementation: str) -> list[Path]:
    runs = sorted(
        path
        for path in directory.glob("run-*")
        if path.is_dir() and path.name.removeprefix("run-").isdigit()
    )
    if not runs:
        raise FileNotFoundError(f"No completed runs found: {directory}")
    for run in runs:
        validate_run(run, implementation)
    return runs


def summarize_benchmark(evaluation_dir: Path) -> pd.DataFrame:
    rows = []
    for implementation in ("localize", "jupyter"):
        runs = completed_runs(evaluation_dir / "benchmark" / implementation, implementation)
        for stage in STAGES:
            summaries = [summarize_stage(run, stage) for run in runs]
            memory = pd.concat(
                [summary["memory"] for summary in summaries],
                ignore_index=True,
            )
            metrics = {str(summary["memory_metric"]) for summary in summaries}
            if len(metrics) != 1:
                raise ValueError(
                    f"Inconsistent memory metrics for {implementation} {stage}: {sorted(metrics)}"
                )
            rows.append(
                {
                    "implementation": IMPLEMENTATION_NAMES[implementation],
                    "stage": stage,
                    "runs": len(runs),
                    "wall_seconds": np.mean(
                        [float(summary["wall_seconds"]) for summary in summaries]
                    ),
                    "core_seconds": np.mean(
                        [float(summary["core_seconds"]) for summary in summaries]
                    ),
                    "memory_mean_mb": memory.mean(),
                    "memory_q1_mb": memory.quantile(0.25),
                    "memory_q3_mb": memory.quantile(0.75),
                    "memory_max_mb": memory.max(),
                    "memory_metric": metrics.pop(),
                }
            )
    return pd.DataFrame(rows)


def summarize_scalability(evaluation_dir: Path) -> pd.DataFrame:
    rows = []
    for size in ("1x", "5x", "10x"):
        run_dir = evaluation_dir / "scalability" / size
        validate_run(run_dir, "localize")
        for stage in STAGES:
            summary = summarize_stage(run_dir, stage)
            memory = summary["memory"]
            rows.append(
                {
                    "size": size,
                    "stage": stage,
                    "wall_seconds": summary["wall_seconds"],
                    "core_seconds": summary["core_seconds"],
                    "memory_mean_mb": memory.mean(),
                    "memory_q1_mb": memory.quantile(0.25),
                    "memory_q3_mb": memory.quantile(0.75),
                    "memory_max_mb": memory.max(),
                    "memory_metric": summary["memory_metric"],
                }
            )
    return pd.DataFrame(rows)


def load_loc(evaluation_dir: Path) -> pd.DataFrame:
    path = evaluation_dir / "loc.json"
    if not path.is_file():
        raise FileNotFoundError(f"LOC results not found: {path}")
    return pd.DataFrame(json.loads(path.read_text(encoding="utf-8")))
