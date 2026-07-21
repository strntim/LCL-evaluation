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
EXPERIMENTAL_IMPLEMENTATION_NAMES = {
    **IMPLEMENTATION_NAMES,
    "kedro": "Kedro",
}
EXPECTED_USAGE_COUNTS = {
    "localize": {
        "prepare": 2,
        "featurize": 2,
        "split": 4,
        "gridsearch": 8,
        "automl": 4,
    },
    "jupyter": {stage: 1 for stage in STAGES},
    "kedro": {stage: 1 for stage in STAGES},
}
EXPECTED_REPORT_COUNT = 12
REQUIRED_USAGE_COLUMNS = {"t", "cpu_cores", "ram_mb", "memory_metric"}
SUBSETS = ("spring", "winter")
SPLITS = ("KFold", "Random")
GRIDSEARCH_MODELS = ("RandomForestRegressor", "KNeighborsRegressor")
GRIDSEARCH_RESULT_COLUMNS = (
    "mean_test_rmse",
    "std_test_rmse",
    "mean_test_r_squared",
    "std_test_r_squared",
)


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


def find_report(run_dir: Path, filename: str) -> Path:
    matches = list((run_dir / "reports").rglob(filename))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one {filename} in {run_dir / 'reports'}, found {len(matches)}"
        )
    return matches[0]


def load_report(path: Path, implementation: str):
    report = joblib.load(path)

    if implementation == "localize":
        if not isinstance(report, dict):
            raise TypeError(f"LOCALIZE report is not a dictionary: {path}")
        model_reports = report.get("model_data", {}).get("reports")
        if not isinstance(model_reports, list) or not model_reports:
            raise ValueError(f"LOCALIZE report has no model reports: {path}")
        for model_report in model_reports:
            required = {"scores", "params", "fit_time", "score_time"}
            if not isinstance(model_report, dict) or not required.issubset(model_report):
                raise ValueError(f"LOCALIZE model report has an invalid structure: {path}")
        return report

    if implementation in {"jupyter", "kedro"}:
        label = "Notebook" if implementation == "jupyter" else "Kedro"
        if isinstance(report, pd.DataFrame):
            required = {"params", *GRIDSEARCH_RESULT_COLUMNS}
            missing = required.difference(report.columns)
            if missing:
                raise ValueError(f"{label} grid-search report is missing {sorted(missing)}: {path}")
            if report.empty:
                raise ValueError(f"{label} grid-search report is empty: {path}")
            if implementation == "kedro":
                for column in GRIDSEARCH_RESULT_COLUMNS:
                    values = pd.to_numeric(report[column], errors="coerce").to_numpy()
                    if not np.isfinite(values).any():
                        raise ValueError(f"Kedro grid-search column {column} is invalid: {path}")
            return report
        if isinstance(report, list) and report and isinstance(report[0], dict):
            required = {"scores", "params", "fit_time", "score_time"}
            if not required.issubset(report[0]):
                raise ValueError(f"{label} AutoML report has an invalid structure: {path}")
            if implementation == "kedro":
                for metric in ("rmse", "r_squared"):
                    scores = report[0]["scores"].get(metric, {})
                    values = [scores.get("mean"), scores.get("std")]
                    if not all(value is not None and np.isfinite(float(value)) for value in values):
                        raise ValueError(f"Kedro AutoML metric {metric} is invalid: {path}")
            return report
        raise TypeError(f"{label} report has an unexpected type: {path}")

    raise ValueError(f"Unknown implementation: {implementation}")


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
    for path in reports:
        load_report(path, implementation)


def _parameter_key(parameters: object) -> str:
    def normalize(value: object):
        if isinstance(value, dict):
            return {str(key): normalize(item) for key, item in sorted(value.items())}
        if isinstance(value, (list, tuple)):
            return [normalize(item) for item in value]
        if isinstance(value, np.generic):
            return value.item()
        return value

    return json.dumps(normalize(parameters), sort_keys=True, default=str)


def _gridsearch_values(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    missing = {"params", *GRIDSEARCH_RESULT_COLUMNS}.difference(frame.columns)
    if missing:
        raise ValueError(f"Grid-search results are missing columns: {sorted(missing)}")

    values = {}
    for _, row in frame.iterrows():
        key = _parameter_key(row["params"])
        if key in values:
            raise ValueError(f"Duplicate grid-search parameters: {key}")
        values[key] = np.asarray(
            [
                abs(float(row["mean_test_rmse"])),
                float(row["std_test_rmse"]),
                float(row["mean_test_r_squared"]),
                float(row["std_test_r_squared"]),
            ]
        )
    return values


def _gridsearch_report_values(path: Path, implementation: str) -> dict[str, np.ndarray]:
    report = load_report(path, implementation)
    if implementation == "localize":
        report = report.get("optimizer_data", {}).get("additional_data", {}).get("results")
    if not isinstance(report, pd.DataFrame):
        raise TypeError(f"Grid-search report is not a DataFrame: {path}")
    return _gridsearch_values(report)


def _automl_report_values(path: Path, implementation: str) -> dict[str, np.ndarray]:
    report = load_report(path, implementation)
    model_reports = report["model_data"]["reports"] if implementation == "localize" else report
    if not isinstance(model_reports, list):
        raise TypeError(f"AutoML report is not a list: {path}")

    values = {}
    for model_report in model_reports:
        key = _parameter_key(model_report["params"])
        if key in values:
            raise ValueError(f"Duplicate AutoML parameters: {key}")
        values[key] = np.asarray(
            [
                float(model_report["scores"][metric][statistic])
                for metric in ("rmse", "r_squared")
                for statistic in ("mean", "std")
            ]
        )
    return values


def _compare_values(
    left: dict[str, np.ndarray],
    right: dict[str, np.ndarray],
    kind: str,
    left_path: Path,
    right_path: Path,
) -> None:
    if left.keys() != right.keys():
        left_only = sorted(left.keys() - right.keys())
        right_only = sorted(right.keys() - left.keys())
        raise ValueError(
            f"{kind} candidates differ between {left_path} and {right_path}; "
            f"left only: {left_only}; right only: {right_only}"
        )

    mismatches = [
        key
        for key in left
        if not np.allclose(left[key], right[key], rtol=1e-6, atol=1e-8, equal_nan=True)
    ]
    if mismatches:
        raise ValueError(
            f"{kind} scores differ for {len(mismatches)} parameter sets between "
            f"{left_path} and {right_path}: {mismatches}"
        )


def _compare_gridsearch(
    left_path: Path,
    left_implementation: str,
    right_path: Path,
    right_implementation: str,
) -> None:
    _compare_values(
        _gridsearch_report_values(left_path, left_implementation),
        _gridsearch_report_values(right_path, right_implementation),
        "Grid-search",
        left_path,
        right_path,
    )


def _compare_automl(
    left_path: Path,
    left_implementation: str,
    right_path: Path,
    right_implementation: str,
) -> None:
    _compare_values(
        _automl_report_values(left_path, left_implementation),
        _automl_report_values(right_path, right_implementation),
        "AutoML",
        left_path,
        right_path,
    )


def _compare_reports(
    left_path: Path,
    right_path: Path,
    implementation: str,
) -> None:
    left = load_report(left_path, implementation)
    right = load_report(right_path, implementation)
    if implementation == "localize":
        left_grid = isinstance(
            left.get("optimizer_data", {}).get("additional_data", {}).get("results"),
            pd.DataFrame,
        )
        right_grid = isinstance(
            right.get("optimizer_data", {}).get("additional_data", {}).get("results"),
            pd.DataFrame,
        )
    else:
        left_grid = isinstance(left, pd.DataFrame)
        right_grid = isinstance(right, pd.DataFrame)

    if left_grid != right_grid:
        raise ValueError(f"Report types differ between {left_path} and {right_path}")
    compare = _compare_gridsearch if left_grid else _compare_automl
    compare(left_path, implementation, right_path, implementation)


def _run_consistency(evaluation_dir: Path, runs: int) -> dict[str, object]:
    results = {}
    for implementation in ("localize", "jupyter", "kedro"):
        failures = []
        total = 0
        reference_run = evaluation_dir / "benchmark" / implementation / "run-01"
        reference_reports = {
            str(path.relative_to(reference_run / "reports")): path
            for path in report_paths(reference_run)
        }

        for run_number in range(2, runs + 1):
            run = evaluation_dir / "benchmark" / implementation / f"run-{run_number:02d}"
            reports = {
                str(path.relative_to(run / "reports")): path
                for path in report_paths(run)
            }
            for relative_path in sorted(reference_reports.keys() | reports.keys()):
                total += 1
                if relative_path not in reference_reports or relative_path not in reports:
                    failures.append(
                        {
                            "reference_run": 1,
                            "run": run_number,
                            "report": relative_path,
                            "error": "Report is missing from one run",
                        }
                    )
                    continue
                try:
                    _compare_reports(
                        reference_reports[relative_path],
                        reports[relative_path],
                        implementation,
                    )
                except (KeyError, TypeError, ValueError) as error:
                    failures.append(
                        {
                            "reference_run": 1,
                            "run": run_number,
                            "report": relative_path,
                            "error": str(error),
                        }
                    )

        results[implementation] = {
            "status": "warning" if failures else "passed",
            "passed": total - len(failures),
            "failed": len(failures),
            "total": total,
            "failures": failures,
        }
    return results


def validate_benchmark_outputs(evaluation_dir: Path, runs: int) -> dict[str, object]:
    failures = []
    total = 0

    for run_number in range(1, runs + 1):
        localize_run = evaluation_dir / "benchmark" / "localize" / f"run-{run_number:02d}"
        jupyter_run = evaluation_dir / "benchmark" / "jupyter" / f"run-{run_number:02d}"
        kedro_run = evaluation_dir / "benchmark" / "kedro" / f"run-{run_number:02d}"
        validate_run(localize_run, "localize")
        validate_run(jupyter_run, "jupyter")
        validate_run(kedro_run, "kedro")

        for subset in SUBSETS:
            for split in SPLITS:
                for model in GRIDSEARCH_MODELS:
                    total += 1
                    try:
                        _compare_gridsearch(
                            find_report(
                                localize_run,
                                f"{subset}-{model}-{split}Split-results.pkl",
                            ),
                            "localize",
                            find_report(
                                jupyter_run,
                                f"Results_{model}-{split}Split-{subset}Subset.pkl",
                            ),
                            "jupyter",
                        )
                    except ValueError as error:
                        failures.append(
                            {
                                "run": run_number,
                                "stage": "gridsearch",
                                "subset": subset,
                                "split": split,
                                "model": model,
                                "error": str(error),
                            }
                        )

                total += 1
                try:
                    _compare_automl(
                        find_report(
                            localize_run,
                            f"{subset}-ExampleModel-{split}Split-results.pkl",
                        ),
                        "localize",
                        find_report(
                            jupyter_run,
                            f"Results_ExampleModel-{split}Split-{subset}Subset.pkl",
                        ),
                        "jupyter",
                    )
                except ValueError as error:
                    failures.append(
                        {
                            "run": run_number,
                            "stage": "automl",
                            "subset": subset,
                            "split": split,
                            "model": "ExampleModel",
                            "error": str(error),
                        }
                    )

    run_consistency = _run_consistency(evaluation_dir, runs)
    consistency_failures = sum(result["failed"] for result in run_consistency.values())

    return {
        "status": "warning" if failures or consistency_failures else "passed",
        "structure": "passed",
        "implementations": ["localize", "jupyter", "kedro"],
        "runs": runs,
        "localize_jupyter_comparisons": {
            "passed": total - len(failures),
            "failed": len(failures),
            "total": total,
            "failures": failures,
        },
        "run_consistency": run_consistency,
    }


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


def completed_runs(
    directory: Path,
    implementation: str,
    limit: int | None = None,
) -> list[Path]:
    if limit is None:
        runs = sorted(
            path
            for path in directory.glob("run-*")
            if path.is_dir() and path.name.removeprefix("run-").isdigit()
        )
    else:
        runs = [directory / f"run-{run_number:02d}" for run_number in range(1, limit + 1)]
    if not runs:
        raise FileNotFoundError(f"No completed runs found: {directory}")
    for run in runs:
        validate_run(run, implementation)
    return runs


def summarize_benchmark(
    evaluation_dir: Path,
    implementation_names: dict[str, str] | None = None,
    runs: int | None = None,
) -> pd.DataFrame:
    implementation_names = implementation_names or IMPLEMENTATION_NAMES
    rows = []
    for implementation, display_name in implementation_names.items():
        implementation_runs = completed_runs(
            evaluation_dir / "benchmark" / implementation,
            implementation,
            limit=runs,
        )
        for stage in STAGES:
            summaries = [summarize_stage(run, stage) for run in implementation_runs]
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
                    "implementation": display_name,
                    "stage": stage,
                    "runs": len(implementation_runs),
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
