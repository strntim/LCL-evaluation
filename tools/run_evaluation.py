#!/usr/bin/env python3

import argparse
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
RESULTS = ROOT / "results" / "evaluations"
LOCALIZE_ARTIFACTS = ROOT / "implementations" / "localize" / "install" / "artifacts"
JUPYTER = ROOT / "implementations" / "jupyter"
JUPYTER_BENCHMARK = JUPYTER / "results" / "artifacts" / "Benchmarking-results"

sys.path.insert(0, str(SRC))


def timestamp() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d%H%M")


def iso_time() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def command_text(command: list[str]) -> str:
    return " ".join(command)


def run_command(command: list[str]) -> None:
    print(f"$ {command_text(command)}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def remove(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def clean_localize(name: str) -> None:
    artifact = LOCALIZE_ARTIFACTS / name
    remove(artifact / "models")
    remove(artifact / "reports" / "usage")


def clean_jupyter() -> None:
    remove(JUPYTER_BENCHMARK)
    remove(JUPYTER / "results" / "Benchmarking-executed.ipynb")


def copy_usage(source: Path, destination: Path) -> None:
    paths = sorted(source.rglob("*_usage.pkl")) if source.is_dir() else []
    if not paths:
        raise FileNotFoundError(f"No usage logs found: {source}")
    destination.mkdir(parents=True, exist_ok=True)
    for path in paths:
        relative = path.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def copy_reports(source: Path, destination: Path) -> None:
    paths = sorted(
        path
        for path in source.rglob("*.pkl")
        if "result" in path.name.lower()
    ) if source.is_dir() else []
    if not paths:
        raise FileNotFoundError(f"No result reports found: {source}")
    destination.mkdir(parents=True, exist_ok=True)
    for path in paths:
        relative = path.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def completed(path: Path, implementation: str) -> bool:
    from evaluation_data import validate_run

    if not path.is_dir():
        return False
    try:
        validate_run(path, implementation)
    except Exception:
        return False
    return True


def collect_unit(
    destination: Path,
    implementation: str,
    command: list[str],
    clean,
    usage_source: Path,
    report_source: Path,
    executed_notebook: Path | None = None,
) -> None:
    from evaluation_data import validate_run

    if completed(destination, implementation):
        print(f"Already complete: {destination}")
        return

    if destination.exists():
        remove(destination)
    partial = destination.with_name(f"{destination.name}.partial")
    if partial.exists():
        remove(partial)
    partial.mkdir(parents=True)
    write_json(partial / "started.json", {"started": iso_time(), "command": command})

    clean()
    run_command(command)
    copy_usage(usage_source, partial / "usage")
    copy_reports(report_source, partial / "reports")
    if executed_notebook is not None:
        if not executed_notebook.is_file():
            raise FileNotFoundError(f"Executed notebook not found: {executed_notebook}")
        shutil.copy2(executed_notebook, partial / "executed.ipynb")

    write_json(partial / "complete.json", {"completed": iso_time(), "command": command})
    validate_run(partial, implementation)
    partial.rename(destination)
    print(f"Completed: {destination}")


def localize_benchmark(session: Path, run_number: int) -> None:
    artifact = LOCALIZE_ARTIFACTS / "Benchmarking"
    collect_unit(
        session / "benchmark" / "localize" / f"run-{run_number:02d}",
        "localize",
        ["bash", str(ROOT / "run_localize.sh"), "Benchmarking", "--force"],
        lambda: clean_localize("Benchmarking"),
        artifact / "reports" / "usage",
        artifact / "models",
    )


def jupyter_benchmark(session: Path, run_number: int) -> None:
    collect_unit(
        session / "benchmark" / "jupyter" / f"run-{run_number:02d}",
        "jupyter",
        [sys.executable, str(JUPYTER / "run_notebooks.py"), "Benchmarking"],
        clean_jupyter,
        JUPYTER_BENCHMARK,
        JUPYTER_BENCHMARK,
        JUPYTER / "results" / "Benchmarking-executed.ipynb",
    )


def scalability(session: Path, factor: int) -> None:
    artifact = LOCALIZE_ARTIFACTS / "Scalability"
    collect_unit(
        session / "scalability" / f"{factor}x",
        "localize",
        [
            "bash",
            str(ROOT / "run_localize.sh"),
            "Scalability",
            "--scale",
            str(factor),
            "--force",
        ],
        lambda: clean_localize("Scalability"),
        artifact / "reports" / "usage",
        artifact / "models",
    )


def create_session(args: argparse.Namespace) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    if args.resume:
        session = RESULTS / args.resume
        if not session.is_dir():
            raise FileNotFoundError(f"Evaluation session not found: {session}")
        return session

    session = RESULTS / timestamp()
    if session.exists():
        raise FileExistsError(f"Evaluation session already exists; use --resume {session.name}")
    session.mkdir(parents=True)
    return session


def update_metadata(session: Path, runs: int, status: str) -> None:
    path = session / "metadata.json"
    if path.is_file():
        metadata = json.loads(path.read_text(encoding="utf-8"))
        if metadata.get("benchmark_runs") != runs:
            raise ValueError(
                f"Session {session.name} was created with "
                f"{metadata.get('benchmark_runs')} benchmark runs, not {runs}"
            )
    else:
        metadata = {
            "session": session.name,
            "created": iso_time(),
            "benchmark_runs": runs,
            "system": platform.platform(),
            "python": platform.python_version(),
        }
    metadata["status"] = status
    metadata["updated"] = iso_time()
    write_json(path, metadata)


def print_dry_run(runs: int) -> None:
    for run_number in range(1, runs + 1):
        print(f"run-{run_number:02d}: LOCALIZE Benchmarking --force")
        print(f"run-{run_number:02d}: Jupyter Benchmarking")
    for factor in (1, 5, 10):
        print(f"scalability: LOCALIZE {factor}x --force")
    print("calculate loc.json")
    print("generate paper figures")


def main() -> int:
    from CountLOC import calculate_loc_changes

    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--resume")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs must be at least 1")
    if args.dry_run:
        print_dry_run(args.runs)
        return 0

    session = create_session(args)
    print(f"Evaluation session: {session}", flush=True)
    update_metadata(session, args.runs, "running")

    try:
        for run_number in range(1, args.runs + 1):
            localize_benchmark(session, run_number)
            jupyter_benchmark(session, run_number)

        for factor in (1, 5, 10):
            scalability(session, factor)

        write_json(session / "loc.json", calculate_loc_changes(ROOT))
        run_command(
            [
                sys.executable,
                str(SRC / "plot.py"),
                "--evaluation-dir",
                str(session),
            ]
        )
    except BaseException:
        update_metadata(session, args.runs, "incomplete")
        raise

    update_metadata(session, args.runs, "complete")
    (RESULTS / "latest.txt").write_text(session.name + "\n", encoding="utf-8")
    print(f"Evaluation complete: {session}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
