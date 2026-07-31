#!/usr/bin/env python3

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path
from shutil import copyfile

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[2]
JUPYTER = ROOT / "implementations" / "jupyter"
NOTEBOOKS = JUPYTER / "notebooks"
RESULTS = JUPYTER / "results"
DATASETS = ROOT / "datasets"
BENCHMARK = "Benchmarking"


def notebook_names() -> list[str]:
    return sorted(path.stem for path in NOTEBOOKS.glob("*.ipynb"))


def prepare_datasets() -> None:
    archives = (
        (DATASETS / "umu" / "umu.zip", (DATASETS / "umu" / "umu",)),
        (
            DATASETS / "logatec" / "logatec.zip",
            (DATASETS / "logatec" / "spring_data.json", DATASETS / "logatec" / "winter_data.json"),
        ),
    )
    for archive, expected in archives:
        if all(path.exists() for path in expected):
            continue
        if not archive.is_file():
            raise FileNotFoundError(f"Dataset archive not found: {archive}")
        with zipfile.ZipFile(archive) as file:
            file.extractall(archive.parent)


def execute(name: str) -> int:
    source = NOTEBOOKS / f"{name}.ipynb"
    if not source.is_file():
        raise ValueError(f"Unknown notebook: {name}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    destination = RESULTS / f"{name}-executed.ipynb"
    copyfile(source, destination)

    with destination.open(encoding="utf-8") as file:
        notebook = nbformat.read(file, as_version=4)

    client = NotebookClient(
        notebook,
        kernel_name="jupyter",
        timeout=None,
        resources={"metadata": {"path": str(RESULTS)}},
    )
    client.execute()

    with destination.open("w", encoding="utf-8") as file:
        nbformat.write(notebook, file)

    print(f"Completed {name}")
    return 0


def main() -> int:
    names = notebook_names()
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", nargs="?", choices=names)
    parser.add_argument("-p", "--parallel", action="store_true")
    parser.add_argument("--run-one", choices=names, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.run_one:
        return execute(args.run_one)

    prepare_datasets()

    if args.parallel and args.notebook:
        parser.error("--parallel cannot be used with a notebook name")

    if args.notebook:
        return execute(args.notebook)

    regular_notebooks = [name for name in names if name != BENCHMARK]
    if args.parallel:
        processes = [
            subprocess.Popen([sys.executable, __file__, "--run-one", name])
            for name in regular_notebooks
        ]
        failed = any(process.wait() != 0 for process in processes)
        if failed:
            return 1
        return execute(BENCHMARK)

    for name in regular_notebooks:
        if execute(name) != 0:
            return 1
    return execute(BENCHMARK)


if __name__ == "__main__":
    raise SystemExit(main())
