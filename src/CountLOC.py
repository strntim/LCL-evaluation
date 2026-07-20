import argparse
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = (
    "00-Initial",
    "01-Changed_and_added_model",
    "02-Changed_dataset_to_logatec",
    "03-Added_split_and_metric",
    "04-Added_automl_model",
)
KEDRO_SNAPSHOTS = (
    "initial",
    "changed_model",
    "changed_dataset",
    "split_metric",
    "automl",
)


def is_blank(line: str) -> bool:
    return line.strip() == ""


def strip_comment(line: str) -> str:
    line = line.split("#", 1)[0]
    line = re.sub(r'("""|\'\'\')(.*?)\1', "", line)
    return line.rstrip()


def strip_structural(line: str) -> str:
    return "".join(token for token in line if token not in "{}[](),")


def is_meaningful_line(line: str) -> bool:
    if is_blank(line):
        return False
    if is_blank(strip_comment(line)):
        return False
    return not is_blank(strip_structural(line))


def strip_meaningless(lines: list[str]) -> list[str]:
    return [strip_comment(line) for line in lines if is_meaningful_line(line)]


def notebook_lines(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as file:
        notebook = json.load(file)

    lines = []
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code":
            lines.extend(cell.get("source", []))
            lines.append("\n")
    return lines


def config_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines(keepends=True)


def source_lines(directory: Path) -> list[str]:
    lines = []
    for path in sorted(directory.rglob("*.py")):
        lines.extend(path.read_text(encoding="utf-8").splitlines(keepends=True))
    return lines


def keep_unique_lines(lines: list[str], other: list[str]) -> list[str]:
    remaining = [line.strip() for line in other]
    unique = []
    for line in lines:
        stripped = line.strip()
        if stripped in remaining:
            remaining.remove(stripped)
        else:
            unique.append(line)
    return unique


def remove_experiment_references(lines: list[str], experiment: str) -> list[str]:
    return [line for line in lines if experiment not in line]


def count_change(before: list[str], after: list[str]) -> tuple[int, int]:
    deleted = keep_unique_lines(before, after)
    added = keep_unique_lines(after, before)
    return len(added), len(deleted)


def notebook_changes(root: Path) -> list[tuple[int, int]]:
    directory = root / "implementations" / "jupyter" / "notebooks"
    previous_name = EXPERIMENTS[0]
    previous = strip_meaningless(notebook_lines(directory / f"{previous_name}.ipynb"))
    previous = remove_experiment_references(previous, previous_name)
    changes = []

    for name in EXPERIMENTS[1:]:
        current = strip_meaningless(notebook_lines(directory / f"{name}.ipynb"))
        current = remove_experiment_references(current, name)
        changes.append(count_change(previous, current))
        previous = current

    return changes


def localize_changes(root: Path) -> list[tuple[int, int]]:
    directory = root / "implementations" / "localize" / "configs"
    totals = [[0, 0] for _ in EXPERIMENTS[1:]]

    for filename in ("dvc.yaml", "params.yaml"):
        previous_name = EXPERIMENTS[0]
        previous = strip_meaningless(config_lines(directory / previous_name / filename))
        previous = remove_experiment_references(previous, previous_name)

        for index, name in enumerate(EXPERIMENTS[1:]):
            current = strip_meaningless(config_lines(directory / name / filename))
            current = remove_experiment_references(current, name)
            added, deleted = count_change(previous, current)
            totals[index][0] += added
            totals[index][1] += deleted
            previous = current

    return [(added, deleted) for added, deleted in totals]


def kedro_changes(root: Path) -> list[tuple[int, int]]:
    implementation = root / "implementations" / "kedro"
    directory = implementation / "conf"
    totals = [[0, 0] for _ in EXPERIMENTS[1:]]

    for filename in ("catalog.yml", "parameters.yml"):
        previous_name = EXPERIMENTS[0]
        previous = strip_meaningless(config_lines(directory / previous_name / filename))
        previous = remove_experiment_references(previous, previous_name)

        for index, name in enumerate(EXPERIMENTS[1:]):
            current = strip_meaningless(config_lines(directory / name / filename))
            current = remove_experiment_references(current, name)
            added, deleted = count_change(previous, current)
            totals[index][0] += added
            totals[index][1] += deleted
            previous = current

    snapshots = implementation / "src" / "lcl_evaluation_kedro" / "pipelines" / "experiments"
    previous = strip_meaningless(source_lines(snapshots / KEDRO_SNAPSHOTS[0]))
    for index, snapshot in enumerate(KEDRO_SNAPSHOTS[1:]):
        current = strip_meaningless(source_lines(snapshots / snapshot))
        added, deleted = count_change(previous, current)
        totals[index][0] += added
        totals[index][1] += deleted
        previous = current

    return [(added, deleted) for added, deleted in totals]


def calculate_loc_changes(root: Path = ROOT) -> list[dict[str, int | str]]:
    notebook = notebook_changes(root)
    localize = localize_changes(root)
    kedro = kedro_changes(root)
    rows = []

    for index, (before, after) in enumerate(zip(EXPERIMENTS, EXPERIMENTS[1:]), start=1):
        localize_added, localize_deleted = localize[index - 1]
        notebook_added, notebook_deleted = notebook[index - 1]
        kedro_added, kedro_deleted = kedro[index - 1]
        rows.append(
            {
                "change": index,
                "from": before,
                "to": after,
                "localize_added": localize_added,
                "localize_deleted": localize_deleted,
                "notebook_added": notebook_added,
                "notebook_deleted": notebook_deleted,
                "kedro_added": kedro_added,
                "kedro_deleted": kedro_deleted,
            }
        )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = calculate_loc_changes()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")

    if args.json:
        print(json.dumps(rows, indent=2))
        return

    for row in rows:
        print(f"Change {row['change']}: {row['from']} -> {row['to']}")
        print(f"  LOCALIZE  +{row['localize_added']} -{row['localize_deleted']}")
        print(f"  Notebook  +{row['notebook_added']} -{row['notebook_deleted']}")
        print(f"  Kedro     +{row['kedro_added']} -{row['kedro_deleted']}")


if __name__ == "__main__":
    main()
