from pathlib import Path

from kedro.framework.cli.utils import KedroCliError
from kedro.framework.startup import bootstrap_project


def main() -> None:
    try:
        metadata = bootstrap_project(Path.cwd())
    except RuntimeError as error:
        raise KedroCliError(str(error)) from error

    from kedro.framework.cli.cli import KedroCLI

    KedroCLI(metadata.project_path)()


if __name__ == "__main__":
    main()

