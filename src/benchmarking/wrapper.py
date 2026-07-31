#!/usr/bin/env python3

import argparse
import subprocess
from pathlib import Path

from performance import start_resource_monitor, stop_resource_monitor


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("provide a command after --")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    start_resource_monitor()
    try:
        return subprocess.run(command, check=False).returncode
    finally:
        stop_resource_monitor(args.output)


if __name__ == "__main__":
    raise SystemExit(main())
