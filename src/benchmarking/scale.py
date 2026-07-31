#!/usr/bin/env python3

import argparse
import json
import math
from pathlib import Path


def timestamps(data: dict) -> list[float]:
    return [
        sample["timestamp"]
        for measurements in data.values()
        for samples in measurements.values()
        for sample in samples
    ]


def scale(data: dict, factor: int) -> dict:
    values = timestamps(data)
    if not values:
        return data

    period = math.floor(max(values)) - math.floor(min(values)) + 1
    scaled = {}
    for position, measurements in data.items():
        scaled[position] = {}
        for device, samples in measurements.items():
            scaled[position][device] = [
                {**sample, "timestamp": sample["timestamp"] + copy * period}
                for copy in range(factor)
                for sample in samples
            ]
    return scaled


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--factor", type=int, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.factor not in (1, 5, 10):
        parser.error("factor must be 1, 5, or 10")

    with args.input.open(encoding="utf-8") as file:
        data = json.load(file)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(scale(data, args.factor), file)


if __name__ == "__main__":
    main()
