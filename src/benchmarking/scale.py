#!/usr/bin/env python3

import argparse
from pathlib import Path

import joblib
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--factor", type=int, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.factor not in (1, 5, 10):
        parser.error("factor must be 1, 5, or 10")

    frame = joblib.load(args.input)
    scaled = pd.concat([frame] * args.factor, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaled, args.output, compress=9)


if __name__ == "__main__":
    main()
