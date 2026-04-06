#!/usr/bin/env python3
"""Run the classical baseline experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vti_repro.base_pipeline import BaseConfig, run_base_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="artifacts/data/train.csv.gz")
    parser.add_argument("--val", default="artifacts/data/val.csv.gz")
    parser.add_argument("--test", default="artifacts/data/test.csv.gz")
    parser.add_argument("--output-dir", default="artifacts/base")
    parser.add_argument("--max-features", type=int, default=20_000)
    parser.add_argument(
        "--p-value-grid",
        default="0.5,0.25,0.1,0.05,0.01,0.005,0.001",
        help="Comma-separated chi-square p-value thresholds.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = BaseConfig(
        max_features=args.max_features,
        p_value_grid=tuple(float(item) for item in args.p_value_grid.split(",") if item),
    )
    summary = run_base_experiment(
        train_path=args.train,
        val_path=args.val,
        test_path=args.test,
        output_dir=args.output_dir,
        config=config,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
