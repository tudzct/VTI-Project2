#!/usr/bin/env python3
"""Build legacy-compatible views from the current sample dataset splits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vti_repro.data_views import ViewConfig, build_views


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="artifacts/data_sample/train.csv.gz")
    parser.add_argument("--val", default="artifacts/data_sample/val.csv.gz")
    parser.add_argument("--test", default="artifacts/data_sample/test.csv.gz")
    parser.add_argument("--output-dir", default="artifacts/legacy_views")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs = build_views(
        ViewConfig(
            train_path=args.train,
            val_path=args.val,
            test_path=args.test,
            output_dir=args.output_dir,
        )
    )
    print(json.dumps(outputs, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
