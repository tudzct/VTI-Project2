#!/usr/bin/env python3
"""Split a prediction table with pred_/true_ columns into raw_preds and labels CSVs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vti_repro.constants import LABEL_COLUMNS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pred-prefix", default="pred_")
    parser.add_argument("--true-prefix", default="true_")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    df = pd.read_csv(args.input)
    pred_cols = [f"{args.pred_prefix}{label}" for label in LABEL_COLUMNS]
    true_cols = [f"{args.true_prefix}{label}" for label in LABEL_COLUMNS]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_preds = df[pred_cols].copy()
    raw_preds.columns = LABEL_COLUMNS
    labels = df[true_cols].copy()
    labels.columns = LABEL_COLUMNS

    raw_preds.to_csv(output_dir / "raw_preds.csv", index=False)
    labels.to_csv(output_dir / "labels.csv", index=False)
    print(f"wrote {output_dir / 'raw_preds.csv'} and {output_dir / 'labels.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
