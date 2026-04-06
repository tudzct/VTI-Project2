#!/usr/bin/env python3
"""Prepare a reproducible VTI dataset from the raw BigVul CSV."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vti_repro.data_prep import prepare_vti_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-csv", default="MSR_data_cleaned.csv")
    parser.add_argument("--output-dir", default="artifacts/data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--report-every", type=int, default=5000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = prepare_vti_dataset(
        raw_csv_path=args.raw_csv,
        output_dir=args.output_dir,
        seed=args.seed,
        max_rows=args.max_rows,
        report_every=args.report_every,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
