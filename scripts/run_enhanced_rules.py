#!/usr/bin/env python3
"""Run the enhancement stage on an experiment's raw predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vti_repro.enhanced_pipeline import EnhancedConfig, run_enhanced_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="artifacts/data_sample/train.csv.gz")
    parser.add_argument("--test", default="artifacts/data_sample/test.csv.gz")
    parser.add_argument("--raw-preds", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output-dir", default="artifacts/enhanced_sample")
    parser.add_argument("--prediction-threshold", type=float, default=0.5)
    parser.add_argument("--freq-threshold", type=float, default=1.5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = EnhancedConfig(
        prediction_threshold=args.prediction_threshold,
        freq_threshold=args.freq_threshold,
    )
    summary = run_enhanced_experiment(
        train_path=args.train,
        test_path=args.test,
        raw_preds_path=args.raw_preds,
        labels_path=args.labels,
        output_dir=args.output_dir,
        config=config,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
