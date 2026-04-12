#!/usr/bin/env python3
"""Run notebook-compatible Enhanced refinement."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vti_repro.enhanced_notebook_compat import NotebookEnhancedConfig, run_notebook_compatible_enhanced


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--raw-preds", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prediction-threshold", type=float, default=0.3)
    parser.add_argument("--freq-threshold", type=float, default=1.5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_notebook_compatible_enhanced(
        train_path=args.train,
        test_path=args.test,
        raw_preds_path=args.raw_preds,
        labels_path=args.labels,
        output_dir=args.output_dir,
        config=NotebookEnhancedConfig(
            prediction_threshold=args.prediction_threshold,
            freq_threshold=args.freq_threshold,
        ),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
