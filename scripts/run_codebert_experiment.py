#!/usr/bin/env python3
"""Run a short local CodeBERT experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vti_repro.codebert_pipeline import CodeBERTConfig, run_codebert_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="artifacts/data_sample/train.csv.gz")
    parser.add_argument("--val", default="artifacts/data_sample/val.csv.gz")
    parser.add_argument("--test", default="artifacts/data_sample/test.csv.gz")
    parser.add_argument("--output-dir", default="artifacts/codebert_sample")
    parser.add_argument("--max-train-rows", type=int, default=256)
    parser.add_argument("--max-val-rows", type=int, default=64)
    parser.add_argument("--max-test-rows", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--model-name", default="microsoft/codebert-base")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def _none_if_non_positive(value: int | None) -> int | None:
    if value is None:
        return None
    return None if value <= 0 else value


def main() -> int:
    args = parse_args()
    config = CodeBERTConfig(
        model_name=args.model_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        max_train_rows=_none_if_non_positive(args.max_train_rows),
        max_val_rows=_none_if_non_positive(args.max_val_rows),
        max_test_rows=_none_if_non_positive(args.max_test_rows),
    )
    summary = run_codebert_experiment(
        train_path=args.train,
        val_path=args.val,
        test_path=args.test,
        output_dir=args.output_dir,
        config=config,
        local_files_only=args.local_files_only,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
