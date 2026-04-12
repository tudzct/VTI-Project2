#!/usr/bin/env python3
"""Run the Word2Vec + BiLSTM + Attention experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

vendor_dir = Path(__file__).resolve().parents[1] / ".vendor"
if vendor_dir.exists():
    sys.path.insert(0, str(vendor_dir))

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vti_repro.word2vec_pipeline import Word2VecConfig, run_word2vec_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="artifacts/data_sample/train.csv.gz")
    parser.add_argument("--val", default="artifacts/data_sample/val.csv.gz")
    parser.add_argument("--test", default="artifacts/data_sample/test.csv.gz")
    parser.add_argument("--output-dir", default="artifacts/word2vec_sample")
    parser.add_argument("--max-train-rows", type=int, default=4096)
    parser.add_argument("--max-val-rows", type=int, default=512)
    parser.add_argument("--max-test-rows", type=int, default=512)
    parser.add_argument("--nn-epochs", type=int, default=2)
    parser.add_argument("--w2v-epochs", type=int, default=10)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--vector-size", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def _none_if_non_positive(value: int | None) -> int | None:
    if value is None:
        return None
    return None if value <= 0 else value


def main() -> int:
    args = parse_args()
    config = Word2VecConfig(
        vector_size=args.vector_size,
        w2v_epochs=args.w2v_epochs,
        nn_epochs=args.nn_epochs,
        max_len=args.max_len,
        batch_size=args.batch_size,
        max_train_rows=_none_if_non_positive(args.max_train_rows),
        max_val_rows=_none_if_non_positive(args.max_val_rows),
        max_test_rows=_none_if_non_positive(args.max_test_rows),
    )
    summary = run_word2vec_experiment(
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
