#!/usr/bin/env python3
"""Watch full-run artifact directories and launch Joern-backed Enhanced runs."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-cpg",
        default="artifacts/joern_cpg/train_subset_16384_with_cpg_fullrows.csv",
    )
    parser.add_argument(
        "--test-cpg",
        default="artifacts/joern_cpg/test_full_with_cpg_fullrows.csv",
    )
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--max-polls", type=int, default=0)
    parser.add_argument("--log-file", default="artifacts/enhanced_full_watch.log")
    return parser.parse_args()


def _prediction_inputs(model_dir: Path) -> tuple[Path, Path] | None:
    raw_preds = model_dir / "raw_preds.csv"
    labels = model_dir / "labels.csv"
    if raw_preds.exists() and labels.exists():
        return raw_preds, labels
    prediction_table = model_dir / "test_predictions.csv"
    if prediction_table.exists():
        cmd = [
            sys.executable,
            "scripts/split_prediction_table.py",
            "--input",
            str(prediction_table),
            "--output-dir",
            str(model_dir),
        ]
        print(f"[watch] preparing raw_preds/labels via {' '.join(cmd)}", flush=True)
        subprocess.call(cmd)
        if raw_preds.exists() and labels.exists():
            return raw_preds, labels
    return None


def _run_enhanced(
    train_cpg: Path,
    test_cpg: Path,
    raw_preds: Path,
    labels: Path,
    output_dir: Path,
) -> int:
    cmd = [
        sys.executable,
        "scripts/run_enhanced_notebook_compat.py",
        "--train",
        str(train_cpg),
        "--test",
        str(test_cpg),
        "--raw-preds",
        str(raw_preds),
        "--labels",
        str(labels),
        "--output-dir",
        str(output_dir),
    ]
    print(f"[watch] launching {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd)


def _log(log_path: Path, message: str) -> None:
    timestamp = datetime.now().isoformat(timespec="seconds")
    line = f"[{timestamp}] {message}\n"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line)
    print(line, end="", flush=True)


def main() -> int:
    args = parse_args()
    train_cpg = Path(args.train_cpg)
    test_cpg = Path(args.test_cpg)
    log_path = Path(args.log_file)

    if not train_cpg.exists():
        raise FileNotFoundError(f"Missing train CPG csv: {train_cpg}")
    if not test_cpg.exists():
        raise FileNotFoundError(f"Missing test CPG csv: {test_cpg}")

    _log(log_path, f"watcher start train_cpg={train_cpg} test_cpg={test_cpg}")

    targets = {
        "base": {
            "model_dir": Path("artifacts/base_full"),
            "output_dir": Path("artifacts/enhanced_base_full_joern"),
        },
        "base_p05": {
            "model_dir": Path("artifacts/base_full_p05"),
            "output_dir": Path("artifacts/enhanced_base_full_p05_joern"),
        },
        "base_p05_v2": {
            "model_dir": Path("artifacts/base_full_p05_v2"),
            "output_dir": Path("artifacts/enhanced_base_full_p05_v2_joern"),
        },
        "word2vec": {
            "model_dir": Path("artifacts/word2vec_full"),
            "output_dir": Path("artifacts/enhanced_word2vec_full_joern"),
        },
        "codebert": {
            "model_dir": Path("artifacts/codebert_full"),
            "output_dir": Path("artifacts/enhanced_codebert_full_joern"),
        },
    }

    completed: set[str] = set()
    polls = 0
    while len(completed) < len(targets):
        for name, spec in targets.items():
            if name in completed:
                continue
            model_dir = spec["model_dir"]
            output_dir = spec["output_dir"]
            metrics_path = output_dir / "metrics.json"
            if metrics_path.exists():
                _log(log_path, f"{name}: enhanced metrics already exist")
                completed.add(name)
                continue

            inputs = _prediction_inputs(model_dir)
            if inputs is None:
                _log(log_path, f"{name}: waiting for raw_preds.csv and labels.csv in {model_dir}")
                continue

            raw_preds, labels = inputs
            output_dir.mkdir(parents=True, exist_ok=True)
            exit_code = _run_enhanced(train_cpg, test_cpg, raw_preds, labels, output_dir)
            if exit_code == 0 and metrics_path.exists():
                _log(log_path, f"{name}: enhanced run completed")
                completed.add(name)
            else:
                _log(log_path, f"{name}: enhanced run failed with exit code {exit_code}")

        polls += 1
        if len(completed) == len(targets):
            break
        if args.max_polls and polls >= args.max_polls:
            _log(log_path, "reached max polls before all outputs became ready")
            break
        time.sleep(args.poll_seconds)

    _log(log_path, f"completed targets: {sorted(completed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
