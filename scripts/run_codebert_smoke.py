#!/usr/bin/env python3
"""Offline smoke test for the CodeBERT pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vti_repro.codebert_pipeline import CodeBERTConfig, run_codebert_smoke


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="artifacts/data_sample/test.csv.gz")
    parser.add_argument("--model-name", default="microsoft/codebert-base")
    parser.add_argument("--sample-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow transformers to download model files if they are not cached locally.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = CodeBERTConfig(
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    try:
        result = run_codebert_smoke(
            data_path=args.data,
            config=config,
            sample_size=args.sample_size,
            local_files_only=not args.allow_download,
        )
    except OSError as exc:
        raise SystemExit(
            "CodeBERT model files are not available locally. "
            "Download them first or rerun with --allow-download in an environment with internet access.\n"
            f"Original error: {exc}"
        ) from exc
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
