#!/usr/bin/env python3
"""Placeholder entrypoint for the Word2Vec experiment."""

from __future__ import annotations

import sys
from pathlib import Path

vendor_dir = Path(__file__).resolve().parents[1] / ".vendor"
if vendor_dir.exists():
    sys.path.insert(0, str(vendor_dir))

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vti_repro.word2vec_pipeline import ensure_word2vec_dependencies


def main() -> int:
    try:
        ensure_word2vec_dependencies()
    except ModuleNotFoundError as exc:
        raise SystemExit(f"Missing dependency for Word2Vec experiment: {exc}") from exc
    raise NotImplementedError(
        "gensim is now discoverable if it exists in .vendor, but the full "
        "Word2Vec + BiLSTM + attention training loop still needs to be completed "
        "in a single environment that also has torch."
    )


if __name__ == "__main__":
    raise SystemExit(main())
