#!/usr/bin/env python3
"""Materialize dataset rows as C source files for Joern parsing."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--extension", default=".cpp")
    return parser.parse_args()


def _safe_project_name(value: str) -> str:
    value = value or "unknown_project"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "unknown_project"


def _ensure_c_source(text: str) -> str:
    body = text or ""
    if "#include" not in body:
        return "#include <stdint.h>\n#include <stddef.h>\n\n" + body + "\n"
    return body + ("\n" if not body.endswith("\n") else "")


def main() -> int:
    args = parse_args()
    df = pd.read_csv(args.input, compression="infer")
    if args.max_rows is not None:
        df = df.head(args.max_rows)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    for row in df.itertuples(index=False):
        project_dir = output_dir / _safe_project_name(getattr(row, "project", ""))
        project_dir.mkdir(parents=True, exist_ok=True)
        sample_id = str(getattr(row, "sample_id"))
        extension = args.extension if args.extension.startswith(".") else f".{args.extension}"
        source_path = project_dir / f"{sample_id}{extension}"
        source_path.write_text(_ensure_c_source(getattr(row, "text")), encoding="utf-8")
        manifest_rows.append(
            {
                "sample_id": sample_id,
                "project": getattr(row, "project", ""),
                "source_path": str(source_path),
                "raw_classification": getattr(row, "raw_classification", ""),
            }
        )

    pd.DataFrame(manifest_rows).to_csv(output_dir / "manifest.csv", index=False)
    print(f"materialized {len(manifest_rows)} sources into {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
