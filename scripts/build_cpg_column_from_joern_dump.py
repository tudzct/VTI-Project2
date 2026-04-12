#!/usr/bin/env python3
"""Build notebook-compatible cpg strings from a Joern TSV dump."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-tsv", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--source-csv")
    parser.add_argument("--output-csv", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    grouped = defaultdict(list)
    with open(args.dump_tsv, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line or "\t" not in line or line.startswith("executing ") or line.startswith("Creating project "):
                continue
            parts = line.split("\t", 3)
            if len(parts) != 4:
                continue
            filename, label, value, code = parts
            if label == "CALL":
                grouped[filename].append({"_label": "CALL", "name": value, "code": code})
            elif label == "IDENTIFIER":
                grouped[filename].append({"_label": "IDENTIFIER", "name": value, "code": code})
            elif label == "CONTROL_STRUCTURE":
                grouped[filename].append({"_label": "CONTROL_STRUCTURE", "name": value, "code": code})

    manifest = pd.read_csv(args.manifest)
    filename_col = manifest["source_path"].map(lambda value: Path(value).as_posix().split("/")[-2] + "/" + Path(value).name)
    manifest = manifest.copy()
    manifest["joern_filename"] = filename_col
    manifest["cpg"] = manifest["joern_filename"].map(lambda name: json.dumps(grouped.get(name, [])) + "--====--")
    output_df = manifest
    if args.source_csv:
      source_df = pd.read_csv(args.source_csv, compression="infer")
      source_df["sample_id"] = source_df["sample_id"].astype(str)
      output_df["sample_id"] = output_df["sample_id"].astype(str)
      output_df = source_df.merge(
          output_df[["sample_id", "source_path", "joern_filename", "cpg"]],
          on="sample_id",
          how="left",
      )
    output_df.to_csv(args.output_csv, index=False)
    print(f"wrote {args.output_csv} with {len(manifest)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
