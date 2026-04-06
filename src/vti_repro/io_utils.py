"""Input/output helpers."""

from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path
from typing import Dict, Iterable


def ensure_directory(path: str | Path) -> Path:
    output_path = Path(path)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def open_csv_writer(path: Path, fieldnames: Iterable[str]) -> tuple[object, csv.DictWriter]:
    handle = gzip.open(path, "wt", encoding="utf-8", newline="") if path.suffix == ".gz" else path.open(
        "w",
        encoding="utf-8",
        newline="",
    )
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    return handle, writer


def dump_json(data: Dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
