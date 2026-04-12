"""Streaming dataset preparation for BigVul-based VTI experiments."""

from __future__ import annotations

import csv
import hashlib
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict

from .constants import LABEL_COLUMNS
from .io_utils import dump_json, ensure_directory, open_csv_writer
from .labels import has_any_label, normalize_labels
from .preprocessing import minimal_clean_text


OUTPUT_FIELDS = [
    "sample_id",
    "split",
    "text",
    "raw_classification",
    "cve_id",
    "cwe_id",
    "project",
    "commit_id",
    "vul",
    *LABEL_COLUMNS,
]


def assign_split(sample_id: str, seed: int) -> str:
    digest = hashlib.sha1(f"{seed}:{sample_id}".encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 10
    if bucket < 8:
        return "train"
    if bucket == 8:
        return "val"
    return "test"


def prepare_vti_dataset(
    raw_csv_path: str | Path,
    output_dir: str | Path,
    seed: int = 42,
    max_rows: int | None = None,
    report_every: int = 5000,
) -> Dict:
    max_csv_field_size = sys.maxsize
    while True:
        try:
            csv.field_size_limit(max_csv_field_size)
            break
        except OverflowError:
            max_csv_field_size //= 10

    output_dir = ensure_directory(output_dir)
    clean_path = output_dir / "vti_dataset.csv.gz"
    split_paths = {
        "train": output_dir / "train.csv.gz",
        "val": output_dir / "val.csv.gz",
        "test": output_dir / "test.csv.gz",
    }

    clean_handle, clean_writer = open_csv_writer(clean_path, OUTPUT_FIELDS)
    split_handles = {}
    split_writers = {}
    for split_name, split_path in split_paths.items():
        handle, writer = open_csv_writer(split_path, OUTPUT_FIELDS)
        split_handles[split_name] = handle
        split_writers[split_name] = writer

    stats = {
        "seed": seed,
        "raw_csv_path": str(raw_csv_path),
        "processed_rows": 0,
        "kept_rows": 0,
        "skipped_no_label": 0,
        "skipped_empty_text": 0,
        "split_sizes": Counter(),
        "label_totals": Counter(),
        "label_by_split": defaultdict(Counter),
        "top_raw_classifications": Counter(),
    }

    raw_path = Path(raw_csv_path)
    with raw_path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw_index, row in enumerate(reader):
            stats["processed_rows"] += 1
            if max_rows is not None and stats["processed_rows"] > max_rows:
                break

            raw_classification = (row.get("Vulnerability Classification") or "").strip()
            label_record = normalize_labels(raw_classification)
            if not has_any_label(label_record):
                stats["skipped_no_label"] += 1
                continue

            text = minimal_clean_text(row.get("func_before") or "")
            if not text:
                stats["skipped_empty_text"] += 1
                continue

            sample_id = str(raw_index)
            split = assign_split(sample_id, seed)
            output_row = {
                "sample_id": sample_id,
                "split": split,
                "text": text,
                "raw_classification": raw_classification,
                "cve_id": row.get("CVE ID", ""),
                "cwe_id": row.get("CWE ID", ""),
                "project": row.get("project", ""),
                "commit_id": row.get("commit_id", ""),
                "vul": row.get("vul", ""),
                **label_record,
            }

            clean_writer.writerow(output_row)
            split_writers[split].writerow(output_row)

            stats["kept_rows"] += 1
            stats["split_sizes"][split] += 1
            stats["top_raw_classifications"][raw_classification] += 1
            for label in LABEL_COLUMNS:
                if label_record[label]:
                    stats["label_totals"][label] += 1
                    stats["label_by_split"][split][label] += 1

            if report_every and stats["kept_rows"] and stats["kept_rows"] % report_every == 0:
                print(
                    f"[prepare_vti_dataset] kept={stats['kept_rows']} "
                    f"processed={stats['processed_rows']} split={dict(stats['split_sizes'])}"
                )

    clean_handle.close()
    for handle in split_handles.values():
        handle.close()

    summary = {
        "seed": stats["seed"],
        "raw_csv_path": stats["raw_csv_path"],
        "processed_rows": stats["processed_rows"],
        "kept_rows": stats["kept_rows"],
        "skipped_no_label": stats["skipped_no_label"],
        "skipped_empty_text": stats["skipped_empty_text"],
        "split_sizes": dict(stats["split_sizes"]),
        "label_totals": dict(stats["label_totals"]),
        "label_by_split": {split: dict(counter) for split, counter in stats["label_by_split"].items()},
        "top_raw_classifications": dict(stats["top_raw_classifications"].most_common(25)),
        "outputs": {
            "dataset": str(clean_path),
            "train": str(split_paths["train"]),
            "val": str(split_paths["val"]),
            "test": str(split_paths["test"]),
        },
    }
    dump_json(summary, output_dir / "summary.json")
    return summary
