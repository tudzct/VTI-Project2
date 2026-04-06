from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vti_repro.base_pipeline import run_base_experiment
from vti_repro.constants import LABEL_COLUMNS


def make_row(sample_id: str, text: str, active_labels: list[str]) -> dict:
    row = {
        "sample_id": sample_id,
        "split": "train",
        "text": text,
        "raw_classification": " ".join(active_labels),
        "cve_id": "",
        "cwe_id": "",
        "project": "",
        "commit_id": "",
        "vul": 1,
    }
    for label in LABEL_COLUMNS:
        row[label] = int(label in active_labels)
    return row


class BasePipelineSmokeTests(unittest.TestCase):
    def test_base_pipeline_runs_on_toy_dataset(self) -> None:
        train_rows = [
            make_row("1", "copy buffer into fixed array", ["overflow"]),
            make_row("2", "copy buffer and crash service", ["overflow", "dos"]),
            make_row("3", "bypass auth check", ["bypass"]),
            make_row("4", "read secret data and leak info", ["info"]),
            make_row("5", "execute shell command from input", ["exec"]),
            make_row("6", "increase privilege after bypass", ["priv", "bypass"]),
            make_row("7", "memory write corrupt pointer", ["mem"]),
            make_row("8", "directory traversal file path", ["other"]),
        ]
        val_rows = [
            make_row("9", "copy array buffer", ["overflow"]),
            make_row("10", "bypass login check", ["bypass"]),
        ]
        test_rows = [
            make_row("11", "privilege escalation via bypass", ["priv", "bypass"]),
            make_row("12", "memory corruption in pointer copy", ["mem", "overflow"]),
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            pd.DataFrame(train_rows).to_csv(tmp_path / "train.csv.gz", index=False, compression="gzip")
            pd.DataFrame(val_rows).to_csv(tmp_path / "val.csv.gz", index=False, compression="gzip")
            pd.DataFrame(test_rows).to_csv(tmp_path / "test.csv.gz", index=False, compression="gzip")

            summary = run_base_experiment(
                train_path=tmp_path / "train.csv.gz",
                val_path=tmp_path / "val.csv.gz",
                test_path=tmp_path / "test.csv.gz",
                output_dir=tmp_path / "out",
            )

            self.assertIn("test_metrics", summary)
            self.assertTrue((tmp_path / "out" / "metrics.json").exists())
            self.assertTrue((tmp_path / "out" / "paper_comparison.csv").exists())


if __name__ == "__main__":
    unittest.main()
