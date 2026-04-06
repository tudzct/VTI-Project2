from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vti_repro.labels import normalize_labels
from vti_repro.metrics import accuracy, compute_metrics, exact_match_ratio, hamming_score


class LabelNormalizationTests(unittest.TestCase):
    def test_maps_known_labels(self) -> None:
        record = normalize_labels("DoS Overflow +Priv Mem. Corr. +Info")
        self.assertEqual(record["dos"], 1)
        self.assertEqual(record["overflow"], 1)
        self.assertEqual(record["priv"], 1)
        self.assertEqual(record["mem"], 1)
        self.assertEqual(record["info"], 1)
        self.assertEqual(record["other"], 0)

    def test_maps_unknown_types_to_other(self) -> None:
        record = normalize_labels("Dir. Trav.")
        self.assertEqual(record["other"], 1)
        self.assertEqual(sum(record.values()), 1)

    def test_empty_label_string_returns_all_zero(self) -> None:
        record = normalize_labels("NaN")
        self.assertEqual(sum(record.values()), 0)


class MetricTests(unittest.TestCase):
    def test_exact_match_ratio(self) -> None:
        y_true = np.array([[1, 0], [1, 1]])
        y_pred = np.array([[1, 0], [1, 0]])
        self.assertAlmostEqual(exact_match_ratio(y_true, y_pred), 0.5)

    def test_accuracy(self) -> None:
        y_true = np.array([[1, 0], [1, 1]])
        y_pred = np.array([[1, 0], [1, 0]])
        self.assertAlmostEqual(accuracy(y_true, y_pred), 0.75)

    def test_hamming_score(self) -> None:
        y_true = np.array([[1, 0], [1, 1]])
        y_pred = np.array([[1, 0], [1, 0]])
        self.assertAlmostEqual(hamming_score(y_true, y_pred), 0.75)

    def test_compute_metrics_contains_expected_keys(self) -> None:
        y_true = np.array([[1, 0, 1], [1, 1, 0]])
        y_pred = np.array([[1, 0, 1], [1, 0, 0]])
        metrics = compute_metrics(y_true, y_pred)
        for key in (
            "exact_match_ratio",
            "hamming_score",
            "accuracy",
            "micro_f1",
            "macro_f1",
            "weighted_f1",
            "samples_f1",
        ):
            self.assertIn(key, metrics)


if __name__ == "__main__":
    unittest.main()
