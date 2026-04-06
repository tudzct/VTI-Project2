"""Metrics used in the VTI paper."""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from .constants import LABEL_COLUMNS


def exact_match_ratio(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.all(y_true == y_pred, axis=1)))


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def hamming_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    scores = []
    for true_row, pred_row in zip(y_true, y_pred, strict=False):
        true_mask = true_row.astype(bool)
        pred_mask = pred_row.astype(bool)
        union = np.logical_or(true_mask, pred_mask).sum()
        if union == 0:
            scores.append(1.0)
            continue
        intersection = np.logical_and(true_mask, pred_mask).sum()
        scores.append(intersection / union)
    return float(np.mean(scores))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    metrics: Dict[str, float] = {
        "exact_match_ratio": exact_match_ratio(y_true, y_pred),
        "hamming_score": hamming_score(y_true, y_pred),
        "accuracy": accuracy(y_true, y_pred),
    }

    for average_name in ("micro", "macro", "weighted", "samples"):
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average=average_name,
            zero_division=0,
        )
        metrics[f"{average_name}_precision"] = float(precision)
        metrics[f"{average_name}_recall"] = float(recall)
        metrics[f"{average_name}_f1"] = float(f1)

    _, _, f1_by_label, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average=None,
        zero_division=0,
    )
    for label, score in zip(LABEL_COLUMNS, f1_by_label, strict=False):
        metrics[f"f1_{label}"] = float(score)

    return metrics
