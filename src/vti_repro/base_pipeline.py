"""Classical baseline reproduction: TF-IDF + Binary Relevance + GaussianNB."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import GaussianNB

from .constants import LABEL_COLUMNS, PAPER_BASE_TABLE1, PAPER_BASE_TABLE2
from .io_utils import dump_json, dump_progress, ensure_directory
from .metrics import compute_metrics
from .preprocessing import preprocess_code


@dataclass
class BaseConfig:
    max_features: int = 20_000
    ngram_range: tuple[int, int] = (1, 2)
    p_value_grid: tuple[float, ...] = (0.5, 0.25, 0.1, 0.05, 0.01, 0.005, 0.001)


class MinPValueSelector:
    """Keep features that are relevant to at least one label."""

    def __init__(self, threshold: float):
        self.threshold = threshold
        self.mask_: np.ndarray | None = None
        self.min_p_values_: np.ndarray | None = None

    def fit(self, x_train: csr_matrix, y_train: np.ndarray) -> "MinPValueSelector":
        p_values = []
        for label_index in range(y_train.shape[1]):
            _, p_val = chi2(x_train, y_train[:, label_index])
            p_values.append(np.nan_to_num(p_val, nan=1.0))
        self.min_p_values_ = np.min(np.vstack(p_values), axis=0)
        self.mask_ = self.min_p_values_ <= self.threshold
        if not np.any(self.mask_):
            smallest_index = int(np.argmin(self.min_p_values_))
            self.mask_[smallest_index] = True
        return self

    def transform(self, matrix: csr_matrix) -> csr_matrix:
        if self.mask_ is None:
            raise RuntimeError("Selector must be fitted before transform.")
        return matrix[:, self.mask_]

    def fit_transform(self, x_train: csr_matrix, y_train: np.ndarray) -> csr_matrix:
        return self.fit(x_train, y_train).transform(x_train)

    @property
    def selected_feature_count(self) -> int:
        if self.mask_ is None:
            return 0
        return int(np.sum(self.mask_))


def _load_split(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, compression="infer")
    for label in LABEL_COLUMNS:
        df[label] = df[label].astype(int)
    return df


def _prepare_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: BaseConfig,
    progress_path: Path | None = None,
):
    if progress_path is not None:
        dump_progress(progress_path, "preprocessing_train_text")
    train_text = train_df["text"].map(preprocess_code)
    if progress_path is not None:
        dump_progress(progress_path, "preprocessing_val_text")
    val_text = val_df["text"].map(preprocess_code)
    if progress_path is not None:
        dump_progress(progress_path, "preprocessing_test_text")
    test_text = test_df["text"].map(preprocess_code)

    vectorizer = TfidfVectorizer(
        tokenizer=str.split,
        preprocessor=None,
        token_pattern=None,
        lowercase=False,
        ngram_range=config.ngram_range,
        max_features=config.max_features,
    )
    if progress_path is not None:
        dump_progress(progress_path, "vectorizer_fit_transform_start")
    x_train = vectorizer.fit_transform(train_text)
    if progress_path is not None:
        dump_progress(progress_path, "vectorizer_transform_val_start", vocabulary_size=int(len(vectorizer.vocabulary_)))
    x_val = vectorizer.transform(val_text)
    if progress_path is not None:
        dump_progress(progress_path, "vectorizer_transform_test_start", vocabulary_size=int(len(vectorizer.vocabulary_)))
    x_test = vectorizer.transform(test_text)
    return vectorizer, x_train, x_val, x_test


def _train_with_threshold(
    x_train: csr_matrix,
    y_train: np.ndarray,
    x_eval: csr_matrix,
    threshold: float,
):
    selector = MinPValueSelector(threshold=threshold)
    x_train_selected = selector.fit_transform(x_train, y_train)
    x_eval_selected = selector.transform(x_eval)
    classifier = MultiOutputClassifier(GaussianNB())
    classifier.fit(x_train_selected.toarray(), y_train)
    y_pred = classifier.predict(x_eval_selected.toarray())
    return selector, classifier, y_pred


def _build_comparison(metrics: dict[str, float]) -> pd.DataFrame:
    rows = []
    for metric_name, paper_value in PAPER_BASE_TABLE1.items():
        rows.append(
            {
                "metric": metric_name,
                "paper_base": paper_value,
                "reproduced_base": metrics.get(metric_name),
                "delta": None if metrics.get(metric_name) is None else metrics[metric_name] - paper_value,
            }
        )
    for label, paper_value in PAPER_BASE_TABLE2.items():
        metric_name = f"f1_{label}"
        rows.append(
            {
                "metric": metric_name,
                "paper_base": paper_value,
                "reproduced_base": metrics.get(metric_name),
                "delta": None if metrics.get(metric_name) is None else metrics[metric_name] - paper_value,
            }
        )
    return pd.DataFrame(rows)


def run_base_experiment(
    train_path: str | Path,
    val_path: str | Path,
    test_path: str | Path,
    output_dir: str | Path,
    config: BaseConfig | None = None,
) -> dict:
    config = config or BaseConfig()
    output_dir = ensure_directory(output_dir)
    progress_path = output_dir / "progress.json"
    dump_progress(progress_path, "starting", config={"max_features": config.max_features, "p_value_grid": list(config.p_value_grid)})

    train_df = _load_split(train_path)
    val_df = _load_split(val_path)
    test_df = _load_split(test_path)
    dump_progress(
        progress_path,
        "splits_loaded",
        train_size=int(len(train_df)),
        val_size=int(len(val_df)),
        test_size=int(len(test_df)),
    )

    y_train = train_df[LABEL_COLUMNS].to_numpy(dtype=int)
    y_val = val_df[LABEL_COLUMNS].to_numpy(dtype=int)
    y_test = test_df[LABEL_COLUMNS].to_numpy(dtype=int)

    vectorizer, x_train, x_val, x_test = _prepare_features(train_df, val_df, test_df, config, progress_path=progress_path)
    dump_progress(
        progress_path,
        "features_prepared",
        vocabulary_size=int(len(vectorizer.vocabulary_)),
        x_train_shape=list(x_train.shape),
        x_val_shape=list(x_val.shape),
        x_test_shape=list(x_test.shape),
    )

    search_results = []
    best_threshold = None
    best_val_metrics = None
    best_selector = None
    best_classifier = None

    search_start = time.perf_counter()
    for threshold in config.p_value_grid:
        selector, classifier, val_pred = _train_with_threshold(x_train, y_train, x_val, threshold)
        val_metrics = compute_metrics(y_val, val_pred)
        search_results.append(
            {
                "p_value_threshold": threshold,
                "selected_feature_count": selector.selected_feature_count,
                **val_metrics,
            }
        )
        if best_val_metrics is None or val_metrics["macro_f1"] > best_val_metrics["macro_f1"]:
            best_threshold = threshold
            best_val_metrics = val_metrics
            best_selector = selector
            best_classifier = classifier
        dump_progress(
            progress_path,
            "threshold_search",
            current_threshold=threshold,
            current_macro_f1=val_metrics["macro_f1"],
            best_threshold=best_threshold,
            best_macro_f1=None if best_val_metrics is None else best_val_metrics["macro_f1"],
            selected_feature_count=selector.selected_feature_count,
        )
    tuning_seconds = time.perf_counter() - search_start

    assert best_selector is not None
    assert best_classifier is not None
    assert best_threshold is not None
    assert best_val_metrics is not None

    train_start = time.perf_counter()
    x_train_selected = best_selector.transform(x_train)
    best_classifier.fit(x_train_selected.toarray(), y_train)
    training_seconds = time.perf_counter() - train_start
    dump_progress(
        progress_path,
        "final_model_trained",
        best_threshold=best_threshold,
        selected_feature_count=best_selector.selected_feature_count,
        training_seconds=training_seconds,
    )

    predict_start = time.perf_counter()
    x_test_selected = best_selector.transform(x_test)
    y_test_pred = best_classifier.predict(x_test_selected.toarray())
    predict_seconds = time.perf_counter() - predict_start
    dump_progress(
        progress_path,
        "prediction_complete",
        predict_seconds=predict_seconds,
    )

    test_metrics = compute_metrics(y_test, y_test_pred)
    report = classification_report(
        y_test,
        y_test_pred,
        target_names=LABEL_COLUMNS,
        zero_division=0,
        output_dict=True,
    )

    test_predictions = test_df[["sample_id", "raw_classification"]].copy()
    for label_index, label in enumerate(LABEL_COLUMNS):
        test_predictions[f"pred_{label}"] = y_test_pred[:, label_index]
        test_predictions[f"true_{label}"] = y_test[:, label_index]

    test_predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    pd.DataFrame(search_results).to_csv(output_dir / "validation_search.csv", index=False)
    _build_comparison(test_metrics).to_csv(output_dir / "paper_comparison.csv", index=False)

    summary = {
        "best_p_value_threshold": best_threshold,
        "selected_feature_count": best_selector.selected_feature_count,
        "vectorizer_vocabulary_size": len(vectorizer.vocabulary_),
        "tuning_seconds": tuning_seconds,
        "training_seconds": training_seconds,
        "predict_seconds": predict_seconds,
        "validation_metrics": best_val_metrics,
        "test_metrics": test_metrics,
        "classification_report": report,
    }
    dump_json(summary, output_dir / "metrics.json")
    dump_progress(
        progress_path,
        "completed",
        best_threshold=best_threshold,
        macro_f1=test_metrics["macro_f1"],
        micro_f1=test_metrics["micro_f1"],
    )
    Path(output_dir / "classification_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary
