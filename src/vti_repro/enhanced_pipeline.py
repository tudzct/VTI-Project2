"""Enhanced post-processing pipeline using code elements compatible with the paper."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .constants import LABEL_COLUMNS
from .io_utils import dump_json, ensure_directory
from .metrics import compute_metrics
from .preprocessing import split_identifier


@dataclass
class EnhancedConfig:
    prediction_threshold: float = 0.5
    freq_threshold: float = 1.5
    require_positive_ids_and_calls: bool = True
    require_negative_all_three: bool = True


KEYWORD_CALL_EXCLUDE = {"if", "for", "while", "switch", "return", "sizeof"}


def _tokenize_identifier_chunks(text: str) -> set[str]:
    tokens: set[str] = set()
    for raw in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text or ""):
        tokens.update(split_identifier(raw))
    return {token for token in tokens if token}


def _extract_calls(text: str) -> set[str]:
    calls = set()
    for match in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", text or ""):
        name = match.group(1)
        if name not in KEYWORD_CALL_EXCLUDE:
            calls.update(split_identifier(name))
    return {token for token in calls if token}


def _extract_controls(text: str) -> set[str]:
    controls = set()
    for keyword in ("if", "for", "while", "switch", "case", "return"):
        if re.search(rf"\b{keyword}\b", text or ""):
            controls.add(keyword)
    return controls


def build_equivalent_elements(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    out["ids"] = out["text"].map(_tokenize_identifier_chunks)
    out["calls"] = out["text"].map(_extract_calls)
    out["controls"] = out["text"].map(_extract_controls)
    return out


def _compute_prevalence(df: pd.DataFrame, label: str, field: str, universe: set[str]) -> dict[str, float]:
    items = df.loc[df[label] == 1]
    total = max(len(items), 1)
    freq = {}
    for token in universe:
        count = sum(1 for tokens in items[field] if token in tokens)
        freq[token] = 100.0 * count / total
    return freq


def _positive_tokens(freqs: dict[str, dict[str, float]], target: str, threshold: float) -> set[str]:
    result = set()
    target_freq = freqs[target]
    for token, rel_target in target_freq.items():
        rel_other = max((freqs[label][token] for label in freqs if label != target), default=0.0)
        if rel_other == 0 and rel_target > 0:
            result.add(token)
            continue
        if rel_other > 0 and rel_target / rel_other > threshold:
            result.add(token)
    return result


def _negative_tokens(freqs: dict[str, dict[str, float]], target: str, threshold: float) -> set[str]:
    result = set()
    target_freq = freqs[target]
    for token, rel_target in target_freq.items():
        rel_other = min((freqs[label][token] for label in freqs if label != target), default=0.0)
        if rel_target == 0 and rel_other > 0:
            result.add(token)
            continue
        if rel_target > 0 and rel_other / rel_target > threshold:
            result.add(token)
    return result


def build_feature_table(train_df: pd.DataFrame, freq_threshold: float) -> dict[str, dict[str, set[str]]]:
    feature_table: dict[str, dict[str, set[str]]] = {}
    for field in ("ids", "calls", "controls"):
        universe = set().union(*train_df[field].tolist()) if len(train_df) else set()
        freqs = {label: _compute_prevalence(train_df, label, field, universe) for label in LABEL_COLUMNS}
        for label in LABEL_COLUMNS:
            feature_table.setdefault(label, {})
            feature_table[label][f"pos_{field}"] = _positive_tokens(freqs, label, freq_threshold)
            feature_table[label][f"neg_{field}"] = _negative_tokens(freqs, label, freq_threshold)
    return feature_table


def refine_predictions(
    score_df: pd.DataFrame,
    label_df: pd.DataFrame,
    element_df: pd.DataFrame,
    feature_table: dict[str, dict[str, set[str]]],
    config: EnhancedConfig,
) -> tuple[pd.DataFrame, dict[str, float]]:
    preds = (score_df[LABEL_COLUMNS] >= config.prediction_threshold).astype(int).copy()
    affected = 0
    corrected = 0
    for index in preds.index:
        for label in LABEL_COLUMNS:
            row_ids = element_df.at[index, "ids"]
            row_calls = element_df.at[index, "calls"]
            row_controls = element_df.at[index, "controls"]
            pos_ids = feature_table[label]["pos_ids"]
            pos_calls = feature_table[label]["pos_calls"]
            pos_controls = feature_table[label]["pos_controls"]
            neg_ids = feature_table[label]["neg_ids"]
            neg_calls = feature_table[label]["neg_calls"]
            neg_controls = feature_table[label]["neg_controls"]

            pos_hit = (
                bool(row_ids.intersection(pos_ids)) and bool(row_calls.intersection(pos_calls))
                if config.require_positive_ids_and_calls
                else bool(row_ids.intersection(pos_ids) or row_calls.intersection(pos_calls) or row_controls.intersection(pos_controls))
            )
            neg_hit = (
                bool(row_ids.intersection(neg_ids))
                and bool(row_calls.intersection(neg_calls))
                and bool(row_controls.intersection(neg_controls))
                if config.require_negative_all_three
                else bool(row_ids.intersection(neg_ids) or row_calls.intersection(neg_calls) or row_controls.intersection(neg_controls))
            )

            if preds.at[index, label] == 0 and pos_hit:
                preds.at[index, label] = 1
                affected += 1
                if int(label_df.at[index, label]) == 1:
                    corrected += 1
            elif preds.at[index, label] == 1 and neg_hit:
                preds.at[index, label] = 0
                affected += 1
                if int(label_df.at[index, label]) == 0:
                    corrected += 1

    correction_rate = corrected / affected if affected else 0.0
    return preds, {"affected_predictions": float(affected), "correction_rate": correction_rate}


def run_enhanced_experiment(
    train_path: str | Path,
    test_path: str | Path,
    raw_preds_path: str | Path,
    labels_path: str | Path,
    output_dir: str | Path,
    config: EnhancedConfig | None = None,
) -> dict:
    config = config or EnhancedConfig()
    output_dir = ensure_directory(output_dir)
    train_df = build_equivalent_elements(pd.read_csv(train_path, compression="infer"))
    test_df = build_equivalent_elements(pd.read_csv(test_path, compression="infer"))
    score_df = pd.read_csv(raw_preds_path)
    label_df = pd.read_csv(labels_path)

    feature_table = build_feature_table(train_df, config.freq_threshold)
    before_preds = (score_df[LABEL_COLUMNS] >= config.prediction_threshold).astype(int)
    before_metrics = compute_metrics(label_df[LABEL_COLUMNS].to_numpy(dtype=int), before_preds.to_numpy(dtype=int))
    after_preds, stats = refine_predictions(score_df, label_df, test_df, feature_table, config)
    after_metrics = compute_metrics(label_df[LABEL_COLUMNS].to_numpy(dtype=int), after_preds.to_numpy(dtype=int))

    pd.DataFrame(after_preds, columns=LABEL_COLUMNS).to_csv(output_dir / "enhanced_preds.csv", index=False)
    serializable_features = {
        label: {name: sorted(list(tokens)) for name, tokens in token_map.items()}
        for label, token_map in feature_table.items()
    }
    dump_json(serializable_features, output_dir / "feature_table.json")
    summary = {
        "config": {
            "prediction_threshold": config.prediction_threshold,
            "freq_threshold": config.freq_threshold,
        },
        "before_metrics": before_metrics,
        "after_metrics": after_metrics,
        **stats,
    }
    dump_json(summary, output_dir / "metrics.json")
    return summary
