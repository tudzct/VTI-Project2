"""Notebook-compatible Enhanced pipeline."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report

from .constants import LABEL_COLUMNS
from .io_utils import dump_json, ensure_directory
from .metrics import compute_metrics
from .preprocessing import split_identifier


NOTEBOOK_LABELS = ["dos", "info", "overflow", "priv", "mem", "exec", "bypass"]


@dataclass
class NotebookEnhancedConfig:
    prediction_threshold: float = 0.3
    freq_threshold: float = 1.5


def _identifier_nodes(text: str) -> list[dict]:
    nodes = []
    for raw in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text or ""):
        nodes.append({"_label": "IDENTIFIER", "name": raw})
    return nodes


def _call_nodes(text: str) -> list[dict]:
    nodes = []
    for match in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", text or ""):
        nodes.append({"_label": "CALL", "name": match.group(1)})
    return nodes


def _control_nodes(text: str) -> list[dict]:
    nodes = []
    for keyword in ("if", "for", "while", "switch", "case", "return"):
        if re.search(rf"\b{keyword}\b", text or ""):
            nodes.append({"_label": "CONTROL_STRUCTURE", "code": keyword})
    return nodes


def build_pseudo_cpg(text: str) -> str:
    elements = _identifier_nodes(text) + _call_nodes(text) + _control_nodes(text)
    return json.dumps(elements) + "--====--"


def get_code_elements(cpg: str) -> dict[str, list[dict]]:
    elements_map = {"IDENTIFIER": [], "CALL": [], "CONTROL_STRUCTURE": []}
    if len(str(cpg)) > 100:
        lines = str(cpg).split("--====--")
        elements = json.loads(lines[0])
        for element in elements:
            label = str(element.get("_label", ""))
            if label in elements_map:
                elements_map[label].append(element)
    return elements_map


def _code_tokenizer(text: str) -> set[str]:
    result = set()
    for token in split_identifier(text):
        if token:
            result.add(token)
    return result


def extract_identifiers(nodes: list[dict]) -> set[str]:
    result = set()
    for node in nodes:
        result.update(_code_tokenizer(str(node.get("name", ""))))
    return result


def extract_calls(nodes: list[dict]) -> set[str]:
    result = set()
    for node in nodes:
        result.update(_code_tokenizer(str(node.get("name", ""))))
    return result


def extract_controls(nodes: list[dict]) -> set[str]:
    result = set()
    for node in nodes:
        result.update(_code_tokenizer(str(node.get("code", ""))))
    return result


def attach_notebook_elements(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    if "cpg" not in out.columns:
        out["cpg"] = out["text"].map(build_pseudo_cpg)
    out["elements"] = out["cpg"].map(get_code_elements)
    out["ids"] = out["elements"].map(lambda m: extract_identifiers(m["IDENTIFIER"]))
    out["calls"] = out["elements"].map(lambda m: extract_calls(m["CALL"]))
    out["controls"] = out["elements"].map(lambda m: extract_controls(m["CONTROL_STRUCTURE"]))
    return out


def _compute_freq(df: pd.DataFrame, label: str, universe: set[str], field: str) -> dict[str, float]:
    items = df.loc[df[label] == 1]
    freq = {}
    for token in universe:
        freq[token] = 0
        for tokens in items[field]:
            if token in tokens:
                freq[token] += 1
    denom = max(len(freq), 1)
    for token in universe:
        freq[token] = 100 * freq[token] / denom
    return freq


def _freq_stat(freqs: list[dict[str, float]], target: int, universe: set[str], threshold: float) -> list[tuple[str, float]]:
    target_vs_others = {}
    target_freq = freqs[target]
    for token in universe:
        rel_target = target_freq[token]
        rel_other = 0
        for index, freq in enumerate(freqs):
            if index != target and freq[token] > rel_other:
                rel_other = freq[token]
        ratio = 0
        if rel_other != 0:
            ratio = rel_target / rel_other
        elif rel_target != 0:
            ratio = 1000
        if ratio > threshold:
            target_vs_others[token] = ratio
    return sorted(target_vs_others.items(), key=lambda item: item[1], reverse=True)


def _infreq_stat(freqs: list[dict[str, float]], target: int, universe: set[str], threshold: float) -> list[tuple[str, float]]:
    target_vs_others = {}
    target_freq = freqs[target]
    for token in universe:
        rel_target = target_freq[token]
        rel_other = 10000
        for index, freq in enumerate(freqs):
            if index != target and freq[token] < rel_other:
                rel_other = freq[token]
        ratio = 0
        if rel_other != 0 and rel_target != 0:
            ratio = rel_other / rel_target
        elif rel_other != 0:
            ratio = 1000
        if ratio > threshold:
            target_vs_others[token] = ratio
    return sorted(target_vs_others.items(), key=lambda item: item[1], reverse=True)


def build_feature_table(train_df: pd.DataFrame, threshold: float) -> dict[str, list[set[str]]]:
    all_ids = set().union(*train_df["ids"].tolist()) if len(train_df) else set()
    all_calls = set().union(*train_df["calls"].tolist()) if len(train_df) else set()
    all_controls = set().union(*train_df["controls"].tolist()) if len(train_df) else set()

    freqs_ids = [_compute_freq(train_df, label, all_ids, "ids") for label in NOTEBOOK_LABELS]
    freqs_calls = [_compute_freq(train_df, label, all_calls, "calls") for label in NOTEBOOK_LABELS]
    freqs_controls = [_compute_freq(train_df, label, all_controls, "controls") for label in NOTEBOOK_LABELS]

    feature_table: dict[str, list[set[str]]] = {}
    for index, label in enumerate(NOTEBOOK_LABELS):
        pos_ids = {token for token, _ in _freq_stat(freqs_ids, index, all_ids, threshold)}
        pos_calls = {token for token, _ in _freq_stat(freqs_calls, index, all_calls, threshold)}
        pos_controls = {token for token, _ in _freq_stat(freqs_controls, index, all_controls, threshold)}
        neg_ids = {token for token, _ in _infreq_stat(freqs_ids, index, all_ids, threshold)}
        neg_calls = {token for token, _ in _infreq_stat(freqs_calls, index, all_calls, threshold)}
        neg_controls = {token for token, _ in _infreq_stat(freqs_controls, index, all_controls, threshold)}
        feature_table[label] = [pos_ids, pos_calls, pos_controls, neg_ids, neg_calls, neg_controls]
    return feature_table


def apply_notebook_rules(
    score_df: pd.DataFrame,
    label_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_table: dict[str, list[set[str]]],
    threshold: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    preds = score_df.copy()
    for label in LABEL_COLUMNS:
        preds[label] = (preds[label] >= threshold).astype(int)

    counter = 0
    correct = 0
    for index in preds.index:
        for label in NOTEBOOK_LABELS:
            if label not in feature_table:
                continue
            pos_ids, pos_calls, pos_controls, neg_ids, neg_calls, neg_controls = feature_table[label]
            intersec_call = pos_calls.intersection(test_df.at[index, "calls"])
            intersec_id = pos_ids.intersection(test_df.at[index, "ids"])
            intersec_con = pos_controls.intersection(test_df.at[index, "controls"])
            if preds.at[index, label] == 0 and (len(intersec_id) > 0 and len(intersec_call) > 0):
                preds.at[index, label] = 1
                counter += 1
                if int(label_df.at[index, label]) == 1:
                    correct += 1

            infreq_intersec_call = neg_calls.intersection(test_df.at[index, "calls"])
            infreq_intersec_id = neg_ids.intersection(test_df.at[index, "ids"])
            infreq_intersec_con = neg_controls.intersection(test_df.at[index, "controls"])
            if preds.at[index, label] == 1 and (
                len(infreq_intersec_call) > 0 and len(infreq_intersec_id) > 0 and len(infreq_intersec_con) > 0
            ):
                preds.at[index, label] = 0
                counter += 1
                if int(label_df.at[index, label]) == 0:
                    correct += 1

    correction_rate = correct / counter if counter else 0.0
    return preds, {"affected_predictions": float(counter), "correction_rate": correction_rate}


def run_notebook_compatible_enhanced(
    train_path: str | Path,
    test_path: str | Path,
    raw_preds_path: str | Path,
    labels_path: str | Path,
    output_dir: str | Path,
    config: NotebookEnhancedConfig | None = None,
) -> dict:
    config = config or NotebookEnhancedConfig()
    output_dir = ensure_directory(output_dir)
    train_df = attach_notebook_elements(pd.read_csv(train_path, compression="infer"))
    test_df = attach_notebook_elements(pd.read_csv(test_path, compression="infer"))
    score_df = pd.read_csv(raw_preds_path)
    label_df = pd.read_csv(labels_path)

    feature_table = build_feature_table(train_df, config.freq_threshold)
    before_preds = score_df.copy()
    for label in LABEL_COLUMNS:
        before_preds[label] = (before_preds[label] >= config.prediction_threshold).astype(int)
    before_metrics = compute_metrics(label_df[LABEL_COLUMNS].to_numpy(dtype=int), before_preds[LABEL_COLUMNS].to_numpy(dtype=int))
    after_preds, stats = apply_notebook_rules(score_df[LABEL_COLUMNS], label_df[LABEL_COLUMNS], test_df, feature_table, config.prediction_threshold)
    after_metrics = compute_metrics(label_df[LABEL_COLUMNS].to_numpy(dtype=int), after_preds[LABEL_COLUMNS].to_numpy(dtype=int))

    after_preds.to_csv(output_dir / "enhanced_preds.csv", index=False)
    dump_json(
        {
            label: [[sorted(list(values)) for values in feature_table[label]]]
            for label in feature_table
        },
        output_dir / "feature_table.json",
    )
    report = classification_report(
        label_df[LABEL_COLUMNS].to_numpy(dtype=int),
        after_preds[LABEL_COLUMNS].to_numpy(dtype=int),
        target_names=LABEL_COLUMNS,
        zero_division=0,
        output_dict=True,
    )
    dump_json(report, output_dir / "classification_report.json")
    summary = {
        "config": {
            "prediction_threshold": config.prediction_threshold,
            "freq_threshold": config.freq_threshold,
            "compat_mode": "notebook",
        },
        "before_metrics": before_metrics,
        "after_metrics": after_metrics,
        **stats,
    }
    dump_json(summary, output_dir / "metrics.json")
    return summary
