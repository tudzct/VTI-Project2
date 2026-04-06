"""Label normalization helpers."""

from __future__ import annotations

import re
from typing import Dict, Iterable

from .constants import LABEL_COLUMNS

_ALIAS_TO_LABEL = (
    ("exec", ("Exec Code", "Code Exec", "Execution Code")),
    ("overflow", ("Overflow",)),
    ("mem", ("Mem. Corr.", "Mem Corr", "Memory Corruption")),
    ("dos", ("DoS", "Denial of Service")),
    ("bypass", ("Bypass",)),
    ("priv", ("+Priv", "Privilege Gain", "Priv Gain")),
    ("info", ("+Info", "Information Gain", "Info Gain")),
)

_NO_LABEL_VALUES = {"", "nan", "none", "null", "n/a"}


def empty_label_record() -> Dict[str, int]:
    return {label: 0 for label in LABEL_COLUMNS}


def normalize_labels(raw_classification: str | None) -> Dict[str, int]:
    """
    Map raw BigVul vulnerability strings into the 8 labels used in the paper.

    Known vulnerability types are mapped explicitly. Any non-empty leftover token
    after removing known aliases is assigned to `other`.
    """

    labels = empty_label_record()
    if raw_classification is None:
        return labels

    cleaned = re.sub(r"\s+", " ", str(raw_classification)).strip()
    if cleaned.lower() in _NO_LABEL_VALUES:
        return labels

    remainder = cleaned
    for label, aliases in _ALIAS_TO_LABEL:
        for alias in aliases:
            if re.search(re.escape(alias), remainder, flags=re.IGNORECASE):
                labels[label] = 1
                remainder = re.sub(
                    re.escape(alias),
                    " ",
                    remainder,
                    flags=re.IGNORECASE,
                )

    remainder = re.sub(r"[+/.,;:_-]", " ", remainder)
    leftover_tokens = [token for token in remainder.split() if token.lower() not in _NO_LABEL_VALUES]
    if leftover_tokens:
        labels["other"] = 1

    return labels


def has_any_label(label_record: Dict[str, int]) -> bool:
    return any(label_record.values())


def labels_to_list(label_record: Dict[str, int]) -> Iterable[str]:
    return [label for label in LABEL_COLUMNS if label_record.get(label, 0) == 1]
