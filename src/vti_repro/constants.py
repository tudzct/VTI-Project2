"""Shared constants for the VTI reproduction project."""

LABEL_COLUMNS = [
    "info",
    "priv",
    "bypass",
    "dos",
    "exec",
    "mem",
    "overflow",
    "other",
]

PAPER_BASE_TABLE1 = {
    "micro_precision": 0.71,
    "micro_recall": 0.69,
    "micro_f1": 0.70,
    "macro_precision": 0.65,
    "macro_recall": 0.60,
    "macro_f1": 0.62,
    "weighted_precision": 0.71,
    "weighted_recall": 0.69,
    "weighted_f1": 0.70,
    "samples_precision": 0.65,
    "samples_recall": 0.68,
    "samples_f1": 0.65,
    "exact_match_ratio": 0.54,
    "hamming_score": 0.62,
    "accuracy": 0.89,
}

PAPER_BASE_TABLE2 = {
    "info": 0.41,
    "priv": 0.63,
    "bypass": 0.42,
    "dos": 0.81,
    "exec": 0.64,
    "mem": 0.70,
    "overflow": 0.67,
    "other": 0.69,
}
