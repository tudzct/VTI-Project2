"""Optional Word2Vec + BiLSTM + Attention experiment scaffolding."""

from __future__ import annotations

from dataclasses import dataclass

try:
    import gensim
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    gensim = None


@dataclass
class Word2VecConfig:
    vector_size: int = 300
    window: int = 25
    min_count: int = 1
    sg: int = 1
    epochs: int = 30
    max_len: int = 512
    hidden_size: int = 128
    batch_size: int = 32


def ensure_word2vec_dependencies() -> None:
    if gensim is None:
        raise ModuleNotFoundError(
            "gensim is required for the Word2Vec experiment. "
            "Install it in the Python environment used to run the scripts."
        )
