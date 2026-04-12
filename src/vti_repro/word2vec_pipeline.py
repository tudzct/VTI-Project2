"""Word2Vec + BiLSTM + Attention experiment runner."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import gensim
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    gensim = None

from .constants import LABEL_COLUMNS
from .io_utils import dump_json, dump_progress, ensure_directory
from .metrics import compute_metrics
from .preprocessing import preprocess_code


@dataclass
class Word2VecConfig:
    vector_size: int = 100
    window: int = 25
    min_count: int = 1
    sg: int = 1
    w2v_epochs: int = 10
    nn_epochs: int = 2
    max_len: int = 256
    hidden_size: int = 64
    batch_size: int = 32
    max_train_rows: int | None = 4096
    max_val_rows: int | None = 512
    max_test_rows: int | None = 512
    threshold_grid: tuple[float, ...] = (0.3, 0.4, 0.5, 0.6)


def ensure_word2vec_dependencies() -> None:
    if gensim is None:
        raise ModuleNotFoundError(
            "gensim is required for the Word2Vec experiment. "
            "Install it in the Python environment used to run the scripts."
        )


def _subset(df: pd.DataFrame, max_rows: int | None, seed: int = 42) -> pd.DataFrame:
    if max_rows is None or len(df) <= max_rows:
        return df.reset_index(drop=True)
    return df.sample(n=max_rows, random_state=seed).sort_values("sample_id").reset_index(drop=True)


def _load_split(path: str | Path, max_rows: int | None) -> pd.DataFrame:
    df = pd.read_csv(path, compression="infer")
    for label in LABEL_COLUMNS:
        df[label] = df[label].astype(int)
    return _subset(df, max_rows=max_rows)


def _tokenize_texts(series: pd.Series) -> list[list[str]]:
    return [preprocess_code(text).split() for text in series.fillna("")]


def _build_embeddings(word_index: dict[str, int], model, vector_size: int) -> np.ndarray:
    embeddings = np.zeros((len(word_index) + 1, vector_size), dtype=np.float32)
    for word, idx in word_index.items():
        if word in model.wv:
            embeddings[idx] = model.wv[word]
    return embeddings


def _search_threshold(y_true: np.ndarray, y_score: np.ndarray, threshold_grid: tuple[float, ...]):
    best_threshold = threshold_grid[0]
    best_metrics = None
    rows = []
    for threshold in threshold_grid:
        y_pred = (y_score >= threshold).astype(int)
        metrics = compute_metrics(y_true, y_pred)
        rows.append({"threshold": threshold, **metrics})
        if best_metrics is None or metrics["macro_f1"] > best_metrics["macro_f1"]:
            best_threshold = threshold
            best_metrics = metrics
    return best_threshold, best_metrics, pd.DataFrame(rows)


def run_word2vec_experiment(
    train_path: str | Path,
    val_path: str | Path,
    test_path: str | Path,
    output_dir: str | Path,
    config: Word2VecConfig | None = None,
) -> dict:
    ensure_word2vec_dependencies()
    import tensorflow as tf
    from tensorflow.keras import Model
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import Bidirectional, Dense, Embedding, Input, LSTM, Softmax
    from tensorflow.keras.layers import Layer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.preprocessing.text import Tokenizer

    config = config or Word2VecConfig()
    output_dir = ensure_directory(output_dir)
    progress_path = output_dir / "progress.json"
    dump_progress(
        progress_path,
        "starting",
        config={
            "vector_size": config.vector_size,
            "w2v_epochs": config.w2v_epochs,
            "nn_epochs": config.nn_epochs,
            "max_len": config.max_len,
            "batch_size": config.batch_size,
            "max_train_rows": config.max_train_rows,
            "max_val_rows": config.max_val_rows,
            "max_test_rows": config.max_test_rows,
        },
    )

    train_df = _load_split(train_path, config.max_train_rows)
    val_df = _load_split(val_path, config.max_val_rows)
    test_df = _load_split(test_path, config.max_test_rows)
    dump_progress(
        progress_path,
        "splits_loaded",
        train_size=int(len(train_df)),
        val_size=int(len(val_df)),
        test_size=int(len(test_df)),
    )

    train_tokens = _tokenize_texts(train_df["text"])
    val_tokens = _tokenize_texts(val_df["text"])
    test_tokens = _tokenize_texts(test_df["text"])
    dump_progress(
        progress_path,
        "tokenized",
        train_documents=int(len(train_tokens)),
        val_documents=int(len(val_tokens)),
        test_documents=int(len(test_tokens)),
    )

    w2v_model = gensim.models.Word2Vec(
        sentences=train_tokens,
        vector_size=config.vector_size,
        window=config.window,
        min_count=config.min_count,
        sg=config.sg,
        epochs=config.w2v_epochs,
        workers=1,
    )
    dump_progress(
        progress_path,
        "word2vec_trained",
        vocabulary_size=int(len(w2v_model.wv)),
    )

    tokenizer = Tokenizer(lower=False, filters="")
    tokenizer.fit_on_texts(train_tokens)
    x_train = pad_sequences(tokenizer.texts_to_sequences(train_tokens), maxlen=config.max_len, padding="post", truncating="post")
    x_val = pad_sequences(tokenizer.texts_to_sequences(val_tokens), maxlen=config.max_len, padding="post", truncating="post")
    x_test = pad_sequences(tokenizer.texts_to_sequences(test_tokens), maxlen=config.max_len, padding="post", truncating="post")

    y_train = train_df[LABEL_COLUMNS].to_numpy(dtype=np.float32)
    y_val = val_df[LABEL_COLUMNS].to_numpy(dtype=int)
    y_test = test_df[LABEL_COLUMNS].to_numpy(dtype=int)

    embeddings = _build_embeddings(tokenizer.word_index, w2v_model, config.vector_size)
    dump_progress(
        progress_path,
        "sequences_prepared",
        tokenizer_vocab_size=int(len(tokenizer.word_index)),
        embedding_shape=list(embeddings.shape),
    )

    class TemporalAttention(Layer):
        def __init__(self):
            super().__init__()
            self.score_dense = Dense(1, activation="tanh")
            self.softmax = Softmax(axis=1)

        def call(self, inputs):
            scores = self.score_dense(inputs)
            weights = self.softmax(scores)
            return tf.reduce_sum(weights * inputs, axis=1)

    inputs = Input(shape=(config.max_len,))
    x = Embedding(
        input_dim=embeddings.shape[0],
        output_dim=embeddings.shape[1],
        weights=[embeddings],
        trainable=False,
    )(inputs)
    x = Bidirectional(LSTM(config.hidden_size, return_sequences=True, dropout=0.2))(x)
    x = TemporalAttention()(x)
    x = Dense(64, activation="relu")(x)
    outputs = Dense(len(LABEL_COLUMNS), activation="sigmoid")(x)
    model = Model(inputs, outputs)
    model.compile(optimizer="adam", loss="binary_crossentropy")

    early_stopping = EarlyStopping(monitor="val_loss", patience=1, restore_best_weights=True)
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val.astype(np.float32)),
        epochs=config.nn_epochs,
        batch_size=config.batch_size,
        verbose=2,
        callbacks=[early_stopping],
    )
    dump_progress(
        progress_path,
        "neural_model_trained",
        epochs_ran=int(len(history.history["loss"])),
        final_loss=float(history.history["loss"][-1]),
        final_val_loss=float(history.history["val_loss"][-1]),
    )

    val_scores = model.predict(x_val, verbose=0)
    best_threshold, best_val_metrics, threshold_df = _search_threshold(y_val, val_scores, config.threshold_grid)
    dump_progress(
        progress_path,
        "threshold_selected",
        best_threshold=float(best_threshold),
        best_macro_f1=float(best_val_metrics["macro_f1"]),
    )

    test_scores = model.predict(x_test, verbose=0)
    test_preds = (test_scores >= best_threshold).astype(int)
    test_metrics = compute_metrics(y_test, test_preds)

    pd.DataFrame({"loss": history.history["loss"], "val_loss": history.history["val_loss"]}).to_csv(
        output_dir / "training_history.csv",
        index=False,
    )
    threshold_df.to_csv(output_dir / "validation_threshold_search.csv", index=False)
    pd.DataFrame(test_scores, columns=LABEL_COLUMNS).to_csv(output_dir / "raw_preds.csv", index=False)
    pd.DataFrame(y_test, columns=LABEL_COLUMNS).to_csv(output_dir / "labels.csv", index=False)
    prediction_frame = test_df[["sample_id", "raw_classification"]].copy()
    for label_idx, label in enumerate(LABEL_COLUMNS):
        prediction_frame[f"score_{label}"] = test_scores[:, label_idx]
        prediction_frame[f"pred_{label}"] = test_preds[:, label_idx]
        prediction_frame[f"true_{label}"] = y_test[:, label_idx]
    prediction_frame.to_csv(output_dir / "test_predictions.csv", index=False)

    summary = {
        "config": {
            "vector_size": config.vector_size,
            "w2v_epochs": config.w2v_epochs,
            "nn_epochs": config.nn_epochs,
            "max_len": config.max_len,
            "batch_size": config.batch_size,
            "max_train_rows": config.max_train_rows,
            "max_val_rows": config.max_val_rows,
            "max_test_rows": config.max_test_rows,
        },
        "best_threshold": best_threshold,
        "best_val_metrics": best_val_metrics,
        "test_metrics": test_metrics,
        "train_size": int(len(train_df)),
        "val_size": int(len(val_df)),
        "test_size": int(len(test_df)),
    }
    dump_json(summary, output_dir / "metrics.json")
    dump_progress(
        progress_path,
        "completed",
        best_threshold=float(best_threshold),
        macro_f1=float(test_metrics["macro_f1"]),
        micro_f1=float(test_metrics["micro_f1"]),
    )
    return summary
