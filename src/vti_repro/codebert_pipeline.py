"""CodeBERT experiment helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from .constants import LABEL_COLUMNS
from .io_utils import dump_progress
from .metrics import compute_metrics


@dataclass
class CodeBERTConfig:
    model_name: str = "microsoft/codebert-base"
    max_length: int = 128
    batch_size: int = 4
    epochs: int = 1
    learning_rate: float = 2e-5
    max_train_rows: int | None = 256
    max_val_rows: int | None = 64
    max_test_rows: int | None = 64
    threshold_grid: tuple[float, ...] = (0.3, 0.4, 0.5, 0.6)


class VTICodeDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, tokenizer: AutoTokenizer, max_length: int):
        self.dataframe = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.dataframe.iloc[index]
        encoded = self.tokenizer(
            row["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        labels = torch.tensor(row[LABEL_COLUMNS].to_numpy(dtype="float32"))
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": labels,
        }


class CodeBERTTagger(nn.Module):
    def __init__(self, model_name: str, num_labels: int, local_files_only: bool = True):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, local_files_only=local_files_only)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs.last_hidden_state[:, 0, :]
        return self.classifier(pooled)


def load_tokenizer(model_name: str, local_files_only: bool = True) -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)


def build_smoke_dataloader(
    data_path: str | Path,
    model_name: str,
    max_length: int,
    batch_size: int,
    sample_size: int,
    local_files_only: bool = True,
) -> DataLoader:
    dataframe = pd.read_csv(data_path, compression="infer").head(sample_size)
    tokenizer = load_tokenizer(model_name=model_name, local_files_only=local_files_only)
    dataset = VTICodeDataset(dataframe, tokenizer, max_length=max_length)
    return DataLoader(dataset, batch_size=batch_size)


def run_codebert_smoke(
    data_path: str | Path,
    config: CodeBERTConfig | None = None,
    sample_size: int = 8,
    local_files_only: bool = True,
) -> dict:
    config = config or CodeBERTConfig()
    dataloader = build_smoke_dataloader(
        data_path=data_path,
        model_name=config.model_name,
        max_length=config.max_length,
        batch_size=config.batch_size,
        sample_size=sample_size,
        local_files_only=local_files_only,
    )
    model = CodeBERTTagger(
        model_name=config.model_name,
        num_labels=len(LABEL_COLUMNS),
        local_files_only=local_files_only,
    )
    model.eval()

    logits_list = []
    labels_list = []
    with torch.no_grad():
        for batch in dataloader:
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            logits_list.append(torch.sigmoid(logits).cpu())
            labels_list.append(batch["labels"].cpu())

    y_score = torch.cat(logits_list, dim=0).numpy()
    y_true = torch.cat(labels_list, dim=0).numpy().astype(int)
    y_pred = (y_score >= 0.5).astype(int)
    return {
        "sample_size": int(len(y_true)),
        "metrics_at_0_5": compute_metrics(y_true, y_pred),
    }


def _subset(df: pd.DataFrame, max_rows: int | None, seed: int = 42) -> pd.DataFrame:
    if max_rows is None or len(df) <= max_rows:
        return df.reset_index(drop=True)
    return df.sample(n=max_rows, random_state=seed).sort_values("sample_id").reset_index(drop=True)


def _load_split(path: str | Path, max_rows: int | None) -> pd.DataFrame:
    df = pd.read_csv(path, compression="infer")
    for label in LABEL_COLUMNS:
        df[label] = df[label].astype(int)
    return _subset(df, max_rows=max_rows)


def _threshold_search(y_true, y_score, threshold_grid):
    rows = []
    best_threshold = threshold_grid[0]
    best_metrics = None
    for threshold in threshold_grid:
        y_pred = (y_score >= threshold).astype(int)
        metrics = compute_metrics(y_true, y_pred)
        rows.append({"threshold": threshold, **metrics})
        if best_metrics is None or metrics["macro_f1"] > best_metrics["macro_f1"]:
            best_threshold = threshold
            best_metrics = metrics
    return best_threshold, best_metrics, pd.DataFrame(rows)


def run_codebert_experiment(
    train_path: str | Path,
    val_path: str | Path,
    test_path: str | Path,
    output_dir: str | Path,
    config: CodeBERTConfig | None = None,
    local_files_only: bool = False,
) -> dict:
    from .io_utils import dump_json, ensure_directory

    config = config or CodeBERTConfig()
    output_dir = ensure_directory(output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    progress_path = output_dir / "progress.json"
    dump_progress(
        progress_path,
        "starting",
        config={
            "model_name": config.model_name,
            "max_length": config.max_length,
            "batch_size": config.batch_size,
            "epochs": config.epochs,
            "learning_rate": config.learning_rate,
            "max_train_rows": config.max_train_rows,
            "max_val_rows": config.max_val_rows,
            "max_test_rows": config.max_test_rows,
        },
        device=str(device),
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

    tokenizer = load_tokenizer(config.model_name, local_files_only=local_files_only)
    train_loader = DataLoader(VTICodeDataset(train_df, tokenizer, config.max_length), batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(VTICodeDataset(val_df, tokenizer, config.max_length), batch_size=config.batch_size)
    test_loader = DataLoader(VTICodeDataset(test_df, tokenizer, config.max_length), batch_size=config.batch_size)
    dump_progress(
        progress_path,
        "dataloaders_ready",
        train_batches=int(len(train_loader)),
        val_batches=int(len(val_loader)),
        test_batches=int(len(test_loader)),
    )

    model = CodeBERTTagger(config.model_name, len(LABEL_COLUMNS), local_files_only=local_files_only).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    dump_progress(progress_path, "model_loaded", device=str(device))

    history = []
    model.train()
    for epoch in range(config.epochs):
        total_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        history.append({"epoch": epoch + 1, "train_loss": total_loss / max(len(train_loader), 1)})
        dump_progress(
            progress_path,
            "epoch_complete",
            epoch=int(epoch + 1),
            train_loss=float(history[-1]["train_loss"]),
        )

    def predict(loader):
        model.eval()
        scores = []
        labels = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                scores.append(torch.sigmoid(logits).cpu())
                labels.append(batch["labels"].cpu())
        y_score = torch.cat(scores).numpy()
        y_true = torch.cat(labels).numpy().astype(int)
        return y_true, y_score

    y_val, val_scores = predict(val_loader)
    best_threshold, best_val_metrics, threshold_df = _threshold_search(y_val, val_scores, config.threshold_grid)
    dump_progress(
        progress_path,
        "threshold_selected",
        best_threshold=float(best_threshold),
        best_macro_f1=float(best_val_metrics["macro_f1"]),
    )
    y_test, test_scores = predict(test_loader)
    test_preds = (test_scores >= best_threshold).astype(int)
    test_metrics = compute_metrics(y_test, test_preds)

    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
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
            "model_name": config.model_name,
            "max_length": config.max_length,
            "batch_size": config.batch_size,
            "epochs": config.epochs,
            "learning_rate": config.learning_rate,
            "max_train_rows": config.max_train_rows,
            "max_val_rows": config.max_val_rows,
            "max_test_rows": config.max_test_rows,
        },
        "device": str(device),
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
