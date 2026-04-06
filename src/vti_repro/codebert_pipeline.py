"""Optional CodeBERT experiment scaffolding."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from .constants import LABEL_COLUMNS
from .metrics import compute_metrics


@dataclass
class CodeBERTConfig:
    model_name: str = "microsoft/codebert-base"
    max_length: int = 512
    batch_size: int = 4
    epochs: int = 1
    learning_rate: float = 2e-5


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
