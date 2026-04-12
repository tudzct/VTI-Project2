"""Dataset adapter views for legacy notebooks and local experiment runners."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .constants import LABEL_COLUMNS


@dataclass
class ViewConfig:
    train_path: str | Path
    val_path: str | Path
    test_path: str | Path
    output_dir: str | Path


def load_split(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, compression="infer")
    for label in LABEL_COLUMNS:
        df[label] = df[label].astype(int)
    return df


def _label_list(row: pd.Series) -> list[str]:
    return [label for label in LABEL_COLUMNS if int(row[label]) == 1]


def make_legacy_view(df: pd.DataFrame) -> pd.DataFrame:
    legacy = df.copy().reset_index(drop=True)
    legacy.insert(0, "index", legacy.index)
    legacy["processed_func"] = legacy["text"]
    legacy["func_before"] = legacy["text"]
    legacy["code"] = legacy["text"]
    legacy["+info"] = legacy["info"]
    legacy["+priv"] = legacy["priv"]
    legacy["mem."] = legacy["mem"]
    legacy["exec_code"] = legacy["exec"]
    legacy["mem_corr"] = legacy["mem"]
    legacy["tags"] = legacy.apply(_label_list, axis=1)
    legacy["labels"] = legacy["tags"].apply(lambda items: ",".join(items))
    if "cpg" not in legacy.columns:
        legacy["cpg"] = ""
    return legacy


def build_views(config: ViewConfig) -> dict[str, str]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    for split_name, path in (
        ("train", config.train_path),
        ("val", config.val_path),
        ("test", config.test_path),
    ):
        df = load_split(path)
        legacy = make_legacy_view(df)
        csv_path = output_dir / f"{split_name}_legacy.csv"
        hash_path = output_dir / f"{split_name}_legacy_hash.csv"
        legacy.to_csv(csv_path, index=False)
        legacy.to_csv(hash_path, index=False, sep="#")
        outputs[f"{split_name}_csv"] = str(csv_path)
        outputs[f"{split_name}_hash_csv"] = str(hash_path)
    return outputs
