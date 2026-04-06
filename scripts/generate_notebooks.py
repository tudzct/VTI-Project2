#!/usr/bin/env python3
"""Generate lightweight teaching notebooks for the VTI reproduction workflow."""

from __future__ import annotations

import json
from pathlib import Path


def markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.strip().splitlines()],
    }


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip().splitlines()],
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_notebook(path: Path, cells: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(notebook(cells), indent=2), encoding="utf-8")


def main() -> int:
    write_notebook(
        Path("notebooks/00_vti_foundations.ipynb"),
        [
            markdown_cell(
                """
                # Phase 0: Foundations for the VTI paper

                Notebook này dành cho việc học khái niệm trước khi chạy mô hình.
                Hãy đọc từng ô markdown rồi mới chạy ô code nhỏ bên dưới.
                """
            ),
            markdown_cell(
                """
                ## 1. Bài toán trong bài báo là gì?

                - Đầu vào: một hàm C/C++ đã biết là có lỗ hổng.
                - Đầu ra: một hoặc nhiều loại lỗ hổng của hàm đó.
                - Đây là **multi-label classification** vì một hàm có thể mang nhiều nhãn cùng lúc.

                Ví dụ: một hàm có thể vừa `DoS` vừa `Overflow`.
                """
            ),
            markdown_cell(
                """
                ## 2. Các khái niệm phải nắm

                - `train / validation / test`: học trên train, chọn tham số bằng validation, chỉ báo cáo cuối trên test.
                - `TF-IDF`: biểu diễn văn bản bằng tần suất token có trọng số.
                - `Word2Vec`: học vector embedding cho token dựa trên ngữ cảnh.
                - `CodeBERT`: mô hình transformer đã pretrain cho mã nguồn.
                - `precision / recall / F1`: đo độ đúng, độ phủ, và cân bằng giữa hai thứ đó.
                - `exact match ratio`: dự đoán đúng toàn bộ bộ nhãn của một mẫu.
                - `hamming score`: mức trùng giữa tập nhãn dự đoán và tập nhãn thật.
                """
            ),
            code_cell(
                """
                import sys
                from pathlib import Path

                ROOT = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
                sys.path.insert(0, str(ROOT / "src"))

                from vti_repro.labels import normalize_labels

                examples = [
                    "DoS Overflow",
                    "DoS Exec Code Overflow Mem. Corr.",
                    "Bypass +Info",
                    "Dir. Trav.",
                ]

                for item in examples:
                    print(item, "->", normalize_labels(item))
                """
            ),
            markdown_cell(
                """
                ## 3. Bản đồ bài báo

                - `Base`: TF-IDF + Binary Relevance + GaussianNB.
                - `Word2Vec`: embedding không ngữ cảnh + BiLSTM + attention.
                - `CodeBERT`: embedding ngữ cảnh + tầng phân loại đa nhãn.
                - `Enhanced`: hậu xử lý dùng distinguishing tokens từ các thành phần cú pháp quan trọng.

                Trong repo mới, bạn nên chạy theo đúng thứ tự đó.
                """
            ),
        ],
    )

    write_notebook(
        Path("notebooks/01_prepare_data.ipynb"),
        [
            markdown_cell(
                """
                # Phase 1: Prepare the VTI dataset

                Notebook này không cố làm mọi thứ trong một ô.
                Nó gọi các hàm đã được đóng gói trong `src/vti_repro/`.
                """
            ),
            code_cell(
                """
                import sys
                from pathlib import Path

                ROOT = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
                sys.path.insert(0, str(ROOT / "src"))

                from vti_repro.data_prep import prepare_vti_dataset
                """
            ),
            markdown_cell(
                """
                Chạy trước trên sample nhỏ để kiểm tra mapping nhãn, split, và schema.
                Khi đã ổn, tăng dần `max_rows` hoặc bỏ hẳn để chạy full.
                """
            ),
            code_cell(
                """
                summary = prepare_vti_dataset(
                    raw_csv_path=ROOT / "MSR_data_cleaned.csv",
                    output_dir=ROOT / "artifacts" / "data_sample",
                    seed=42,
                    max_rows=20_000,
                    report_every=2_000,
                )
                summary
                """
            ),
            code_cell(
                """
                import json
                from pathlib import Path

                ROOT = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
                summary_path = ROOT / "artifacts" / "data_sample" / "summary.json"
                print(summary_path.read_text()[:2000])
                """
            ),
        ],
    )

    write_notebook(
        Path("notebooks/02_base_baseline.ipynb"),
        [
            markdown_cell(
                """
                # Phase 2: Reproduce the classical baseline

                Mục tiêu của notebook này là hiểu rõ vì sao baseline của bài báo lại mạnh.
                """
            ),
            code_cell(
                """
                import sys
                from pathlib import Path

                ROOT = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
                sys.path.insert(0, str(ROOT / "src"))

                from vti_repro.base_pipeline import run_base_experiment
                """
            ),
            markdown_cell(
                """
                Chỉ chạy notebook này sau khi bạn đã có `train.csv.gz`, `val.csv.gz`, `test.csv.gz`.
                """
            ),
            code_cell(
                """
                result = run_base_experiment(
                    train_path=ROOT / "artifacts" / "data_sample" / "train.csv.gz",
                    val_path=ROOT / "artifacts" / "data_sample" / "val.csv.gz",
                    test_path=ROOT / "artifacts" / "data_sample" / "test.csv.gz",
                    output_dir=ROOT / "artifacts" / "base_sample",
                )
                result["test_metrics"]
                """
            ),
            code_cell(
                """
                import pandas as pd
                from pathlib import Path

                ROOT = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
                comparison = pd.read_csv(ROOT / "artifacts" / "base_sample" / "paper_comparison.csv")
                comparison
                """
            ),
        ],
    )

    write_notebook(
        Path("notebooks/03_next_steps.ipynb"),
        [
            markdown_cell(
                """
                # Phases 3-5: Word2Vec, CodeBERT, Enhanced

                Notebook này là bản đồ cho các bước tiếp theo.
                Phần code nặng đã được tách ra khỏi notebook vì phụ thuộc môi trường và tài nguyên.
                """
            ),
            markdown_cell(
                """
                ## Word2Vec
                - Cần `gensim` và `torch` trong cùng một Python environment.
                - Entry point: `scripts/run_word2vec_experiment.py`

                ## CodeBERT
                - Cần model weights `microsoft/codebert-base` có sẵn local hoặc internet để tải.
                - Scaffold: `src/vti_repro/codebert_pipeline.py`

                ## Enhanced
                - Cần export thêm code elements từ Joern/CPG.
                - Entry point: `scripts/run_enhanced_rules.py`
                """
            ),
        ],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
