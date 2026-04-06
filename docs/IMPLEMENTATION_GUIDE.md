# VTI Reproduction Guide

## 1. Mục tiêu
- Học lại bài toán VTI theo thứ tự: kiến thức nền, chuẩn hóa dữ liệu, `Base`, rồi mới tới các mô hình nặng hơn.
- Tách phần dùng chung thành `src/vti_repro/` để notebook chỉ còn những bước dễ đọc.

## 2. Cấu trúc mới
- `src/vti_repro/labels.py`: map nhãn thô của BigVul sang 8 nhãn trong bài báo.
- `src/vti_repro/data_prep.py`: đọc CSV gốc theo kiểu streaming, bỏ record không dùng được, tạo split cố định.
- `src/vti_repro/preprocessing.py`: tiền xử lý code cho baseline TF-IDF.
- `src/vti_repro/metrics.py`: exact match ratio, hamming score, accuracy, macro/micro/weighted/sample F1.
- `src/vti_repro/base_pipeline.py`: baseline `TF-IDF + Binary Relevance + GaussianNB`.
- `scripts/prepare_vti_dataset.py`: tạo dataset sạch.
- `scripts/run_base_experiment.py`: chạy lại baseline.

## 3. Luồng chạy tối thiểu
1. Tạo dataset sạch trên một sample nhỏ để kiểm tra pipeline:
```bash
python3 scripts/prepare_vti_dataset.py --max-rows 20000 --output-dir artifacts/data_sample
```
2. Chạy baseline trên sample vừa tạo:
```bash
python3 scripts/run_base_experiment.py \
  --train artifacts/data_sample/train.csv.gz \
  --val artifacts/data_sample/val.csv.gz \
  --test artifacts/data_sample/test.csv.gz \
  --output-dir artifacts/base_sample
```
3. Khi sample chạy ổn, bỏ `--max-rows` để chuẩn bị full dataset.

## 4. Những gì đã được sửa so với notebook cũ
- Không còn `fit` vectorizer trên test set.
- Không đọc trực tiếp các file `YOUR_*_PATH`.
- Không phụ thuộc vào cột đã xử lý sẵn như `processed_func`.
- Split `train/val/test` là cố định và được lưu lại.
- Chọn `chi-square` trên `val`, không nhìn vào `test`.

## 5. Phần còn mở
- `Word2Vec`: còn thiếu môi trường chung có cả `gensim` và `torch`.
- `CodeBERT`: đã có scaffold nhưng cần model weights có sẵn local hoặc internet để tải.
- `Enhanced`: cần thêm dữ liệu Joern/CPG ngoài repo hiện tại.
