# Báo cáo tóm tắt quá trình tái lập thực nghiệm

## 1. Mục tiêu và cách tiếp cận

Trong giai đoạn này, mục tiêu chính của em là tái lập lại pipeline thực nghiệm của bài toán phân loại đa nhãn lỗ hổng phần mềm trong phạm vi tài nguyên thực tế mà em có. Cụ thể, em tập trung vào ba nhánh mô hình nền là `Base`, `Word2Vec` và `CodeBERT`, sau đó thử thêm thành phần `Enhanced` vì đây là phần tác giả đề xuất để cải thiện kết quả dự đoán.

Cách tiếp cận của em không phải là cố ép tất cả mọi thứ chạy đúng nguyên bản bất chấp tài nguyên, mà là làm theo hướng thực dụng hơn: giữ nguyên logic pipeline và cách đánh giá, sau đó điều chỉnh kích thước dữ liệu hoặc số chiều đặc trưng ở những chỗ quá nặng. Em ưu tiên hai tiêu chí:

1. Kết quả phải chạy được và kiểm chứng được.
2. Những gì báo cáo phải nói rõ setting nào là full, setting nào là reduced-full để tránh gây hiểu nhầm.

## 2. Hiểu nội dung bài báo trước khi làm thực nghiệm

Trước khi bắt tay vào chạy thử nghiệm, em dành thời gian đọc lại phần mô tả bài báo và tài liệu trong repo để hiểu rõ tác giả đang giải quyết bài toán gì. Theo phần giới thiệu trong [README.md](/Users/ldt/UET/KLTN/papers/VTI-Project-2/README.md:1), bài báo không tập trung vào phát hiện xem đoạn mã có lỗ hổng hay không, mà đi vào một bài toán hẹp hơn nhưng cũng rất quan trọng, đó là **xác định loại lỗ hổng** của đoạn mã dễ bị tấn công. Nói cách khác, đầu ra không phải chỉ là nhãn “có lỗi” hay “không có lỗi”, mà là các nhóm kiểu lỗ hổng như `DoS`, `Overflow`, `Exec Code`, `Mem. Corr.`, `Bypass`, `Priv`, `Info`, `Other`.

Điểm em hiểu được từ bài báo là tác giả mô hình hóa đây là một **bài toán phân loại đa nhãn**. Một đoạn mã có thể đồng thời thuộc nhiều loại lỗ hổng, nên không thể dùng cách phân loại một nhãn thông thường. Đây là lý do tại sao trong toàn bộ pipeline, cả dữ liệu lẫn đánh giá đều xoay quanh các vector nhãn nhị phân thay vì chỉ một cột nhãn duy nhất.

Một ý nữa em thấy rất quan trọng là bài báo không chỉ so sánh các mô hình học sâu, mà còn đặt ra một câu hỏi khá thú vị: liệu một cách tiếp cận cổ điển như `TF-IDF` có thực sự yếu hơn nhiều so với các mô hình embedding hiện đại hay không. Từ đó, bài báo xây dựng ba nhóm để so sánh:

- một baseline cổ điển dựa trên `bag-of-words/TF-IDF`,
- các mô hình neural hoặc pre-trained như `Word2Vec` và `CodeBERT`,
- và một thành phần hậu xử lý nhẹ tên là `Enhanced`.

Phần em chú ý nhất là `Enhanced`, vì đây chính là đóng góp phương pháp của bài báo. Ý tưởng của tác giả là dự đoán loại lỗ hổng có thể được cải thiện nếu khai thác thêm những token đặc trưng xuất hiện trong các phần quan trọng của mã nguồn. Nói đơn giản, nếu một loại lỗ hổng thường đi kèm với một số định danh, lời gọi hàm hoặc cấu trúc điều khiển đặc biệt, thì có thể dùng các tín hiệu đó để sửa hoặc tinh chỉnh dự đoán ban đầu của mô hình nền.

Chính vì hiểu bài báo theo cách đó, em không triển khai thực nghiệm theo kiểu chỉ chạy cho có kết quả, mà đi theo đúng logic của tác giả:

1. chuẩn bị dữ liệu đa nhãn đúng format,
2. chạy các mô hình nền để có kết quả gốc,
3. lưu lại prediction artifact,
4. sau đó mới đưa prediction này qua `Enhanced` để so sánh trước và sau.

Nói cách khác, phần thực nghiệm của em được xây dựng dựa trên cách hiểu rằng: trọng tâm của bài báo không chỉ nằm ở việc “mô hình nào cao điểm hơn”, mà còn nằm ở việc kiểm tra xem một thành phần hậu xử lý nhẹ, độc lập với backbone, có thực sự giúp cải thiện bài toán nhận diện loại lỗ hổng hay không.

## 3. Các bước chuẩn bị ban đầu

Trước khi đi vào chạy mô hình, việc đầu tiên em làm là đọc lại cấu trúc repo để hiểu bài toán đang được tổ chức như thế nào. Em kiểm tra các thư mục chính như `src/`, `scripts/`, `artifacts/` và `notebooks/` để xác định:

- script nào dùng để chạy từng mô hình,
- dữ liệu đã được xử lý nằm ở đâu,
- đầu ra của mỗi lần chạy được lưu ở đâu,
- và phần `Enhanced` đang phụ thuộc vào những file trung gian nào.

Sau đó, em chuẩn bị môi trường chạy theo hai hướng song song là local và Colab.

Ở local, em dùng máy cá nhân để chạy những phần có thể kiểm soát tốt bằng CPU và thuận tiện theo dõi file đầu ra, ví dụ như `Base 5k`, `Base 10k`, một số lần chạy `Word2Vec`, và các bước xử lý artifact phục vụ `Enhanced`. Trong quá trình này, em cũng gặp vấn đề phiên bản Python với một số thư viện như `gensim` và `tensorflow`, nên có những lần em phải dùng môi trường Python riêng để tránh xung đột dependency.

Ở Colab, em dùng chủ yếu cho các phần nặng hơn hoặc cần GPU, đặc biệt là `CodeBERT`. Vì vậy em cũng chỉnh lại notebook để luồng chạy trên Colab rõ ràng hơn: mount Google Drive, clone repo, kiểm tra dữ liệu, cài package theo từng nhóm, rồi mới chạy mô hình. Mục đích của việc này là để giảm tình trạng cell chạy rất lâu nhưng khó biết đang lỗi ở đâu.

Về cài đặt thư viện, em không cài một lần toàn bộ theo kiểu “gom hết vào một cell”, mà tách theo từng nhóm để dễ theo dõi hơn:

- nhóm cơ bản cho xử lý dữ liệu như `pandas`, `numpy`, `scipy`, `scikit-learn`,
- nhóm cho `Word2Vec` như `gensim`, `tensorflow`, `sentencepiece`, `accelerate`,
- nhóm cho `CodeBERT` như `torch` và `transformers`.

Làm như vậy giúp em biết chính xác mô hình nào phụ thuộc thư viện nào, và khi có lỗi thì khoanh vùng nhanh hơn.

Sau bước môi trường là bước chuẩn bị dữ liệu. Em kiểm tra hai bộ dữ liệu đã được repo sinh sẵn:

- `artifacts/data_sample`
- `artifacts/data_full`

Em dùng các file `summary.json` để xác nhận số mẫu giữ lại sau khi làm sạch, kích thước train/val/test và phân bố nhãn. Việc này quan trọng vì nếu không kiểm tra từ đầu thì rất dễ chạy mô hình trên một tập không đúng với giả định ban đầu.

Ngoài ra, em cũng theo dõi rất kỹ các file đầu ra của từng lần thực nghiệm, ví dụ:

- `metrics.json` để lấy các chỉ số cuối cùng,
- `progress.json` để biết mô hình đang dừng ở bước nào,
- `test_predictions.csv`, `raw_preds.csv`, `labels.csv` để phục vụ so sánh hoặc dùng tiếp cho `Enhanced`.

Nói ngắn gọn, trước khi thật sự báo cáo về mô hình, em đã làm ba việc nền:

1. Hiểu cấu trúc repo và luồng chạy của từng script.
2. Dựng môi trường local và Colab theo đúng nhu cầu của từng mô hình.
3. Kiểm tra dữ liệu, dependency và artifact đầu ra để đảm bảo mỗi bước về sau đều có thể kiểm chứng lại.

## 4. Những gì em đã làm được

### 4.1. Xử lý và chạy trên bộ sample ban đầu

Trước khi chuyển sang các thực nghiệm lớn hơn, em có làm một bước trung gian trên bộ `sample` để kiểm tra xem toàn bộ pipeline có chạy đúng hay không. Đây là bước khá quan trọng, vì nếu vào thẳng full data thì rất khó phân biệt lỗi đến từ code, từ dữ liệu hay từ giới hạn tài nguyên.

Từ file dữ liệu gốc, em đã tạo được bộ `artifacts/data_sample` sau khi làm sạch và chuẩn hóa nhãn. Theo [artifacts/data_sample/summary.json](/Users/ldt/UET/KLTN/papers/VTI-Project-2/artifacts/data_sample/summary.json:1), bộ này giữ lại `17,341` mẫu có nhãn hợp lệ, sau đó được chia thành:

- `train`: `13,896`
- `val`: `1,684`
- `test`: `1,761`

Ở giai đoạn này, mục tiêu chính của em không phải là lấy số đẹp để báo cáo, mà là:

- xác minh dữ liệu sau tiền xử lý đã đúng định dạng,
- kiểm tra các script huấn luyện có chạy hết từ đầu đến cuối,
- kiểm tra output như `metrics.json`, `test_predictions.csv`, `raw_preds.csv`, `labels.csv`,
- và thử nghiệm sớm phần `Enhanced`/Joern trên quy mô nhỏ hơn để giảm rủi ro kỹ thuật.

Nhờ bước này, em xác nhận được pipeline cơ bản là chạy được, rồi mới chuyển sang `data_full` và các thiết lập reduced-full. Vì vậy, trong báo cáo em xem phần `sample` là bước **kiểm tra kỹ thuật và xác minh pipeline**, còn phần kết quả chính vẫn dựa trên các thực nghiệm lớn hơn ở `data_full`.

### 4.2. Tái lập các mô hình nền

Em đã chạy được các mô hình sau:

- `Base` với full split nhưng giảm số chiều TF-IDF xuống `5000` và `10000`.
- `Word2Vec` với các mức reduced-full `30k/5k/5k` và `50k/8k/8k`.
- `CodeBERT` với setting `30k/5k/5k`.

Trong quá trình chạy, em cũng kiểm tra được rằng:

- `Base` ở mức `20000` features không phù hợp với máy `16 GB RAM` và cả Colab CPU standard, vì bước biến ma trận sparse sang dense làm tốn rất nhiều RAM.
- `Word2Vec` full cũng khá nặng trên CPU do bước huấn luyện embedding bằng `gensim`, nên em phải chuyển sang reduced-full để có kết quả hoàn chỉnh trong thời gian hợp lý.
- `CodeBERT` phù hợp hơn với Colab GPU, nhưng cũng cần giảm dữ liệu xuống `30k/5k/5k` để chạy ổn định.

### 4.3. Triển khai phần Enhanced theo hướng Joern-backed

Sau khi có kết quả của các mô hình nền, em tiếp tục triển khai `Enhanced` theo hướng dùng Joern thật thay vì chỉ dùng fallback từ text. Phần này em đã làm được các việc sau:

- Viết thêm runner [scripts/run_joern_aligned_enhanced.py](/Users/ldt/UET/KLTN/papers/VTI-Project-2/scripts/run_joern_aligned_enhanced.py:1) để:
  - tái tạo train/test view đúng với run đang xét,
  - chuẩn bị `raw_preds.csv` và `labels.csv`,
  - materialize source code,
  - chạy `joern-parse`,
  - dump node từ CPG,
  - ghép lại thành cột `cpg`,
  - rồi gọi pipeline `Enhanced`.
- Tối ưu lại phần tính feature table trong [src/vti_repro/enhanced_notebook_compat.py](/Users/ldt/UET/KLTN/papers/VTI-Project-2/src/vti_repro/enhanced_notebook_compat.py:107) để tránh vòng lặp quá chậm khi số dòng train lớn.

Nhờ đó, em đã chạy được `Enhanced` cho:

- `Word2Vec 30k` theo kiểu Joern-backed aligned.
- `CodeBERT 30k` theo kiểu Joern-backed aligned.
- `Base 10k` theo kiểu Joern-backed nhưng dùng `train subset 16384` và `full test`, vì full-train Joern parse quá nặng.

## 5. Cách em xử lý bài toán tài nguyên

Trong quá trình làm, em gặp ba nút thắt chính.

Thứ nhất là `Base`. Mô hình này dùng TF-IDF và có bước chuyển sparse matrix sang dense trước khi huấn luyện, nên khi chạy full với `20000` features thì rất dễ vượt RAM. Vì vậy, em chọn hai mức `5000` và `10000` features để so sánh, và dùng `10000` làm baseline chính vì kết quả tốt hơn rõ rệt mà vẫn chạy được.

Thứ hai là `Word2Vec`. Dù phần mạng phía sau không quá lớn, bước huấn luyện embedding với `gensim` trên full train vẫn chậm và nặng CPU. Do đó, em giảm dữ liệu xuống `30k` và `50k` để đảm bảo có kết quả hoàn chỉnh.

Thứ ba là `Enhanced` với Joern. Vấn đề ở đây không chỉ là chạy Joern, mà còn phải đảm bảo dữ liệu `train/test`, `raw_preds.csv` và `labels.csv` thật sự khớp nhau. Em ưu tiên tính đúng trước, nên chỉ chạy những trường hợp mà em xác nhận được alignment qua `sample_id` và số dòng. Chính vì vậy, bước chạy trên `sample` ban đầu cũng giúp em kiểm tra trước logic này ở quy mô nhỏ, rồi mới áp dụng sang các setting lớn hơn.

## 6. Kết quả chính

### 6.1. Kết quả của các mô hình nền

Hai run thăm dò quan trọng mà em giữ lại để quan sát ảnh hưởng của cấu hình là:

- `Base 5k`: `macro_f1 = 0.3833`, `micro_f1 = 0.4455`
- `Word2Vec 50k`: `macro_f1 = 0.4483`, `micro_f1 = 0.6574`

Tuy nhiên, để bảng chính nhất quán hơn, em chọn ba baseline đại diện sau:

- `Base 10k`
- `Word2Vec 30k`
- `CodeBERT 30k`

### 6.2. Bảng so sánh trước và sau Enhanced

| Mô hình | Setting | Macro-F1 | Micro-F1 | Weighted-F1 | Exact Match | Hamming Score |
|---|---|---:|---:|---:|---:|---:|
| Base | full split, 10k features | 0.4401 | 0.5154 | 0.5815 | 0.2737 | 0.4746 |
| Base + Enhanced | Joern train-subset 16384, full-test aligned | 0.4463 | 0.5604 | 0.6158 | 0.2278 | 0.4953 |
| Word2Vec | 30k/5k/5k reduced-full | 0.3298 | 0.6269 | 0.5688 | 0.3622 | 0.5517 |
| Word2Vec + Enhanced | Joern-backed aligned 30k/5k/5k | 0.3455 | 0.6299 | 0.5763 | 0.3556 | 0.5543 |
| CodeBERT | 30k/5k/5k reduced-full | 0.6259 | 0.7468 | 0.7373 | 0.5376 | 0.6933 |
| CodeBERT + Enhanced | Joern-backed aligned 30k/5k/5k | 0.6141 | 0.7248 | 0.7130 | 0.4598 | 0.6591 |

### 6.3. Đối chiếu với kết quả và kết luận của bài báo

Đối với baseline `Base`, em có thể đối chiếu khá trực tiếp với bài báo vì repo đã lưu sẵn bảng so sánh trong [artifacts/base_full_p05_10k_local/paper_comparison.csv](/Users/ldt/UET/KLTN/papers/VTI-Project-2/artifacts/base_full_p05_10k_local/paper_comparison.csv:1). Nếu lấy một vài chỉ số quan trọng thì kết quả như sau:

| Chỉ số | Bài báo | Kết quả em tái lập (`Base 10k`) | Chênh lệch |
|---|---:|---:|---:|
| Macro-F1 | 0.62 | 0.4401 | -0.1799 |
| Micro-F1 | 0.70 | 0.5154 | -0.1846 |
| Weighted-F1 | 0.70 | 0.5815 | -0.1185 |
| Exact Match Ratio | 0.54 | 0.2737 | -0.2663 |
| Hamming Score | 0.62 | 0.4746 | -0.1454 |

Có thể thấy baseline mà em tái lập thấp hơn khá rõ so với số của bài báo. Theo em, nguyên nhân lớn nhất là em không thể giữ đúng full setting nguyên bản. Cụ thể, `Base` phải giảm xuống `10000` features để chạy được ổn định trên máy `16 GB RAM`, trong khi bài báo và logic repo hướng đến một không gian đặc trưng lớn hơn. Với mô hình dựa trên `TF-IDF`, việc giảm số chiều như vậy có thể làm mất nhiều tín hiệu phân biệt.

Ngoài ra, môi trường thực nghiệm của em cũng khác với điều kiện lý tưởng của tác giả. Em phải chia việc chạy giữa local và Colab, và nhiều lần phải đổi cấu hình vì giới hạn RAM hoặc thời gian. Vì vậy, kết quả của em nên được hiểu là một bản tái lập thực dụng trong điều kiện tài nguyên hạn chế, chứ không phải sao chép hoàn toàn môi trường gốc của bài báo.

Nếu nhìn ở mức xu hướng, kết quả của em vẫn cho thấy một ý khá phù hợp với bài báo: baseline cổ điển không hề là một mốc yếu. Dù thấp hơn paper, `Base 10k` vẫn đủ mạnh để cạnh tranh với một số thiết lập neural, chứ không phải chỉ đóng vai trò đối chứng hình thức.

Đối với các mô hình neural, em chưa có trong repo một bảng số paper đầy đủ để đối chiếu từng chỉ số như baseline `Base`. Tuy nhiên, nếu đối chiếu theo luận điểm chính mà bài báo nêu trong [README.md](/Users/ldt/UET/KLTN/papers/VTI-Project-2/README.md:1), em thấy có hai điểm đáng bàn.

Thứ nhất, bài báo cho rằng các mô hình học sâu không vượt trội quá xa so với baseline cổ điển. Kết quả của em chỉ khớp một phần với nhận định này:

- `Word2Vec 30k` có `macro_f1 = 0.3298`, thấp hơn `Base 10k = 0.4401`. Ở điểm này, kết quả của em khá phù hợp với tinh thần bài báo.
- `CodeBERT 30k` có `macro_f1 = 0.6259`, cao hơn rõ rệt so với `Base 10k`. Điều này cho thấy trong setting reduced-full mà em chạy, `CodeBERT` tỏ ra vượt trội hơn baseline.

Theo em, sự khác biệt này đến từ việc phép so sánh hiện tại chưa hoàn toàn cùng mặt bằng. `CodeBERT` được chạy trên GPU với backbone mạnh, còn `Base` lại là phiên bản đã phải giảm xuống `10k` features để tránh lỗi bộ nhớ. Vì vậy, việc `CodeBERT` mạnh hơn nhiều trong thực nghiệm của em không nhất thiết phủ định kết luận của bài báo, mà chủ yếu phản ánh sự khác nhau về điều kiện tái lập.

Thứ hai, bài báo nhấn mạnh rằng `Enhanced` là một thành phần nhẹ nhưng có thể cải thiện kết quả, thậm chí README còn nêu rằng nó có thể giúp các mô hình neural tăng mạnh trong một số trường hợp. Kết quả của em xác nhận điều này ở mức có điều kiện:

- `Base 10k + Enhanced`: `macro_f1` tăng từ `0.4401` lên `0.4463`, `micro_f1` tăng từ `0.5154` lên `0.5604`.
- `Word2Vec 30k + Enhanced`: `macro_f1` tăng từ `0.3298` lên `0.3455`, `micro_f1` tăng từ `0.6269` lên `0.6299`.
- `CodeBERT 30k + Enhanced`: `macro_f1` giảm từ `0.6259` xuống `0.6141`, `micro_f1` giảm từ `0.7468` xuống `0.7248`.

Như vậy, trong thực nghiệm của em, `Enhanced` không cải thiện đồng đều trên mọi backbone. Nó có tác dụng rõ hơn với `Base` và `Word2Vec`, nhưng lại không giúp `CodeBERT 30k`. Theo em, có hai lời giải thích hợp lý.

Một là khi mô hình nền đã mạnh, prediction ban đầu của nó đã đủ tốt, nên việc thêm một lớp hậu xử lý bằng luật có thể bắt đầu sửa quá tay. Khi đó recall có thể tăng nhưng precision giảm, và tổng thể metric cuối cùng lại xấu đi.

Hai là các thí nghiệm `Enhanced` của em chưa phải full setting như bài báo. Với `Base`, em phải dùng `train subset 16384` thay vì full train cho phần Joern. Với `Word2Vec` và `CodeBERT`, em chỉ chạy được ở mức `30k/5k/5k`. Vì vậy, mức cải thiện nhỏ hơn bài báo, hoặc thậm chí giảm ở `CodeBERT`, là điều có thể hiểu được.

Tóm lại, nếu so với bài báo, em sẽ diễn đạt theo hướng sau:

- Về mặt xu hướng, kết quả của em **ủng hộ một phần** kết luận của tác giả: baseline cổ điển không yếu, và `Enhanced` có thể giúp cải thiện một số mô hình.
- Tuy nhiên, mức độ cải thiện trong thực nghiệm của em **không mạnh và không đồng đều** như mô tả tổng quát trong bài báo.
- Sự khác biệt chủ yếu đến từ ràng buộc tài nguyên, việc phải dùng reduced-full setting, và việc chưa thể giữ nguyên toàn bộ thiết lập gốc của tác giả.

## 7. Nhận xét từ kết quả

Điểm dễ thấy nhất là `CodeBERT 30k` đang là mô hình mạnh nhất trong các run em đã hoàn thành. Chỉ xét baseline, nó cao hơn khá rõ so với `Base 10k` và `Word2Vec 30k` ở hầu hết các chỉ số chính.

Đối với `Enhanced`, kết quả không đồng đều giữa các backbone:

- Với `Base 10k`, `Enhanced` giúp tăng `macro_f1`, `micro_f1`, `weighted_f1` và `hamming_score`. Dù `exact_match_ratio` giảm, nhìn chung đây vẫn là một cải thiện có ý nghĩa nếu ưu tiên khả năng bắt đúng nhãn hơn là độ khớp tuyệt đối toàn bộ mẫu.
- Với `Word2Vec 30k`, `Enhanced` cũng cải thiện nhưng mức tăng khá nhẹ. Điều này cho thấy luật hậu xử lý có tác dụng, nhưng chưa tạo ra bước nhảy lớn.
- Với `CodeBERT 30k`, `Enhanced` lại làm kết quả giảm. Điều này cho thấy phần hậu xử lý không phải lúc nào cũng giúp ích; khi backbone đã mạnh, việc áp rule thủ công có thể làm mất bớt độ chính xác ban đầu.

Từ đây, em thấy có thể rút ra một nhận xét tương đối tự nhiên: `Enhanced` có xu hướng hữu ích hơn cho các mô hình nền yếu hoặc trung bình, còn với mô hình mạnh hơn như `CodeBERT`, tác dụng của nó cần được kiểm tra kỹ hơn chứ không nên mặc định là sẽ tăng kết quả.

## 8. Hạn chế và cách em trình bày trong báo cáo

Em nghĩ phần này cần nói rõ để báo cáo trung thực.

Thứ nhất, không phải tất cả kết quả đều là full reproduction đúng nguyên bản. Một số run là `reduced-full`, ví dụ `Word2Vec 30k`, `Word2Vec 50k` và `CodeBERT 30k`. Em vẫn giữ nguyên pipeline và cách đánh giá, nhưng giảm số lượng mẫu để phù hợp tài nguyên.

Thứ hai, với `Base + Enhanced`, em chưa chạy được full-train Joern parse vì quá nặng. Thay vào đó, em dùng `train subset 16384` và `full test`. Em cho rằng đây là một compromise hợp lý, vì test vẫn là full test của baseline chính, còn train subset vẫn giữ được bản chất của phương pháp.

Thứ ba, kết quả `Enhanced` không nên được mô tả như một cải tiến chắc chắn cho mọi mô hình. Kết quả thực tế của em cho thấy điều đó không đúng trong mọi trường hợp.

## 9. Kết luận

Tóm lại, trong phạm vi thời gian và tài nguyên hiện có, em đã tái lập được các mô hình nền quan trọng của bài toán và triển khai được thêm phần `Enhanced` theo hướng Joern-backed cho các setting phù hợp. Kết quả cho thấy:

- `CodeBERT 30k` là baseline mạnh nhất trong các run chính.
- `Enhanced` có thể cải thiện `Base` và `Word2Vec`.
- `Enhanced` không cải thiện `CodeBERT 30k` trong setting em chạy, thậm chí còn làm giảm một số chỉ số.

Vì vậy, nếu trình bày ngắn gọn trước thầy, em sẽ nói rằng phần tái lập của em không chỉ dừng ở việc chạy mô hình nền, mà còn đi tiếp tới phần đóng góp chính của bài báo là `Enhanced`, đồng thời kiểm tra được rằng hiệu quả của phần này còn phụ thuộc khá nhiều vào backbone và setting thực nghiệm.
