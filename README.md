# quant_ml — Hành trình học Machine Learning, Deep Learning và Reinforcement Learning

Đây là lab cá nhân để học **Machine Learning (ML)**, **Deep Learning (DL)** và **Reinforcement Learning (RL)** thông qua việc đọc lý thuyết, tự triển khai thuật toán và làm các bài thực hành. Hai hướng ứng dụng đang được khám phá là **tài chính định lượng** và **thị giác máy tính**.

Mục tiêu của lab là hiểu một mô hình từ bên trong: dữ liệu được biểu diễn thế nào, mô hình học điều gì, hàm mất mát được tối ưu ra sao và cần đánh giá kết quả bằng thước đo nào. Mỗi bài học hướng tới chu trình:

**Đặt câu hỏi → hiểu dữ liệu → xây dựng baseline → huấn luyện/thử nghiệm → đánh giá → phân tích lỗi → ghi lại điều học được.**

Repo đang trong quá trình phát triển. Các phần bên dưới phân biệt nội dung đã có trong mã nguồn với lộ trình sẽ tiếp tục thực hiện; chưa có phần triển khai RL.

## Mục lục

- [Trạng thái hiện tại](#trạng-thái-hiện-tại)
- [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [Cài đặt môi trường](#cài-đặt-môi-trường)
- [Các bài thực hành hiện có](#các-bài-thực-hành-hiện-có)
- [Lộ trình học chi tiết](#lộ-trình-học-chi-tiết)
- [Quy trình thực hiện một bài lab](#quy-trình-thực-hiện-một-bài-lab)
- [Nguyên tắc đánh giá và tái lập](#nguyên-tắc-đánh-giá-và-tái-lập)
- [Các bước tiếp theo](#các-bước-tiếp-theo)

## Trạng thái hiện tại

| Nội dung | Trạng thái trong repo | Mục tiêu học |
| --- | --- | --- |
| Logistic Regression với scikit-learn | Có notebook tải Digits, chia train/validation/test, huấn luyện và lấy xác suất trên validation | Phân loại nhị phân, mất cân bằng lớp, ý nghĩa của accuracy và recall |
| Logistic Regression bằng NumPy | Có sigmoid, BCE, gradient, vòng lặp huấn luyện và đánh giá trên dữ liệu giả lập | Hiểu công thức và tự xây dựng mô hình |
| Sai số dự đoán khoảng cách | Có dữ liệu CSV, ví dụ tính MAE/RMSE và script đánh giá theo loại xe | Phân tích lỗi, chọn metric và kiểm tra giả định |
| Nhận diện xe bằng YOLOv8 | Có pipeline xử lý ảnh/video với trọng số pretrained | Thực hành suy luận DL, bounding box và xử lý kết quả |
| Học không giám sát cho giao dịch | Có script tải dữ liệu và xây dựng một phần chỉ báo; cần hoàn thiện và sửa các điểm còn vướng | Feature engineering, clustering, tối ưu và đánh giá danh mục |
| Tự huấn luyện mạng neural | Dự kiến | Tensor, autograd, backpropagation và tối ưu hóa |
| Reinforcement Learning | Dự kiến | MDP, value function, policy và học qua tương tác |

## Cấu trúc thư mục

```text
quant_ml/
├── README.md
├── requirements.txt
├── .gitignore
└── app/
    ├── model-ml/
    │   ├── train-logistic-regression.ipynb
    │   └── train-logistic-regression-math.ipynb
    ├── data/
    │   ├── distance_dataset.csv
    │   ├── test_data.py
    │   └── evaluation.py
    └── ml_service/
        ├── vision_core.py
        ├── vision_orchestration.py
        └── unsupervised.py
```

- `app/model-ml/`: notebook học thuật toán, từ sử dụng thư viện đến tự triển khai công thức.
- `app/data/`: dữ liệu mẫu và các bài thực hành đo lường sai số. `test_data.py` là ví dụ tính metric, chưa phải bộ kiểm thử tự động.
- `app/ml_service/vision_core.py`: các lớp nhận diện xe, ước lượng khoảng cách, đánh giá theo ngưỡng và vẽ kết quả.
- `app/ml_service/vision_orchestration.py`: ghép các bước thành pipeline xử lý frame và video.
- `app/ml_service/unsupervised.py`: bản thực hành đang phát triển về dữ liệu cổ phiếu và chỉ báo kỹ thuật.

## Cài đặt môi trường

Chạy các lệnh sau từ **thư mục gốc của repo**. Môi trường được ghi chú trong `requirements.txt` sử dụng Python 3.11.

### 1. Tạo và kích hoạt virtual environment

```bash
python3.11 -m venv venv
source venv/bin/activate
```

Trên Windows PowerShell, lệnh kích hoạt tương ứng là:

```powershell
.\venv\Scripts\Activate.ps1
```

Nếu đã có `venv`, chỉ cần kích hoạt môi trường đó.

### 2. Cài thư viện

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Các nhóm thư viện được khai báo:

| Nhóm | Thư viện | Vai trò |
| --- | --- | --- |
| Dữ liệu và tính toán | NumPy, pandas | Mảng, phép toán vector, bảng và chuỗi thời gian |
| Machine Learning | scikit-learn | Dataset và mô hình ML |
| Thống kê | statsmodels | Hồi quy, bao gồm rolling regression cho hướng nghiên cứu tài chính |
| Dữ liệu thị trường | yfinance, pandas_datareader | Tải giá và dữ liệu tài chính |
| Danh mục và biến động | PyPortfolioOpt, arch | Phục vụ các bài tối ưu danh mục và mô hình biến động dự kiến |
| Trực quan hóa | Matplotlib | Biểu đồ và phân tích sai số |
| Notebook | notebook, ipykernel | Chạy notebook tương tác |
| Computer Vision | ultralytics, opencv-python | YOLO và xử lý ảnh/video |
| Dịch vụ | fastapi, uvicorn, websockets | Đã khai báo phụ thuộc; chưa có API server trong repo |

Một số import hiện có chưa được khai báo trực tiếp đầy đủ trong `requirements.txt`:

- Notebook NumPy có hai import từ `sympy`, dù phần tính toán hiện tại không sử dụng chúng. Để chạy nguyên trạng notebook, cài thêm `sympy`.
- Script tài chính sử dụng `requests` và `pandas.read_html`; cài trực tiếp `requests` và parser `lxml` để chuẩn bị cho phần này.

```bash
python -m pip install sympy requests lxml
```

`requirements.txt` chưa khóa phiên bản thư viện, nên kết quả và khả năng chạy có thể khác giữa các môi trường. Repo chưa cấu hình môi trường huấn luyện GPU riêng.

### 3. Đăng ký kernel và mở notebook

```bash
python -m ipykernel install --user --name quant-ml --display-name "Python (quant_ml venv)"
python -m notebook
```

Mở notebook trong `app/model-ml/`, chọn kernel **Python (quant_ml venv)** và chạy các cell theo thứ tự. Notebook dùng NumPy hiện lưu metadata kernel `Python 3`; cần chọn lại kernel nếu nó trỏ sang môi trường khác.

## Các bài thực hành hiện có

### 1. Logistic Regression với scikit-learn: accuracy cao đã đủ chưa?

Notebook: [train-logistic-regression.ipynb](app/model-ml/train-logistic-regression.ipynb).

**Câu hỏi:** Nếu phần lớn ảnh không phải số 7, một mô hình luôn đoán “không phải số 7” có thể có accuracy cao. Vậy làm sao biết mô hình thực sự nhận ra số 7?

Nội dung đã triển khai:

1. Nạp bộ dữ liệu `load_digits()` có sẵn trong scikit-learn.
2. Chia giá trị pixel cho `16.0` để đưa đặc trưng về khoảng `[0, 1]`.
3. Chuyển nhãn thành bài toán nhị phân: số 7 là `1`, các chữ số còn lại là `0`.
4. Chia dữ liệu theo tỷ lệ xấp xỉ **60% train / 20% validation / 20% test**.
5. Sử dụng `stratify` để giữ tỷ lệ lớp và `random_state=42` để tái lập cách chia.
6. Huấn luyện `LogisticRegression(max_iter=1000)` và lấy xác suất lớp 1 trên validation.

Output đang lưu cho thấy các tập có kích thước lần lượt là **1.078 / 359 / 360 mẫu**, mỗi mẫu có **64 đặc trưng**. Đây là output lưu trong notebook, chưa phải kết quả chạy lại ở mọi môi trường.

**Phần cần làm tiếp:** xây dựng baseline luôn đoán lớp 0; tính confusion matrix, precision, recall và F1; so sánh các ngưỡng phân loại trên validation. Sau khi chọn ngưỡng, dùng test để đánh giá cuối cùng. Notebook hiện mới dừng ở bước lấy xác suất validation, chưa có kết quả đánh giá test.

### 2. Logistic Regression bằng NumPy: hiểu mô hình từ công thức

Notebook: [train-logistic-regression-math.ipynb](app/model-ml/train-logistic-regression-math.ipynb).

Bài này tự triển khai các thành phần chính của Logistic Regression:

- **Logit:** `z = X @ w + b`.
- **Sigmoid:** chuyển logit thành xác suất lớp 1.
- **Binary Cross-Entropy:** đo sai lệch giữa dự đoán và nhãn thật.
- **Gradient:** tính đạo hàm của loss theo trọng số và bias.
- **Gradient descent:** cập nhật tham số qua nhiều bước.
- **Dự đoán:** chuyển xác suất thành nhãn theo ngưỡng, mặc định là `0.5`.

Các công thức được sử dụng:

```text
p = sigmoid(Xw + b)
BCE = mean(logaddexp(0, z) - y * z)
dw = X.T @ (p - y) / n
db = mean(p - y)
w = w - learning_rate * dw
b = b - learning_rate * db
```

`np.logaddexp` được dùng để tính sigmoid và BCE ổn định hơn về mặt số học. Notebook bắt đầu bằng hai mẫu để kiểm tra một bước cập nhật, sau đó gom logic vào class có các phương thức `fit`, `predict_proba` và `predict`.

Thí nghiệm tiếp theo tạo **500 mẫu giả lập với 2 đặc trưng**, dùng seed `42`, chia **400 mẫu train / 100 mẫu test** và huấn luyện với `learning_rate=0.1`, `n_steps=1000`.

Kết quả đang lưu trong notebook:

| Chỉ số | Giá trị |
| --- | ---: |
| BCE train ban đầu | 0.6931 |
| BCE train sau cập nhật cuối | 0.2016 |
| BCE test | 0.2138 |
| Accuracy test | 0.91 |

Các con số này mô tả thí nghiệm trên dữ liệu giả lập trong notebook; chưa có đối chiếu trực tiếp với scikit-learn trên cùng cấu hình.

**Điều cần hiểu sau bài:** hình dạng tensor/mảng, ý nghĩa của `w` và `b`, vì sao gradient dẫn đến cập nhật tham số, ảnh hưởng của learning rate và sự khác nhau giữa xác suất với nhãn dự đoán.

### 3. Đánh giá sai số ước lượng khoảng cách

Các file: [test_data.py](app/data/test_data.py), [evaluation.py](app/data/evaluation.py), [distance_dataset.csv](app/data/distance_dataset.csv).

Chạy ví dụ tính sai số bằng các phép toán Python cơ bản:

```bash
python -m app.data.test_data
```

Chạy đánh giá bộ dữ liệu mẫu và mở biểu đồ sai số:

```bash
python -m app.data.evaluation
```

Dữ liệu hiện có **5 mẫu**, gồm `car` và `truck`, với ba cột:

| Cột | Ý nghĩa |
| --- | --- |
| `vehicle_type` | Loại phương tiện |
| `pixel_width` | Chiều rộng bounding box theo pixel |
| `ground_truth_distance` | Khoảng cách tham chiếu, đơn vị mét |

Bộ ước lượng sử dụng công thức hình học:

```text
distance_m = real_width_m * focal_length_px / pixel_width
```

Trong bài đánh giá, tiêu cự mặc định là `800` pixel và chiều rộng phương tiện giả định là `1.8` mét. Script tính prediction, sai số có dấu, sai số tuyệt đối, MAE, RMSE và MAPE; thống kê sai số theo loại xe; lọc các mẫu có phần trăm sai số lớn hơn `10%`; cuối cùng vẽ sai số theo khoảng cách thật.

- **MAE:** sai số tuyệt đối trung bình, cùng đơn vị mét với khoảng cách.
- **RMSE:** nhạy hơn với những sai số lớn.
- **MAPE:** sai số tuyệt đối tương đối tính theo phần trăm; cần xử lý riêng nếu khoảng cách tham chiếu bằng 0.

Đây là mô hình hình học với tham số cố định, chưa phải mô hình hồi quy được huấn luyện từ CSV. Việc dùng chung chiều rộng `1.8` mét cho cả xe con và xe tải là một giả định cần kiểm tra khi phân tích lỗi. Bộ dữ liệu nhỏ hiện tại phục vụ việc học cách đánh giá, chưa đủ để kết luận khả năng tổng quát hóa.

### 4. Computer Vision: nhận diện xe và ước lượng khoảng cách

Mã nguồn: [vision_core.py](app/ml_service/vision_core.py) và [vision_orchestration.py](app/ml_service/vision_orchestration.py).

Pipeline hiện tại:

```text
Ảnh / frame video
    → VehicleDetector: YOLOv8 nhận diện phương tiện
    → DistanceEstimator: ước lượng khoảng cách từ chiều rộng bounding box
    → SafetyEvaluator: so sánh với ngưỡng khoảng cách
    → DrawDetections: vẽ bounding box và nhãn
    → Ảnh đã chú thích + danh sách detections
```

Các thiết lập mặc định trong mã:

- Trọng số: `yolov8n.pt`; Ultralytics có thể tải trọng số khi khởi tạo lần đầu nếu máy chưa có file.
- Các class được chọn: `2`, `3`, `5`, `7` — tương ứng car, motorcycle, bus và truck trong hệ nhãn COCO của mô hình mặc định.
- Ngưỡng confidence: `0.5`.
- Tiêu cự giả định: `800` pixel; chiều rộng thực giả định: `1.8` mét.
- Ngưỡng khoảng cách: `55.0` mét; khoảng cách nhỏ hơn ngưỡng được gắn `is_dangerous=True`.

Sau khi cài môi trường, có thể chạy video bằng đoạn lệnh sau. Thay đường dẫn mẫu bằng video trên máy; repo chưa kèm video đầu vào.

```bash
python - <<'PY'
from pathlib import Path
from app.ml_service.vision_orchestration import HighwaySafetyService

video_path = Path("path/to/your/video.mp4")
if not video_path.is_file():
    raise FileNotFoundError(f"Không tìm thấy video: {video_path}")

service = HighwaySafetyService()
service.handle_video(str(video_path))
PY
```

OpenCV mở cửa sổ hiển thị kết quả; nhấn **`q`** để dừng. Cách chạy này cần môi trường có giao diện hiển thị.

Để tích hợp với luồng ảnh khác, gọi `service.process_frame(image)`, trong đó `image` là frame đọc bằng OpenCV. Kết quả trả về là `(image, detections)`; mỗi detection chứa `bbox`, `confidence`, `class_id`, `pixels_width`, `distance_m` và `is_dangerous`.

**Phạm vi hiện tại:** đây là thực hành suy luận với mô hình pretrained, chưa có quy trình tự huấn luyện YOLO. Nhãn `SAFE`/`DANGER` chỉ phản ánh quy tắc ngưỡng cố định; pipeline chưa sử dụng tốc độ, làn đường, hiệu chỉnh camera hay kích thước riêng của từng loại xe để đánh giá tình huống giao thông.

### 5. Học không giám sát cho chiến lược giao dịch

Mã nguồn: [unsupervised.py](app/ml_service/unsupervised.py).

Hướng nghiên cứu được ghi trong script là xây dựng chuỗi xử lý:

1. Lấy danh sách mã S&P 500 và tải dữ liệu giá.
2. Tạo đặc trưng: Garman–Klass volatility, RSI, Bollinger Bands, ATR, MACD và dollar volume.
3. Tổng hợp theo tháng, chọn 150 cổ phiếu thanh khoản cao.
4. Tính lợi suất ở nhiều khoảng thời gian và rolling Fama–French factor betas.
5. Dùng K-Means để nhóm cổ phiếu có đặc trưng tương tự.
6. Chọn tài sản từ cụm và tối ưu danh mục theo mục tiêu max Sharpe.
7. So sánh lợi suất danh mục với benchmark S&P 500.

**Hiện đã có:** lấy danh sách mã từ Wikipedia, tải giá qua yfinance và viết một phần tính toán đặc trưng. Script cố định `end_date = '2023-09-27'` và tính ngày bắt đầu bằng cách lùi `365 * 8` ngày; không tự lấy dữ liệu đến hôm nay. Các bước tổng hợp tháng, K-Means, factor betas, tối ưu danh mục và backtest mới được mô tả trong phần chú thích.

**Các điểm cần xử lý trước khi kỳ vọng chạy trọn vẹn:**

- Hàm `bbands()` trả về tuple gồm ba Series, nhưng các dòng sử dụng lại truy cập `.iloc[:, ...]` như một DataFrame.
- Mã đang giả định dữ liệu tải về có cột `adj close`; cần kiểm tra cấu trúc dữ liệu và cấu hình điều chỉnh giá theo phiên bản yfinance được cài.
- Phần chuẩn hóa ATR/MACD hiện tính mean và standard deviation trên toàn chuỗi của từng mã; cần chuyển sang cách chỉ sử dụng lịch sử khi xây dựng backtest để tránh nhìn trước dữ liệu.
- Danh sách S&P 500 được lấy tại thời điểm chạy; backtest lịch sử cần xem xét thành phần chỉ số theo thời điểm để hạn chế survivorship bias.

Lệnh dưới đây dành cho việc tiếp tục debug và hoàn thiện script; hiện chưa phải một chiến lược chạy hoàn chỉnh:

```bash
python -m app.ml_service.unsupervised
```

Phần này cần kết nối mạng tới nguồn dữ liệu. Repo chưa có kết quả backtest để kết luận về hiệu quả chiến lược.

## Lộ trình học chi tiết

Đây là lộ trình dự kiến để mở rộng lab. Các mục bên dưới là định hướng học, không có nghĩa toàn bộ đã được triển khai.

### Giai đoạn 1 — Nền tảng Python, dữ liệu và toán học

**Nội dung:**

- Python: hàm, class, module, virtual environment và cách đọc lỗi.
- NumPy: shape, axis, broadcasting, vector hóa và phép nhân ma trận.
- pandas: index, groupby, missing values và dữ liệu chuỗi thời gian.
- Đại số tuyến tính: vector, ma trận, tích vô hướng và biểu diễn đặc trưng.
- Giải tích: đạo hàm, gradient và chain rule.
- Xác suất/thống kê: kỳ vọng, phương sai, phân phối và lấy mẫu.
- Tối ưu hóa: loss, learning rate và gradient descent.

**Bài thực hành:** kiểm tra shape ở từng phép toán; vẽ sigmoid; tính loss và gradient bằng tay cho vài mẫu; đối chiếu với NumPy.

**Tiêu chí hoàn thành:** giải thích được `X @ w + b`, tính được một bước cập nhật tham số và hiểu mỗi trục của dữ liệu đại diện cho điều gì.

### Giai đoạn 2 — Machine Learning có giám sát

**Nội dung:**

- Phân biệt regression với classification; feature, label và prediction.
- Chia train/validation/test, xây dựng baseline và tránh data leakage.
- Linear Regression, Logistic Regression và regularization.
- Decision Tree, Random Forest và gradient boosting.
- Scaling, encoding, pipeline và cross-validation.
- Overfitting, underfitting và ảnh hưởng của độ phức tạp mô hình.
- Đánh giá classification bằng confusion matrix, precision, recall, F1, ROC-AUC và PR-AUC; regression bằng MAE/RMSE.

**Bài thực hành:** hoàn thiện notebook nhận diện số 7; so sánh NumPy với scikit-learn trên cùng dữ liệu; thay đổi learning rate, số bước và ngưỡng phân loại; phân tích từng nhóm lỗi.

**Tiêu chí hoàn thành:** chọn metric theo câu hỏi của bài toán, giải thích được kết quả so với baseline và sử dụng validation để ra quyết định trước khi đánh giá test.

### Giai đoạn 3 — Học không giám sát và tài chính định lượng

**Nội dung:**

- Scaling đặc trưng, K-Means, lựa chọn số cụm và diễn giải cụm.
- PCA và trực quan hóa dữ liệu nhiều chiều.
- Chỉ báo kỹ thuật, lợi suất, biến động và đặc trưng theo thời gian.
- Time-based split, walk-forward validation và thời điểm thông tin khả dụng.
- Tái cân bằng, tối ưu danh mục, turnover, phí giao dịch và slippage.
- Đánh giá lợi suất, biến động, Sharpe ratio và maximum drawdown.

**Bài thực hành:** sửa và hoàn thiện `unsupervised.py` theo từng bước; kiểm tra dữ liệu đầu vào trước khi clustering; xây baseline danh mục đơn giản; thêm chi phí và so sánh trên cùng giai đoạn.

**Tiêu chí hoàn thành:** mô tả được tài sản được chọn bằng thông tin nào, ở thời điểm nào; kết quả có baseline và không sử dụng dữ liệu tương lai.

### Giai đoạn 4 — Deep Learning từ nền tảng đến thị giác máy tính

**Nội dung:**

- Tensor, autograd, computational graph và backpropagation.
- MLP, activation, batch, epoch và mini-batch training.
- SGD/Adam, learning rate, initialization và theo dõi loss.
- Regularization, dropout và early stopping.
- CNN, convolution, pooling và biểu diễn ảnh.
- Transfer learning, fine-tuning và data augmentation.
- Object detection: bounding box, confidence, IoU, NMS và mAP.
- Lưu/nạp trọng số; phân biệt chế độ train với inference.

**Bài thực hành:** chọn framework học sâu như PyTorch; huấn luyện MLP trên Digits để so với Logistic Regression; làm bài CNN; mở rộng bài YOLO bằng phân tích bỏ sót, nhận diện nhầm và sai số khoảng cách theo loại xe.

**Tiêu chí hoàn thành:** tự viết được vòng lặp train/validation, đọc learning curve và giải thích sự khác nhau giữa dùng trọng số pretrained với huấn luyện/fine-tune mô hình. Phụ thuộc cho các bài mới sẽ được bổ sung khi triển khai.

### Giai đoạn 5 — Reinforcement Learning cơ bản

**Nội dung:**

- Agent, environment, state/observation, action, reward và episode.
- Markov Decision Process, discount factor và discounted return.
- Policy, state value, action value và phương trình Bellman.
- Exploration/exploitation và epsilon-greedy.
- Dynamic Programming, Monte Carlo và Temporal Difference learning.
- Q-Learning, SARSA và đánh giá policy.

**Bài thực hành dự kiến:** bắt đầu bằng GridWorld hoặc môi trường bảng nhỏ; tự cài Q-table; ghi reward và số bước mỗi episode; so sánh policy học được với random policy trên nhiều seed.

**Tiêu chí hoàn thành:** giải thích được agent tối ưu tổng reward dài hạn ra sao, cập nhật một giá trị Q bằng tay và đánh giá policy riêng với quá trình học.

### Giai đoạn 6 — Deep Reinforcement Learning và ứng dụng

**Nội dung:**

- DQN, replay buffer, target network và các vấn đề ổn định huấn luyện.
- Policy Gradient, Actor–Critic và PPO.
- Reward design, reward scaling và kiểm tra agent có khai thác sai mục tiêu hay không.
- Đánh giá nhiều seed, độ biến thiên kết quả và lưu checkpoint.
- Thiết kế môi trường giao dịch với observation, action, reward và chi phí rõ ràng.

**Bài thực hành dự kiến:** triển khai DQN trên môi trường nhỏ như CartPole; so sánh với baseline; sau đó mới thử môi trường giao dịch có giới hạn vị thế, phí và dữ liệu chia theo thời gian.

**Tiêu chí hoàn thành:** policy được đánh giá ngoài các episode huấn luyện, kết quả có đối chứng và phân tích được ảnh hưởng của reward, seed cùng chi phí. Các môi trường và thư viện RL hiện chưa được thêm vào repo.

## Quy trình thực hiện một bài lab

Mỗi bài nên có ghi chú theo mẫu:

```markdown
# Tên bài lab

## Câu hỏi và mục tiêu
- Muốn kiểm tra hoặc hiểu điều gì?
- Baseline nào dùng để so sánh?

## Dữ liệu
- Nguồn, đặc trưng, nhãn, số lượng mẫu và đơn vị.
- Cách chia train/validation/test và các giả định.

## Phương pháp
- Công thức, mô hình, loss và metric.
- Seed, siêu tham số và cách chạy.

## Kết quả
- Bảng metric, biểu đồ và cấu hình tương ứng.
- So sánh với baseline.

## Phân tích lỗi
- Mẫu/nhóm nào dự đoán chưa tốt? Bằng chứng là gì?
- Những giả định hoặc giới hạn nào có thể ảnh hưởng kết quả?

## Điều học được và bước tiếp theo
- Có thể giải thích lại điều gì sau bài này?
- Thử nghiệm tiếp theo sẽ kiểm tra giả thuyết nào?
```

Ưu tiên mỗi thử nghiệm trả lời một câu hỏi rõ ràng. Ví dụ: “tăng recall của lớp số 7 khi hạ ngưỡng có làm precision giảm bao nhiêu?” hoặc “sai số khoảng cách của xe tải khác xe con thế nào khi dùng chung chiều rộng giả định?”.

## Nguyên tắc đánh giá và tái lập

- **Giữ test cho đánh giá cuối:** chọn mô hình và siêu tham số bằng train/validation; không tối ưu liên tục dựa trên test.
- **Fit preprocessing trên train:** khi dùng scaling, imputation hoặc lựa chọn đặc trưng có học từ dữ liệu, chỉ fit trên train rồi áp dụng cho tập còn lại.
- **Tôn trọng thứ tự thời gian:** bài tài chính cần cách chia theo thời gian, xử lý thời điểm tín hiệu và thời điểm giao dịch rõ ràng.
- **Luôn có baseline:** mô hình đơn giản giúp biết phần phức tạp thêm có thực sự cải thiện kết quả.
- **Đọc lỗi theo nhóm:** xem loại xe, khoảng cách, lớp dữ liệu hoặc điều kiện thị trường thay vì chỉ xem trung bình chung.
- **Ghi cấu hình:** lưu seed, siêu tham số, phiên bản thư viện, dữ liệu và cách chạy cùng kết quả. Seed `42` đang được dùng trong hai notebook, chưa phải cấu hình chung toàn repo.
- **Phân biệt output đã lưu và kết quả mới:** khi thay đổi notebook, khởi động lại kernel và chạy toàn bộ để tránh phụ thuộc vào trạng thái cell cũ.

## Các bước tiếp theo

- [ ] Hoàn thiện baseline, confusion matrix và lựa chọn ngưỡng trong notebook Digits.
- [ ] Đối chiếu Logistic Regression NumPy với scikit-learn trên cùng dữ liệu và làm rõ khác biệt regularization.
- [ ] Vẽ loss theo số bước, thử learning rate và kiểm tra gradient bằng sai phân hữu hạn.
- [ ] Thu thập thêm dữ liệu khoảng cách, ghi cách đo và hiệu chỉnh camera/kích thước phương tiện.
- [ ] Đánh giá riêng chất lượng detection và chất lượng ước lượng khoảng cách.
- [ ] Sửa xử lý Bollinger Bands, kiểm tra dữ liệu giá và hoàn thiện luồng học không giám sát.
- [ ] Thiết lập backtest theo thời gian với baseline và chi phí giao dịch.
- [ ] Bổ sung bài MLP/CNN tự huấn luyện cùng ghi chép thí nghiệm.
- [ ] Bắt đầu RL bằng môi trường nhỏ và Q-Learning trước khi học Deep RL.
- [ ] Khai báo đầy đủ phụ thuộc trực tiếp và khóa phiên bản khi có cấu hình chạy ổn định.

Lab sẽ được cập nhật theo quá trình học: mỗi phần mới bổ sung mã chạy được, kết quả quan sát và lời giải thích về điều đã hiểu cũng như những câu hỏi còn mở.
