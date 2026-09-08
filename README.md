# 🧠 Machine Learning Mastery & Production Pipeline

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Kho tài liệu, mã nguồn thực hành và pipeline học máy chuẩn mực phục vụ nghiên cứu, đào tạo và triển khai mô hình học máy vào thực tế. Dự án kết hợp giữa **Pipeline phân loại nhị phân thực chiến (Dự đoán tiểu đường)** và **Toàn bộ kho tài liệu 18 bài học AI & Data Science nâng cao**.

---

## 📑 Mục Lục

1. [Tổng Quan Dự Án](#-tổng-quan-dự-án)
2. [Cấu Trúc Thư Mục](#-cấu-trúc-thư-mục)
3. [Lab Thực Hành Trực Quan Động (Ex1)](#-lab-thực-hành-trực-quan-động-ex1)
4. [Pipeline Dự Đoán Tiểu Đường (End-to-End)](#-pipeline-dự-đoán-tiểu-đường-end-to-end)
5. [Chương Trình AI/ML Nâng Cao (`Learning_AI_VietNguyen`)](#-chương-trình-aiml-nâng-cao-learning_ai_vietnguyen)
6. [Hướng Dẫn Cài Đặt & Chạy](#-hướng-dẫn-cài-đặt--chạy)
7. [Lưu Ý Về Tài Nguyên Dung Lượng Lớn](#-lưu-ý-về-tài-nguyên-dung-lượng-lớn)

---

## 🎯 Tổng Quan Dự Án

Dự án gồm 3 phần cốt lõi:

1. **Lab Thực Hành Trực Quan Động (Ex1 - Foundations Lab)**:
   - Hệ thống trực quan hoá thuật toán và nền tảng lập trình cho AI/ML với giao diện Web tương tác (SVG graphs, BFS pathfinding, 1D Max Pooling, ReLU visualizer, Decision Tree explorer).
   - 16 bài tập tự làm thực chiến kèm trình chấm điểm tự động CLI và bộ kiểm thử toàn diện 43 test cases.

2. **Pipeline Chuẩn Mực Trong Machine Learning**:
   - Giải quyết bài toán phân loại nhị phân trên bộ dữ liệu `diabetes.csv`.
   - Áp dụng triệt để các kỹ thuật: Phân tích khám phá dữ liệu (EDA), xử lý giá trị khuyết thiếu phi thực tế (Missing values), nhận diện ngoại lai (Outliers with IQR), phân tầng dữ liệu tránh rò rỉ (Stratified Split, No Data Leakage), chuẩn hoá (Scaling), so sánh đa thuật toán, tinh chỉnh siêu tham số (Hyperparameter Tuning), đánh giá toàn diện (ROC-AUC, Precision-Recall Curve, Tối ưu ngưỡng quyết định) và lưu trữ Artifacts (`.pkl`).

3. **Khoá Học Toàn Tập 18 Bài Học AI & Machine Learning**:
   - Hệ thống giáo trình chi tiết, slide bài giảng HTML tương tác kèm công thức KaTeX.
   - Hướng dẫn từ các thuật toán cổ điển (Linear, Logistic, Trees, SVM, Ensemble, KNN, K-Means) đến các bài toán nâng cao: Xử lý ngôn ngữ tự nhiên (NLP / TF-IDF), Dự báo chuỗi thời gian (Time-series Forecasting), Hệ thống gợi ý (Recommendation System), Giảm chiều dữ liệu (PCA) và Đóng gói triển khai với FastAPI & Streamlit.

---

## 📁 Cấu Trúc Thư Mục

```
Learning_Machine/
├── Ex1/                        # Lab thực hành trực quan & tương tác động (Ex1 -> Ex8)
│   ├── interactive_hub.html   # Giao diện Web tương tác động kèm Code Arena trực tiếp
│   ├── launch_hub.py          # Script 1-click khởi chạy local server và mở trình duyệt
│   ├── practice_exercises.py  # 16 bài tập tự làm thực chiến kèm trình chấm điểm
│   ├── solutions_practice.py  # Lời giải mẫu chuẩn & phân tích độ phức tạp O(N)
│   ├── test_all.py            # Bộ test tự động 43 test cases (pytest)
│   ├── README.md              # Hướng dẫn chi tiết lab Ex1
│   └── Ex1.py ... Ex8.py      # Mã nguồn nền tảng được module hoá & type hints
├── EDA.py                     # Khám phá dữ liệu, phân tích thống kê & phân phối
├── b2&3.py                    # Xử lý giá trị 0 không hợp lệ & lọc Outliers theo IQR
├── b4_split.py                # Chia tập dữ liệu Train/Test (Stratified, chống Data Leakage)
├── b5_feature.py              # Chuẩn hoá dữ liệu với StandardScaler
├── traind.py                  # Huấn luyện và so sánh sơ bộ 5 mô hình ML
├── b8_tune.py                 # Tinh chỉnh siêu tham số RandomForest qua GridSearchCV
├── b9_evaluate.py             # Đánh giá toàn diện, vẽ ROC curve, PR curve, chọn Threshold
├── b10_save.py                # Huấn luyện mô hình tốt nhất và đóng gói file pickle
├── huong_dan_cac_buoc.md      # Hướng dẫn chi tiết từng bước bằng tiếng Việt
├── diabetes.csv               # Dataset gốc về bệnh tiểu đường (Pima Indians)
├── diabetes_cleaned.csv       # Dataset sau khi đã tiền xử lý
├── outlier_explained.html     # Báo cáo trực quan minh hoạ bản chất của Outliers
├── model.pkl                  # Mô hình Random Forest đã được tối ưu
├── scaler.pkl                 # Transformer chuẩn hoá StandardScaler
├── threshold.pkl              # Ngưỡng quyết định tối ưu đã được tính toán
├── requirements.txt           # Danh sách các thư viện phụ thuộc
├── .gitignore                 # Cấu hình loại trừ file rác & file dung lượng lớn
│
└── Learning_AI_VietNguyen/    # Kho tài liệu & bài tập thực chiến 18 bài học
    ├── GIAO_TRINH_CHI_TIET_TOAN_BO_KHOA_HOC.md  # Giáo trình toàn tập 18 chuyên đề
    ├── AI_ML_Mastery_Slides.html               # Slide bài giảng tương tác (HTML + KaTeX)
    ├── Datasets/                               # Các tập dữ liệu mẫu (MNIST, CSGO, Movies,...)
    ├── Exercises/                              # Các bài tập thực hành theo tuần
    ├── Scripts/                                # Mã nguồn thuật toán & Triển khai ứng dụng
    │   ├── classification.py                   # Pipeline phân loại mẫu
    │   ├── regression.py                       # Pipeline hồi quy mẫu
    │   ├── job_classification.py               # Bài toán NLP phân loại tin tuyển dụng
    │   ├── time_series_forecasting.py          # Dự báo chuỗi thời gian (Recursive)
    │   ├── time_series_direct_forecasting.py   # Dự báo chuỗi thời gian (Direct Multi-step)
    │   ├── server.py                           # Backend REST API bằng FastAPI
    │   ├── client.py                           # Client kiểm thử API dự đoán
    │   └── inference.py                        # Script suy luận mô hình độc lập
    ├── Transcripts/                            # Toàn bộ phụ đề/transcript các bài giảng
    └── recommendation_system.py                # Thuật toán gợi ý phim (Content-based)
```

---

## 🎮 Lab Thực Hành Trực Quan Động (Ex1)

Phân hệ `Ex1/` cung cấp một hệ thống học tập trực quan tương tác toàn diện, kết nối các bài toán lập trình nền tảng với các khái niệm cốt lõi trong Machine Learning & Deep Learning:

### 1. Khởi Chạy Giao Diện Trực Quan (Interactive Hub)
- **Cách 1:** Mở terminal trong thư mục `Ex1/` và chạy:
  ```powershell
  py launch_hub.py
  ```
  *(Trình duyệt sẽ tự động mở `http://localhost:8080` với đầy đủ Code Arena và bộ chấm điểm thời gian thực)*
- **Cách 2:** Mở trực tiếp file `Ex1/interactive_hub.html` trên bất kỳ trình duyệt nào (hoạt động 100% offline).

### 2. Bộ Công Cụ Tương Tác & Machine Learning Connections
- **Ex1 (Số Âm/Dương):** Thùng phân loại động + Hàm kích hoạt **ReLU** ($f(x) = \max(0, x)$) & đo tỷ lệ *Dead Neurons*.
- **Ex2 (Tần Suất K):** Biểu đồ tần suất SVG động + Kỹ thuật **Vocabulary Pruning** trong NLP.
- **Ex3 (Cặp Liền Kề):** Khung trượt trực quan động (Sliding Window) mô phỏng lớp **1D Max Pooling** trong CNN.
- **Ex4 (Hoán Vị):** Cây quyết định phân nhánh 3 tầng + Thuật toán **GridSearchCV** tìm kiếm siêu tham số.
- **Ex5 (Khử Trùng Lặp):** Băng chuyền Hash Set $O(1)$ giữ thứ tự + Gộp bản ghi **Feature Store**.
- **Ex6 (Ma Trận Chia Hết):** Lưới lọc số đổi màu động + Thuật toán chia tập **K-Fold Cross Validation**.
- **Ex7 (Số Chẵn & Parity):** Bộ phân tích số học thời gian thực, tính **Hamming Weight** & Parity Bit trong mã hoá nhị phân.
- **Ex8 (Word Chain):** Đồ thị mạng lưới SVG, trực quan hoá thuật toán **BFS Shortest Path**, Queue & Visited state.

### 3. Tự Làm 16 Bài Tập Thực Chiến (`practice_exercises.py`)
```powershell
# Chấm điểm toàn bộ 16 bài tập:
py Ex1/practice_exercises.py

# Chạy bộ kiểm thử tự động toàn diện (43 test cases):
py -m pytest Ex1/test_all.py -v
```

---

## 🔬 Pipeline Dự Đoán Tiểu Đường (End-to-End)

Quy trình được chuẩn hoá thành các module rõ ràng:

| Bước | Script | Nhiệm vụ chính |
|:---|:---|:---|
| **1. EDA** | `EDA.py` | Kiểm tra kích thước, kiểu dữ liệu, tỷ lệ mất cân bằng lớp và biểu đồ phân phối. |
| **2. Làm sạch** | `b2&3.py` | Loại bỏ các mẫu có giá trị bằng 0 vô lý ở Glucose, BloodPressure, SkinThickness, Insulin, BMI; cắt lọc ngoại lai theo IQR. |
| **3. Phân chia** | `b4_split.py` | Phân chia 80% Train, 20% Test theo tỷ lệ cân bằng lớp (`stratify=y`). |
| **4. Feature Scaling** | `b5_feature.py` | `StandardScaler` vừa vặn trên Train và áp dụng trên Test để tránh rò rỉ dữ liệu. |
| **5. Huấn luyện** | `traind.py` | So sánh nhanh Logistic Regression, KNN, Decision Tree, Random Forest, SVM. |
| **6. Tối ưu** | `b8_tune.py` | Dùng `GridSearchCV` tìm tổ hợp siêu tham số tối ưu cho Random Forest. |
| **7. Đánh giá** | `b9_evaluate.py` | Tính toán Precision, Recall, F1-Score, ROC-AUC; xác định ngưỡng tối ưu cân bằng Precision/Recall. |
| **8. Đóng gói** | `b10_save.py` | Lưu `model.pkl`, `scaler.pkl`, `threshold.pkl` sẵn sàng cho suy luận. |

---

## 📚 Chương Trình AI/ML Nâng Cao (`Learning_AI_VietNguyen`)

Giáo trình toàn diện 18 bài học được biên soạn công phu tại `Learning_AI_VietNguyen/GIAO_TRINH_CHI_TIET_TOAN_BO_KHOA_HOC.md`:

- **Lesson 01 - 03:** Nhập môn AI/ML, Phân loại dữ liệu, Loss Functions & Evaluation Metrics.
- **Lesson 04 - 07:** Data Preprocessing, Feature Engineering, Linear/Logistic Regression, Decision Trees, SVM, KNN, K-Means & Random Forest.
- **Lesson 08 - 10:** Xây dựng Classification & Regression Pipeline hoàn chỉnh, xử lý imbalanced data (SMOTE), siêu tham số với GridSearchCV.
- **Lesson 11 - 12:** Xử lý ngôn ngữ tự nhiên chuyên sâu (NLP, Text Preprocessing, TF-IDF, Job Classification Project).
- **Lesson 13 - 14:** Bản chất dự báo chuỗi thời gian (Time-series Forecasting: Recursive vs Direct Multi-step).
- **Lesson 15 - 16:** Giảm chiều dữ liệu (Curse of Dimensionality, PCA) & Hệ thống gợi ý phim (Content-based Filtering).
- **Lesson 17 - 18:** Triển khai ứng dụng sản phẩm (REST API với FastAPI, Giao diện tương tác với Streamlit, Đóng gói Docker), Đánh giá Trade-off Bias/Variance & Lộ trình nâng cao.

Slide bài giảng tương tác mở trực tiếp không cần mạng:
```bash
open Learning_AI_VietNguyen/AI_ML_Mastery_Slides.html
```

---

## 🚀 Hướng Dẫn Cài Đặt & Chạy

### 1. Chuẩn Bị Môi Trường

```bash
# Clone repository
git clone https://github.com/DuongNAD/Learning_Machine.git
cd Learning_Machine

# Tạo môi trường ảo và kích hoạt
python3 -m venv .venv
source .venv/bin/activate   # Trên Linux/macOS
# .venv\Scripts\activate     # Trên Windows

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

### 2. Chạy Pipeline Học Máy

```bash
# Bước 1: Khám phá dữ liệu
python EDA.py

# Bước 2 & 3: Làm sạch và xử lý ngoại lai
python b2\&3.py

# Bước 4 & 5 & Train thử nghiệm
python traind.py

# Bước 6: Tinh chỉnh siêu tham số
python b8_tune.py

# Bước 7: Đánh giá chi tiết mô hình
python b9_evaluate.py

# Bước 8: Lưu mô hình phục vụ suy luận
python b10_save.py
```

### 3. Chạy Thử Nghiệm API Suy Luận (FastAPI)

```bash
cd Learning_AI_VietNguyen/Scripts
uvicorn server:app --reload --port 8000
```
Ở một terminal khác, thực thi client để gửi yêu cầu dự đoán:
```bash
python client.py
```

---

## ⚠️ Lưu Ý Về Tài Nguyên Dung Lượng Lớn

Để tuân thủ chính sách kích thước tệp của GitHub (< 100MB cho mỗi tệp), các tệp tài nguyên kích thước lớn đã được cấu hình trong `.gitignore`:
- **18 Video bài giảng MP4 (`Learning_AI_VietNguyen/Videos/`)**: Tổng dung lượng > 5GB được lưu trữ tại môi trường cục bộ.
- **Tệp slide bài giảng PDF (`Slide AIDSML course_new_version.pdf`)**: Dung lượng 657MB được thay thế hoàn hảo bằng bản trình chiếu tương tác trực tuyến `AI_ML_Mastery_Slides.html`.

---

## 👤 Tác Giả

- **Dương Nguyễn** — [@DuongNAD](https://github.com/DuongNAD)
