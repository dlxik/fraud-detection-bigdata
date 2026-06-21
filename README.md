# Fraud Detection System using Big Data Analytics and Machine Learning

Project này xây dựng một pipeline phát hiện giao dịch gian lận tài chính từ dữ liệu giao dịch thẻ. Hệ thống kết hợp phân tích dữ liệu, tiền xử lý, xử lý dữ liệu lớn bằng Apache Spark và huấn luyện mô hình Machine Learning để phân loại giao dịch hợp lệ/gian lận.

## Mục Tiêu

- Phân tích đặc điểm của dữ liệu giao dịch tài chính.
- Làm rõ vấn đề mất cân bằng dữ liệu trong bài toán fraud detection.
- Tiền xử lý dữ liệu và xây dựng đặc trưng phục vụ mô hình học máy.
- Sử dụng Apache Spark/PySpark cho các tác vụ xử lý và tổng hợp dữ liệu lớn.
- Huấn luyện, đánh giá và so sánh các mô hình Logistic Regression, Random Forest và HistGradientBoosting.
- Đánh giá mô hình bằng các độ đo phù hợp với dữ liệu mất cân bằng như Precision, Recall, F1-score, ROC-AUC và PR-AUC.

## Dataset

Nguồn dữ liệu: [Kaggle - Credit Card Transactions Fraud Detection](https://www.kaggle.com/datasets/kartik2112/fraud-detection)

Các file sử dụng:

- `data/raw/fraudTrain.csv`: tập huấn luyện
- `data/raw/fraudTest.csv`: tập kiểm thử

Biến mục tiêu:

- `is_fraud = 0`: giao dịch hợp lệ
- `is_fraud = 1`: giao dịch gian lận

Thống kê chính:

- Train: `1,296,675` giao dịch
- Test: `555,719` giao dịch
- Fraud trong train: `7,506`
- Fraud rate trong train: khoảng `0.5789%`

Dữ liệu bị mất cân bằng rất mạnh, do đó Accuracy không đủ để đánh giá mô hình. Project ưu tiên các metric như Recall, F1-score và PR-AUC.

## Big Data 5V

Dataset giao dịch tài chính phù hợp với bối cảnh Big Data theo mô hình 5V:

- **Volume:** hơn 1.85 triệu giao dịch trên cả train và test.
- **Velocity:** trong thực tế, giao dịch tài chính phát sinh liên tục và cần phát hiện rủi ro nhanh.
- **Variety:** dữ liệu gồm số tiền, thời gian, loại giao dịch, merchant, thông tin khách hàng và vị trí địa lý.
- **Veracity:** cần kiểm tra, làm sạch, xử lý kiểu dữ liệu và loại bỏ các thuộc tính định danh trực tiếp.
- **Value:** dữ liệu có thể hỗ trợ phát hiện giao dịch rủi ro, giảm thiệt hại tài chính và nâng cao an toàn hệ thống thanh toán.

## Cấu Trúc Project

```text
fraud-detection-bigdata/
  data/
    raw/
      fraudTrain.csv
      fraudTest.csv
    processed/
      fraud_train_processed.csv
      fraud_test_processed.csv

  models/
    logistic_regression.joblib
    random_forest.joblib
    hist_gradient_boosting.joblib

  notebooks/
    01_EDA.ipynb
    02_Preprocessing.ipynb
    03_Spark_Processing.ipynb
    04_Modeling.ipynb

  reports/
    figures/
    tables/
    report.pdf

  src/
    preprocessing.py
    spark_processing.py
    evaluate.py
    train.py
    generate_artifacts.py

  requirements.txt
  README.md
```

## Pipeline

Pipeline tổng thể:

```text
Raw Data
  -> Exploratory Data Analysis
  -> Preprocessing / Feature Engineering
  -> Spark Processing
  -> Model Training
  -> Evaluation
  -> Report / Figures / Tables / Model Artifacts
```

Các bước chính:

1. **EDA:** phân tích target distribution, amount, category, merchant, thời gian, vị trí địa lý và tương quan.
2. **Preprocessing:** tạo feature thời gian, tuổi khách hàng, khoảng cách khách hàng-merchant, encode categorical features và scale numerical features.
3. **Spark Processing:** dùng Spark DataFrame để tổng hợp fraud theo category, state, merchant, hour và so sánh Pandas/Spark.
4. **Modeling:** huấn luyện Logistic Regression, Random Forest và HistGradientBoosting.
5. **Evaluation:** đánh giá bằng Accuracy, Precision, Recall, F1-score, ROC-AUC, PR-AUC, confusion matrix, ROC curve và Precision-Recall curve.

## Tiền Xử Lý Và Feature Engineering

Script chính:

```text
src/preprocessing.py
```

Các bước xử lý:

- Parse `trans_date_trans_time`.
- Tạo feature thời gian:
  - `transaction_hour`
  - `transaction_dayofweek`
  - `transaction_month`
  - `weekend_flag`
- Tạo `customer_age` từ `dob`.
- Tạo `customer_merchant_distance_km` bằng công thức Haversine.
- One-hot encode:
  - `category`
  - `gender`
  - `state`
- Frequency encode:
  - `merchant`
  - `job`
  - `city`
- Scale các feature số.
- Loại bỏ các cột định danh trực tiếp như `cc_num`, `first`, `last`, `street`, `trans_num`.

Output:

- `data/processed/fraud_train_processed.csv`
- `data/processed/fraud_test_processed.csv`
- `reports/tables/t2.1_preprocessing_report.json`
- `reports/tables/t2.2_processed_feature_summary.csv`
- `reports/tables/t2.3_frequency_encoding_summary.csv`
- `reports/tables/t2.4_scaler_summary.csv`

Sau preprocessing:

- Train: `1,296,675` dòng, `85` cột
- Test: `555,719` dòng, `85` cột

## Apache Spark Processing

Script chính:

```text
src/spark_processing.py
```

Spark được sử dụng để xử lý và tổng hợp dữ liệu lớn, không trực tiếp train mô hình chính.

Các bảng phân tích bằng Spark:

- `t3.1_dataset_summary.csv`: tổng quan dataset
- `t3.2_fraud_by_category.csv`: fraud theo loại giao dịch
- `t3.3_fraud_by_state.csv`: fraud theo bang
- `t3.4_high_risk_merchants.csv`: merchant có rủi ro cao
- `t3.5_fraud_by_hour.csv`: fraud theo giờ giao dịch
- `t3.6_amount_by_target.csv`: thống kê amount theo target
- `t3.7_pandas_comparison.csv`: so sánh Pandas và Spark
- `t3.8_processing_report.json`: manifest output Spark

Ý nghĩa của Spark trong project:

- Thể hiện khả năng xử lý dữ liệu lớn.
- Thực hiện aggregation theo nhiều chiều.
- Tạo insight hỗ trợ EDA và báo cáo.
- Có thể mở rộng lên cluster/cloud khi dữ liệu tăng kích thước.

## Modeling

Script chính:

```text
src/train.py
```

Các mô hình sử dụng:

- **Logistic Regression:** baseline tuyến tính.
- **Random Forest:** ensemble nhiều cây quyết định, bắt quan hệ phi tuyến tốt hơn.
- **HistGradientBoosting:** boosting model dựa trên cây quyết định, là mô hình tốt nhất trong project.

Chiến lược xử lý mất cân bằng:

- Giữ toàn bộ giao dịch fraud trong train.
- Lấy mẫu non-fraud theo tỷ lệ `5:1` so với fraud.
- Train trên sampled train set.
- Đánh giá cuối cùng trên toàn bộ test set.

Threshold tuning:

- Mô hình sinh xác suất fraud.
- Threshold mặc định `0.50` chưa chắc tối ưu với dữ liệu mất cân bằng.
- Project tune threshold theo F1-score trên validation split.
- HistGradientBoosting tuned F1 chọn threshold `0.80`.

## Kết Quả Chính

Kết quả trên test set:

| Model | Threshold | Accuracy | Precision | Recall | F1-score | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| HistGradientBoosting tuned F1 | 0.80 | 0.9932 | 0.3557 | 0.9245 | 0.5137 | 0.9976 | 0.8701 |
| HistGradientBoosting default | 0.50 | 0.9843 | 0.1938 | 0.9669 | 0.3229 | 0.9976 | 0.8701 |
| Random Forest default/tuned | 0.50 | 0.9836 | 0.1802 | 0.9147 | 0.3011 | 0.9899 | 0.7441 |
| Logistic Regression tuned F1 | 0.71 | 0.9849 | 0.1577 | 0.6699 | 0.2553 | 0.9055 | 0.1189 |

Nhận xét:

- HistGradientBoosting là mô hình tốt nhất.
- Recall đạt `0.9245`, tức phát hiện khoảng `92.45%` giao dịch gian lận trong test set.
- F1-score đạt `0.5137`, cao hơn rõ so với Random Forest (`0.3011`).
- PR-AUC đạt `0.8701`, rất quan trọng vì fraud là lớp thiểu số.
- Precision chưa quá cao do tỷ lệ fraud gốc rất thấp, nhưng vẫn cao hơn rất nhiều so với fraud baseline `0.5789%`.

## Quy Ước Tên Artifact

Các hình và bảng trong `reports/figures` và `reports/tables` được đặt tên theo nguồn sinh ra artifact:

- `f1.x_...`: hình sinh từ `notebooks/01_EDA.ipynb`
- `t1.x_...`: bảng sinh từ `notebooks/01_EDA.ipynb`
- `t2.x_...`: bảng sinh từ preprocessing
- `t3.x_...`: bảng sinh từ Spark processing
- `f4.x_...`: hình sinh từ modeling/evaluation
- `t4.x_...`: bảng sinh từ modeling/evaluation

Ví dụ:

```text
reports/figures/f1.1_target_distribution.png
reports/figures/f4.1_confusion_matrix_hist_gradient_boosting_tuned_f1.png
reports/figures/f4.2_roc_curve_models.png
reports/tables/t3.2_fraud_by_category.csv
reports/tables/t4.1_model_metrics.csv
```

## Cách Cài Đặt

Cài dependencies:

```bash
pip install -r requirements.txt
```

Các thư viện chính:

- pandas
- numpy
- matplotlib
- seaborn
- pyspark
- scikit-learn
- reportlab
- python-pptx

## Cách Chạy Pipeline

Chạy từ thư mục gốc project.

### 1. Preprocessing

```bash
python src/preprocessing.py
```

Output chính:

```text
data/processed/fraud_train_processed.csv
data/processed/fraud_test_processed.csv
reports/tables/t2.*
```

### 2. Spark Processing

```bash
python src/spark_processing.py
```

Output chính:

```text
reports/tables/t3.*
```

Lưu ý: trên Windows có thể xuất hiện warning liên quan `winutils.exe` hoặc Hadoop native library. Các warning này không nhất thiết làm pipeline thất bại nếu script vẫn chạy xong và sinh bảng output.

### 3. Training Và Evaluation

```bash
python src/train.py --negative-ratio 5
```

Output chính:

```text
models/logistic_regression.joblib
models/random_forest.joblib
models/hist_gradient_boosting.joblib
reports/tables/t4.*
reports/figures/f4.*
```

### 4. Generate Report/Slide Artifacts

```bash
python src/generate_artifacts.py
```

Output:

```text
reports/final_report.pdf
reports/fraud_detection_presentation.pptx
```

Nếu dùng bản LaTeX riêng, file report cuối có thể nằm tại:

```text
reports/report.pdf
```

## Demo Pipeline

Một demo ngắn có thể chạy theo thứ tự:

```bash
python src/preprocessing.py
python src/spark_processing.py
python src/train.py --negative-ratio 5
python src/generate_artifacts.py
```

Sau đó kiểm tra:

```text
reports/tables/t4.1_model_metrics.csv
reports/figures/f4.1_confusion_matrix_hist_gradient_boosting_tuned_f1.png
reports/figures/f4.2_roc_curve_models.png
reports/figures/f4.3_precision_recall_curve_models.png
models/hist_gradient_boosting.joblib
```

Có thể xem nhanh bảng metric bằng:

```bash
python -c "import pandas as pd; cols=['model','precision','recall','f1_score','roc_auc','pr_auc','threshold']; print(pd.read_csv('reports/tables/t4.1_model_metrics.csv')[cols].head(5).to_string(index=False))"
```

## Output Quan Trọng

### Tables

- `reports/tables/t1.1_target_distribution.csv`
- `reports/tables/t2.1_preprocessing_report.json`
- `reports/tables/t3.1_dataset_summary.csv`
- `reports/tables/t3.2_fraud_by_category.csv`
- `reports/tables/t3.4_high_risk_merchants.csv`
- `reports/tables/t4.1_model_metrics.csv`
- `reports/tables/t4.2_classification_reports_all_models.csv`
- `reports/tables/t4.3_threshold_tuning_all_models.csv`

### Figures

- `reports/figures/f1.1_target_distribution.png`
- `reports/figures/f1.2_amount_distribution_log_scale.png`
- `reports/figures/f1.4_category_transaction_count_and_fraud_rate.png`
- `reports/figures/f1.6_fraud_rate_by_hour.png`
- `reports/figures/f4.1_confusion_matrix_hist_gradient_boosting_tuned_f1.png`
- `reports/figures/f4.2_roc_curve_models.png`
- `reports/figures/f4.3_precision_recall_curve_models.png`
- `reports/figures/f4.4_model_metrics_comparison.png`

### Models

- `models/logistic_regression.joblib`
- `models/random_forest.joblib`
- `models/hist_gradient_boosting.joblib`

## Hạn Chế Và Hướng Phát Triển

Hạn chế:

- Project mới xử lý theo lô trên dữ liệu có sẵn, chưa triển khai real-time fraud detection.
- Spark được dùng cho phân tích/aggregation, chưa tích hợp trực tiếp vào toàn bộ scoring pipeline.
- Chưa thử các mô hình mạnh hơn như XGBoost, LightGBM hoặc CatBoost.
- Precision của model tốt nhất chưa quá cao do fraud là lớp rất hiếm.
- Chưa có explainability như SHAP để giải thích vì sao giao dịch bị gắn nhãn fraud.

Hướng phát triển:

- Thử XGBoost, LightGBM, CatBoost.
- Hyperparameter tuning sâu hơn.
- Cost-sensitive learning theo chi phí false positive/false negative.
- Batch scoring pipeline bằng Spark.
- Real-time streaming bằng Spark Streaming hoặc Kafka.
- Dashboard theo dõi fraud và cảnh báo.
- Explainability bằng SHAP.

## Tài Liệu Tham Khảo

- Kaggle Credit Card Transactions Fraud Detection Dataset: https://www.kaggle.com/datasets/kartik2112/fraud-detection
- Apache Spark Documentation: https://spark.apache.org/docs/latest/
- scikit-learn Documentation: https://scikit-learn.org/stable/

