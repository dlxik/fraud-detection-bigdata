# Fraud Detection System using Big Data Analytics

Hệ thống phát hiện giao dịch gian lận từ dữ liệu giao dịch tài chính, sử dụng Pandas, Apache Spark/PySpark và Machine Learning.

## Dataset

Nguồn dữ liệu: https://www.kaggle.com/datasets/kartik2112/fraud-detection

File đã dùng:

- `data/raw/fraudTrain.csv`
- `data/raw/fraudTest.csv`

Biến mục tiêu:

- `is_fraud = 0`: giao dịch hợp lệ
- `is_fraud = 1`: giao dịch gian lận

## Cấu trúc dự án

```text
data/
  raw/
  processed/
models/
notebooks/
  01_EDA.ipynb
  02_Preprocessing.ipynb
  03_Spark_Processing.ipynb
  04_Modeling.ipynb
reports/
  figures/
  tables/
src/
  preprocessing.py
  spark_processing.py
  evaluate.py
  train.py
requirements.txt
```

## Pipeline

1. EDA: phân tích phân bố target, amount, category, merchant, thời gian, địa lý.
2. Preprocessing: tạo feature thời gian, tuổi khách hàng, khoảng cách khách hàng-merchant, encode categorical, scale numerical.
3. Spark Processing: xử lý dữ liệu bằng Spark DataFrame, fraud theo category/state/merchant/hour, so sánh Pandas và Spark.
4. Modeling: huấn luyện Logistic Regression, Random Forest và HistGradientBoosting.
5. Evaluation: Accuracy, Precision, Recall, F1-score, ROC-AUC, PR-AUC, confusion matrix, ROC curve, precision-recall curve.

## Cách chạy

Cài thư viện:

```bash
pip install -r requirements.txt
```

Chạy preprocessing:

```bash
python src/preprocessing.py
```

Chạy Spark processing:

```bash
python src/spark_processing.py
```

Chạy training và evaluation:

```bash
python src/train.py
```

## Kết quả chính

Dataset train:

- Tổng giao dịch: 1,296,675
- Fraud: 7,506
- Fraud rate: 0.5789%

Kết quả mô hình trên test set:

| Model | Threshold | Accuracy | Precision | Recall | F1-score | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| HistGradientBoosting tuned F1 | 0.80 | 0.9932 | 0.3557 | 0.9245 | 0.5137 | 0.9976 | 0.8701 |
| HistGradientBoosting default | 0.50 | 0.9843 | 0.1938 | 0.9669 | 0.3229 | 0.9976 | 0.8701 |
| Random Forest default/tuned | 0.50 | 0.9836 | 0.1802 | 0.9147 | 0.3011 | 0.9899 | 0.7441 |
| Logistic Regression tuned F1 | 0.71 | 0.9849 | 0.1577 | 0.6699 | 0.2553 | 0.9055 | 0.1189 |

HistGradientBoosting cho kết quả tốt nhất, vừa giữ Recall rất cao để giảm bỏ sót fraud, vừa tăng mạnh Precision và F1-score so với Random Forest. F1-score tăng từ 0.3011 của Random Forest lên 0.5137, PR-AUC tăng từ 0.7441 lên 0.8701.

## Output quan trọng

Tables:

- `reports/tables/t4.1_model_metrics.csv`
- `reports/tables/t4.2_classification_reports_all_models.csv`
- `reports/tables/t3.1_dataset_summary.csv`
- `reports/tables/t3.2_fraud_by_category.csv`
- `reports/tables/t3.4_high_risk_merchants.csv`

Figures:

- `reports/figures/f4.1_confusion_matrix_hist_gradient_boosting_tuned_f1.png`
- `reports/figures/f4.1_confusion_matrix_random_forest_tuned_f1.png`
- `reports/figures/f4.2_roc_curve_models.png`
- `reports/figures/f4.3_precision_recall_curve_models.png`
- `reports/figures/f4.4_model_metrics_comparison.png`
