# Slide Outline - Fraud Detection System using Big Data Analytics

## Slide 1 - Title

Fraud Detection System using Big Data Analytics

Tên nhóm / môn học / giảng viên / ngày trình bày.

## Slide 2 - Problem Statement

- Giao dịch gian lận gây thiệt hại tài chính lớn.
- Cần hệ thống phát hiện fraud từ dữ liệu giao dịch.
- Bài toán khó vì dữ liệu mất cân bằng mạnh.

## Slide 3 - Dataset

- Nguồn: Kaggle Fraud Detection.
- Train: 1,296,675 giao dịch.
- Test: 555,719 giao dịch.
- Target: `is_fraud`.
- Fraud rate train: 0.5789%.

## Slide 4 - System Pipeline

Raw Data -> EDA -> Preprocessing -> Feature Engineering -> Spark Processing -> Machine Learning -> Evaluation -> Report

## Slide 5 - EDA Highlights

- Fraud chiếm dưới 1%.
- Amount phân phối lệch phải.
- Fraud rate khác nhau theo category/merchant.
- Thời gian giao dịch và khoảng cách địa lý có thể hữu ích.

Hình gợi ý:

- `target_distribution.png`
- `amount_distribution_log_scale.png`
- `category_transaction_count_and_fraud_rate.png`

## Slide 6 - Preprocessing

- Tạo feature thời gian.
- Tạo tuổi khách hàng.
- Tạo khoảng cách customer-merchant.
- One-hot encode category/gender/state.
- Frequency encode merchant/job/city.
- Scale feature số.

## Slide 7 - Apache Spark

- Dùng Spark DataFrame đọc `fraudTrain.csv`.
- Tính fraud theo category/state/merchant/hour.
- Xuất bảng kết quả vào `reports/tables`.
- Spark version: 3.5.1.

## Slide 8 - Pandas vs Spark

Aggregation theo category:

- Pandas: khoảng 3.40s.
- Spark: khoảng 0.65s.

Thông điệp: Spark phù hợp hơn khi dữ liệu lớn và cần xử lý phân tán.

## Slide 9 - Models

Mô hình:

- Logistic Regression.
- Random Forest.

Chiến lược xử lý mất cân bằng:

- Giữ toàn bộ fraud.
- Sample non-fraud theo tỷ lệ 5:1.
- Đánh giá trên toàn bộ test set.

## Slide 10 - Evaluation Metrics

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0.9836 | 0.1802 | 0.9147 | 0.3011 | 0.9899 |
| Logistic Regression | 0.9393 | 0.0451 | 0.7305 | 0.0850 | 0.9055 |

## Slide 11 - Visualization Results

Hình gợi ý:

- `confusion_matrix_random_forest.png`
- `roc_curve_models.png`
- `model_metrics_comparison.png`

## Slide 12 - Conclusion

- Hoàn thành pipeline Big Data + ML cho fraud detection.
- Random Forest là mô hình tốt nhất trong thử nghiệm.
- Recall cao giúp giảm bỏ sót fraud.
- Precision còn thấp do dữ liệu cực kỳ mất cân bằng.

## Slide 13 - Future Work

- Tối ưu threshold.
- Thử XGBoost/LightGBM.
- Thêm PR-AUC.
- Triển khai batch scoring bằng Spark.
