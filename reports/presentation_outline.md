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

## Slide 5 - Big Data 5V Mapping

- Volume: hơn 1.85 triệu giao dịch train + test.
- Velocity: giao dịch tài chính phát sinh liên tục, cần phát hiện nhanh.
- Variety: amount, time, location, merchant, category, user profile.
- Veracity: kiểm tra missing, duplicate, loại bỏ cột định danh/nhiễu.
- Value: phát hiện fraud giúp giảm thiệt hại và hỗ trợ ra quyết định.

## Slide 6 - EDA Highlights

- Fraud chiếm dưới 1%.
- Amount phân phối lệch phải.
- Fraud rate khác nhau theo category/merchant.
- Thời gian giao dịch và khoảng cách địa lý có thể hữu ích.

Hình gợi ý:

- `f1.1_target_distribution.png`
- `f1.2_amount_distribution_log_scale.png`
- `f1.4_category_transaction_count_and_fraud_rate.png`

## Slide 7 - Preprocessing

- Tạo feature thời gian.
- Tạo tuổi khách hàng.
- Tạo khoảng cách customer-merchant.
- One-hot encode category/gender/state.
- Frequency encode merchant/job/city.
- Scale feature số.

## Slide 8 - Apache Spark

- Dùng Spark DataFrame đọc `fraudTrain.csv`.
- Tính fraud theo category/state/merchant/hour.
- Xuất bảng kết quả vào `reports/tables`.
- Spark version: 3.5.1.

## Slide 9 - Pandas vs Spark

Aggregation theo category:

- Pandas: khoảng 3.40s.
- Spark: khoảng 0.65s.

Thông điệp: Spark phù hợp hơn khi dữ liệu lớn và cần xử lý phân tán.

## Slide 10 - Models

Mô hình:

- Logistic Regression.
- Random Forest.
- HistGradientBoosting.
- Threshold tuning:
  - Tune threshold trên validation split theo F1-score.
  - Evaluate cuối trên toàn bộ test set.

Chiến lược xử lý mất cân bằng:

- Giữ toàn bộ fraud.
- Sample non-fraud theo tỷ lệ 5:1.
- Đánh giá trên toàn bộ test set.

## Slide 11 - Evaluation Metrics

| Model | Threshold | Accuracy | Precision | Recall | F1-score | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| HistGradientBoosting tuned F1 | 0.80 | 0.9932 | 0.3557 | 0.9245 | 0.5137 | 0.9976 | 0.8701 |
| HistGradientBoosting default | 0.50 | 0.9843 | 0.1938 | 0.9669 | 0.3229 | 0.9976 | 0.8701 |
| Random Forest default/tuned | 0.50 | 0.9836 | 0.1802 | 0.9147 | 0.3011 | 0.9899 | 0.7441 |
| Logistic Regression tuned F1 | 0.71 | 0.9849 | 0.1577 | 0.6699 | 0.2553 | 0.9055 | 0.1189 |

## Slide 12 - Visualization Results

Hình gợi ý:

- `f4.1_confusion_matrix_hist_gradient_boosting_tuned_f1.png`
- `f4.2_roc_curve_models.png`
- `f4.3_precision_recall_curve_models.png`
- `f4.4_model_metrics_comparison.png`

## Slide 13 - Conclusion

- Hoàn thành pipeline Big Data + ML cho fraud detection.
- HistGradientBoosting là mô hình tốt nhất trong thử nghiệm.
- Recall đạt 0.9245, giúp giảm bỏ sót fraud.
- F1-score tăng từ 0.3011 của Random Forest lên 0.5137.
- PR-AUC tăng từ 0.7441 lên 0.8701, phù hợp với dữ liệu mất cân bằng.

## Slide 14 - Future Work

- Tối ưu threshold.
- Thử XGBoost/LightGBM.
- Thêm PR-AUC.
- Triển khai batch scoring bằng Spark.
