# Báo Cáo Dự Án: Fraud Detection System using Big Data Analytics

## 1. Giới Thiệu

Gian lận giao dịch tài chính là một bài toán quan trọng trong ngân hàng, thương mại điện tử và các hệ thống thanh toán số. Mục tiêu của dự án là xây dựng quy trình phân tích dữ liệu lớn để phát hiện giao dịch gian lận từ dữ liệu lịch sử.

Dự án sử dụng dataset Credit Card Transactions Fraud Detection từ Kaggle, gồm dữ liệu train và test với nhãn `is_fraud`.

## 2. Mục Tiêu

- Khám phá và hiểu cấu trúc dữ liệu giao dịch.
- Tiền xử lý dữ liệu và xây dựng đặc trưng.
- Thể hiện xử lý dữ liệu lớn bằng Apache Spark.
- Huấn luyện ít nhất hai mô hình Machine Learning.
- Đánh giá mô hình bằng Accuracy, Precision, Recall, F1-score và ROC-AUC.
- Trực quan hóa kết quả phục vụ báo cáo môn Big Data.

## 3. Dataset

Nguồn dữ liệu: Kaggle - Fraud Detection.

File sử dụng:

- `fraudTrain.csv`: 1,296,675 giao dịch.
- `fraudTest.csv`: 555,719 giao dịch.

Biến mục tiêu:

- `0`: giao dịch hợp lệ.
- `1`: giao dịch gian lận.

Phân bố train:

- Non-fraud: 1,289,169.
- Fraud: 7,506.
- Fraud rate: 0.5789%.

Dataset bị mất cân bằng mạnh, do đó Accuracy không đủ để đánh giá mô hình. Các metric quan trọng hơn là Precision, Recall, F1-score và ROC-AUC.

## 4. Exploratory Data Analysis

Notebook: `notebooks/01_EDA.ipynb`

Các nội dung đã thực hiện:

- Kiểm tra kích thước dữ liệu, kiểu dữ liệu và missing values.
- Phân tích phân bố lớp `is_fraud`.
- Phân tích số tiền giao dịch `amt`.
- Phân tích fraud rate theo `category`.
- Phân tích merchant phổ biến và merchant có rủi ro cao.
- Phân tích thời gian giao dịch theo giờ, ngày trong tuần, tháng.
- Phân tích thông tin khách hàng như gender, age, state.
- Tạo và phân tích khoảng cách giữa khách hàng và merchant.

Một số biểu đồ đã xuất:

- `target_distribution.png`
- `amount_distribution_log_scale.png`
- `category_transaction_count_and_fraud_rate.png`
- `merchant_transaction_count_and_fraud_rate.png`
- `fraud_rate_by_hour.png`
- `correlation_heatmap.png`

## 5. Preprocessing Và Feature Engineering

Notebook: `notebooks/02_Preprocessing.ipynb`  
Script: `src/preprocessing.py`

Các bước chính:

- Tạo feature thời gian:
  - `transaction_hour`
  - `transaction_dayofweek`
  - `transaction_month`
  - `weekend_flag`
- Tạo `customer_age` từ ngày sinh và thời điểm giao dịch.
- Tạo `customer_merchant_distance_km` bằng công thức Haversine.
- One-hot encode:
  - `category`
  - `gender`
  - `state`
- Frequency encode:
  - `merchant`
  - `job`
  - `city`
- Loại bỏ các cột định danh trực tiếp:
  - `cc_num`
  - `first`
  - `last`
  - `street`
  - `trans_num`
  - `trans_date_trans_time`
  - `dob`
- Scale các biến số bằng mean/std từ train set.

Output:

- `data/processed/fraud_train_processed.csv`
- `data/processed/fraud_test_processed.csv`

Dữ liệu processed:

- Train: 1,296,675 dòng, 85 cột.
- Test: 555,719 dòng, 85 cột.
- Missing values: 0.

## 6. Apache Spark Processing

Notebook: `notebooks/03_Spark_Processing.ipynb`  
Script: `src/spark_processing.py`

Spark được dùng để đọc và xử lý dữ liệu train bằng Spark DataFrame.

Các bảng đã tạo:

- Tổng quan dataset.
- Fraud theo category.
- Fraud theo state.
- Merchant có tỷ lệ fraud cao.
- Fraud theo giờ giao dịch.
- Thống kê amount theo target.
- So sánh Pandas và Spark.

Kết quả Spark:

- Spark version: 3.5.1.
- Tổng dòng xử lý: 1,296,675.
- Fraud count: 7,506.
- Fraud rate: 0.578865%.

So sánh aggregation theo category:

| Method | Time |
|---|---:|
| Pandas | khoảng 3.40s |
| Spark | khoảng 0.65s |

Kết quả cho thấy Spark xử lý aggregation trên dữ liệu lớn hiệu quả và phù hợp với yêu cầu Big Data.

## 7. Machine Learning

Notebook: `notebooks/04_Modeling.ipynb`  
Scripts:

- `src/train.py`
- `src/evaluate.py`

Mô hình đã huấn luyện:

- Logistic Regression.
- Random Forest.

Do dữ liệu mất cân bằng mạnh, training sử dụng sampling:

- Giữ toàn bộ giao dịch fraud.
- Lấy mẫu non-fraud theo tỷ lệ 5:1 so với fraud.
- Đánh giá trên toàn bộ test set.

## 8. Evaluation

Kết quả trên test set:

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0.9836 | 0.1802 | 0.9147 | 0.3011 | 0.9899 |
| Logistic Regression | 0.9393 | 0.0451 | 0.7305 | 0.0850 | 0.9055 |

Nhận xét:

- Random Forest vượt Logistic Regression ở Recall, F1-score và ROC-AUC.
- Recall của Random Forest đạt khoảng 91.47%, nghĩa là mô hình phát hiện được phần lớn giao dịch gian lận.
- Precision còn thấp do tỷ lệ fraud rất nhỏ trong dữ liệu thực tế.
- Với hệ thống fraud detection, Recall cao thường được ưu tiên để giảm rủi ro bỏ sót gian lận.

Biểu đồ đã xuất:

- `confusion_matrix_random_forest.png`
- `confusion_matrix_logistic_regression.png`
- `roc_curve_models.png`
- `model_metrics_comparison.png`

## 9. Kết Luận

Dự án đã hoàn thành pipeline phân tích dữ liệu lớn cho bài toán phát hiện gian lận:

- Có EDA đầy đủ.
- Có preprocessing và feature engineering.
- Có xử lý dữ liệu bằng Apache Spark.
- Có huấn luyện hai mô hình Machine Learning.
- Có đánh giá bằng các metric phù hợp với dữ liệu mất cân bằng.
- Có biểu đồ và bảng kết quả phục vụ báo cáo.

Random Forest là mô hình tốt nhất trong phạm vi thử nghiệm hiện tại với ROC-AUC khoảng 0.9899 và Recall khoảng 0.9147.

## 10. Hướng Phát Triển

- Thử XGBoost hoặc LightGBM.
- Tối ưu threshold thay vì dùng mặc định 0.5.
- Dùng Precision-Recall AUC do dữ liệu mất cân bằng mạnh.
- Thử target encoding có regularization cho merchant/job.
- Triển khai pipeline batch scoring bằng Spark.
