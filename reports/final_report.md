# Báo Cáo Dự Án: Fraud Detection System using Big Data Analytics

## 1. Giới Thiệu

Gian lận giao dịch tài chính là một bài toán quan trọng trong ngân hàng, thương mại điện tử và các hệ thống thanh toán số. Mục tiêu của dự án là xây dựng quy trình phân tích dữ liệu lớn để phát hiện giao dịch gian lận từ dữ liệu lịch sử.

Dự án sử dụng dataset Credit Card Transactions Fraud Detection từ Kaggle, gồm dữ liệu train và test với nhãn `is_fraud`.

## 2. Mục Tiêu

- Khám phá và hiểu cấu trúc dữ liệu giao dịch.
- Tiền xử lý dữ liệu và xây dựng đặc trưng.
- Thể hiện xử lý dữ liệu lớn bằng Apache Spark.
- Huấn luyện ít nhất hai mô hình Machine Learning.
- Đánh giá mô hình bằng Accuracy, Precision, Recall, F1-score, ROC-AUC và PR-AUC.
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

Dataset bị mất cân bằng mạnh, do đó Accuracy không đủ để đánh giá mô hình. Các metric quan trọng hơn là Precision, Recall, F1-score, ROC-AUC và đặc biệt là PR-AUC vì fraud là lớp thiểu số.

## 3.1. Liên Hệ Với Đặc Trưng Dữ Liệu Lớn

Dự án được đặt trong bối cảnh môn Kỹ thuật và công nghệ dữ liệu lớn theo mô hình 5V:

- **Volume:** dữ liệu gồm hơn 1.85 triệu giao dịch trên cả train và test, đủ lớn để cần quy trình xử lý có tổ chức thay vì thao tác thủ công.
- **Velocity:** giao dịch tài chính trong thực tế phát sinh liên tục, hệ thống phát hiện gian lận cần có khả năng xử lý nhanh để hỗ trợ cảnh báo kịp thời.
- **Variety:** dữ liệu gồm nhiều kiểu thuộc tính như số tiền, thời gian, vị trí địa lý, category, merchant, job, gender, state.
- **Veracity:** cần kiểm tra missing values, duplicate, kiểu dữ liệu và loại bỏ các cột định danh trực tiếp để giảm nhiễu/rủi ro học thuộc.
- **Value:** kết quả mô hình giúp phát hiện giao dịch rủi ro cao, hỗ trợ giảm thiệt hại tài chính và tối ưu quy trình kiểm tra giao dịch.

Theo góc nhìn Big Data, dự án bao gồm ba nhiệm vụ chính:

- **Data Management:** tổ chức dữ liệu raw/processed, lưu bảng kết quả và model artifact.
- **Data Modeling and Analytics:** EDA, feature engineering, Spark aggregation và huấn luyện mô hình.
- **Visualization, Decisions and Values:** biểu đồ, bảng metrics, báo cáo PDF và slide thuyết trình.

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

- `f1.1_target_distribution.png`
- `f1.2_amount_distribution_log_scale.png`
- `f1.4_category_transaction_count_and_fraud_rate.png`
- `f1.5_merchant_transaction_count_and_fraud_rate.png`
- `f1.6_fraud_rate_by_hour.png`
- `f1.12_correlation_heatmap.png`

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

Spark được chọn vì phù hợp với các yêu cầu xử lý dữ liệu lớn:

- **Khả năng mở rộng:** cùng logic có thể chạy từ local mode lên cluster/cloud.
- **Xử lý song song/phân tán:** các phép groupBy/aggregation được Spark chia thành nhiều task.
- **Hỗ trợ tác vụ lặp:** phù hợp với quy trình phân tích nhiều lần trong EDA và feature aggregation.
- **Tích hợp hệ sinh thái:** Spark SQL/DataFrame dễ kết hợp với Python và pipeline Machine Learning.

## 7. Machine Learning

Notebook: `notebooks/04_Modeling.ipynb`  
Scripts:

- `src/train.py`
- `src/evaluate.py`

Mô hình đã huấn luyện:

- Logistic Regression.
- Random Forest.
- HistGradientBoosting.

Do dữ liệu mất cân bằng mạnh, training sử dụng sampling:

- Giữ toàn bộ giao dịch fraud.
- Lấy mẫu non-fraud theo tỷ lệ 5:1 so với fraud.
- Đánh giá trên toàn bộ test set.

Ngoài threshold mặc định 0.50, dự án thực hiện **threshold tuning** trên validation split của sampled train set. Threshold được chọn theo F1-score để cân bằng Precision và Recall, sau đó mô hình cuối được fit lại trên toàn bộ sampled train và đánh giá trên test set.

## 8. Evaluation

Kết quả trên test set:

| Model | Threshold | Accuracy | Precision | Recall | F1-score | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| HistGradientBoosting tuned F1 | 0.80 | 0.9932 | 0.3557 | 0.9245 | 0.5137 | 0.9976 | 0.8701 |
| HistGradientBoosting default | 0.50 | 0.9843 | 0.1938 | 0.9669 | 0.3229 | 0.9976 | 0.8701 |
| Random Forest default/tuned | 0.50 | 0.9836 | 0.1802 | 0.9147 | 0.3011 | 0.9899 | 0.7441 |
| Logistic Regression tuned F1 | 0.71 | 0.9849 | 0.1577 | 0.6699 | 0.2553 | 0.9055 | 0.1189 |

Nhận xét:

- HistGradientBoosting tuned F1 là mô hình tốt nhất theo F1-score, Recall, ROC-AUC và PR-AUC.
- Recall đạt khoảng 92.45%, nghĩa là mô hình phát hiện được phần lớn giao dịch gian lận.
- F1-score tăng từ 0.3011 của Random Forest lên 0.5137.
- PR-AUC tăng từ 0.7441 của Random Forest lên 0.8701, rất quan trọng vì fraud là lớp thiểu số.
- HistGradientBoosting default đạt Recall khoảng 96.69%, phù hợp nếu mục tiêu là giảm tối đa bỏ sót fraud, đổi lại Precision thấp hơn.

Biểu đồ đã xuất:

- `f4.1_confusion_matrix_hist_gradient_boosting_tuned_f1.png`
- `f4.1_confusion_matrix_random_forest_tuned_f1.png`
- `f4.2_roc_curve_models.png`
- `f4.3_precision_recall_curve_models.png`
- `f4.4_model_metrics_comparison.png`

## 9. Kết Luận

Dự án đã hoàn thành pipeline phân tích dữ liệu lớn cho bài toán phát hiện gian lận:

- Có EDA đầy đủ.
- Có preprocessing và feature engineering.
- Có xử lý dữ liệu bằng Apache Spark.
- Có huấn luyện ba mô hình Machine Learning.
- Có đánh giá bằng các metric phù hợp với dữ liệu mất cân bằng.
- Có biểu đồ và bảng kết quả phục vụ báo cáo.

HistGradientBoosting tuned F1 là mô hình tốt nhất trong phạm vi thử nghiệm hiện tại với Accuracy khoảng 0.9932, Precision khoảng 0.3557, Recall khoảng 0.9245, F1-score khoảng 0.5137, ROC-AUC khoảng 0.9976 và PR-AUC khoảng 0.8701. So với Random Forest, mô hình này cải thiện rõ rệt cả khả năng phát hiện fraud và chất lượng cảnh báo.

## 10. Hướng Phát Triển

- Thử XGBoost hoặc LightGBM.
- Tối ưu threshold thay vì dùng mặc định 0.5.
- Dùng Precision-Recall AUC do dữ liệu mất cân bằng mạnh.
- Thử target encoding có regularization cho merchant/job.
- Triển khai pipeline batch scoring bằng Spark.
