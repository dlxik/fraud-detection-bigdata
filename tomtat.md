# Tóm Tắt Project Fraud Detection

## 1. Project Này Là Gì

Project này là một pipeline phát hiện giao dịch gian lận tài chính bằng Big Data Analytics và Machine Learning.

Nói đơn giản: cho một tập dữ liệu giao dịch thẻ, mỗi giao dịch có thông tin thời gian, số tiền, merchant, vị trí, khách hàng... Project học từ dữ liệu cũ để dự đoán giao dịch mới có phải fraud hay không.

Biến cần dự đoán:

```text
is_fraud
0 = giao dịch hợp lệ
1 = giao dịch gian lận
```

Dataset lấy từ Kaggle Fraud Detection, gồm:

```text
fraudTrain.csv
fraudTest.csv
```

Train có khoảng `1,296,675` giao dịch. Trong đó fraud chỉ có `7,506`, tức fraud rate khoảng `0.5789%`.

Đây là điểm rất quan trọng: dữ liệu cực kỳ mất cân bằng. Trong 1,000 giao dịch, chỉ khoảng 6 giao dịch là fraud. Vì vậy nếu model đoán tất cả là non-fraud thì accuracy vẫn rất cao, nhưng model vô dụng.

Do đó project không đánh giá chủ yếu bằng accuracy, mà dùng thêm:

- Recall
- Precision
- F1-score
- ROC-AUC
- PR-AUC

Đặc biệt với fraud detection, `Recall` và `PR-AUC` rất quan trọng.

## 2. Pipeline Tổng Thể

Luồng tổng thể của project:

```text
Raw Data
  -> EDA
  -> Preprocessing / Feature Engineering
  -> Spark Processing
  -> Machine Learning
  -> Evaluation
  -> Report / Slide / Model Artifact
```

Mỗi phần có vai trò riêng.

## 2.1. Quy Ước Đánh Số Hình Và Bảng

Các hình và bảng trong `reports/figures` và `reports/tables` được đánh số theo nguồn sinh ra artifact, không đánh theo thứ tự đang nằm trong folder.

Quy ước:

- `f1.x_...`: hình sinh từ notebook `01_EDA.ipynb`.
- `t1.x_...`: bảng sinh từ notebook `01_EDA.ipynb`.
- `t2.x_...`: bảng sinh từ script/notebook preprocessing.
- `t3.x_...`: bảng sinh từ `src/spark_processing.py` hoặc notebook Spark.
- `f4.x_...`: hình sinh từ `src/evaluate.py` trong quá trình modeling.
- `t4.x_...`: bảng sinh từ `src/train.py`.

Ý nghĩa:

- Nhìn tên file là biết artifact thuộc phần nào của pipeline.
- Nếu chạy lại code/notebook thì artifact vẫn được sinh ra đúng tên đã đánh số.
- Không phụ thuộc vào thứ tự file hiển thị trong folder.

Ví dụ:

```text
f1.1_target_distribution.png
t3.2_fraud_by_category.csv
f4.2_roc_curve_models.png
t4.1_model_metrics.csv
```

## 3. EDA Làm Gì

EDA là bước khám phá dữ liệu.

Mục tiêu của EDA là hiểu dữ liệu:

- Dữ liệu có bao nhiêu dòng/cột?
- Fraud chiếm bao nhiêu phần trăm?
- Amount phân phối như thế nào?
- Category nào có fraud rate cao?
- Merchant nào rủi ro?
- Fraud xảy ra nhiều vào giờ nào?
- Vị trí khách hàng và merchant có liên quan không?
- Tuổi, giới tính, bang, nghề nghiệp có dấu hiệu gì không?

Các biểu đồ quan trọng:

- `f1.1_target_distribution.png`
- `f1.2_amount_distribution_log_scale.png`
- `f1.4_category_transaction_count_and_fraud_rate.png`
- `f1.5_merchant_transaction_count_and_fraud_rate.png`
- `f1.6_fraud_rate_by_hour.png`
- `f1.12_correlation_heatmap.png`

Thông điệp chính của EDA:

> Fraud rất hiếm, dữ liệu lệch mạnh, và fraud có xu hướng khác nhau theo category, merchant, thời gian, amount và vị trí.

## 4. Preprocessing Làm Gì

Script chính:

```text
src/preprocessing.py
```

Raw data ban đầu không thể đưa thẳng vào model vì có nhiều cột dạng text, ngày tháng, định danh cá nhân.

### Tạo Feature Thời Gian

Từ `trans_date_trans_time`, tạo:

```text
transaction_hour
transaction_dayofweek
transaction_month
weekend_flag
```

Ý nghĩa: fraud có thể xảy ra nhiều ở giờ hoặc khoảng thời gian nhất định.

### Tạo Tuổi Khách Hàng

Từ `dob` và thời điểm giao dịch:

```text
customer_age
```

Ý nghĩa: hành vi giao dịch có thể khác nhau theo tuổi.

### Tạo Khoảng Cách Khách Hàng - Merchant

Dùng tọa độ:

```text
lat, long
merch_lat, merch_long
```

Tạo:

```text
customer_merchant_distance_km
```

Ý nghĩa: nếu khách hàng và merchant cách nhau bất thường, có thể là dấu hiệu fraud.

### Encode Categorical

Model không hiểu text như `category`, `gender`, `state`, `merchant`, `job`, `city`, nên phải encode.

One-hot encode:

```text
category
gender
state
```

Frequency encode:

```text
merchant
job
city
```

Frequency encoding nghĩa là thay giá trị text bằng tần suất xuất hiện của nó trong train set.

### Scale Numeric Features

Các cột số được chuẩn hóa:

```text
amt
city_pop
lat
long
merch_lat
merch_long
unix_time
customer_age
customer_merchant_distance_km
```

Sau preprocessing:

```text
Train: 1,296,675 dòng, 85 cột
Test: 555,719 dòng, 85 cột
```

Trong đó có `84` feature và `1` target là `is_fraud`.

## 5. Spark Processing Làm Gì

Script chính:

```text
src/spark_processing.py
```

Phần này dùng PySpark để thể hiện năng lực Big Data.

Spark không phải để train model chính trong project này, mà để xử lý và aggregation dữ liệu lớn.

Nó tính các bảng:

- Tổng quan dataset
- Fraud theo category
- Fraud theo state
- Merchant rủi ro cao
- Fraud theo giờ
- Amount theo target
- So sánh Pandas vs Spark

Ý nghĩa trong báo cáo:

> Project không chỉ dùng sklearn, mà có thành phần xử lý dữ liệu lớn bằng Spark DataFrame. Spark phù hợp khi dữ liệu lớn hơn và cần xử lý phân tán.

Nên nhấn:

- Spark đọc raw transaction data.
- Spark groupBy/aggregate fraud theo nhiều chiều.
- Spark nhanh hơn Pandas trong aggregation category trên máy này.
- Logic Spark có thể mở rộng lên cluster/cloud.

## 6. Machine Learning Làm Gì

Script chính:

```text
src/train.py
```

Project train 3 model:

```text
Logistic Regression
Random Forest
HistGradientBoosting
```

### Logistic Regression

Model tuyến tính, dùng làm baseline.

Ưu điểm:

- Dễ hiểu
- Nhanh
- Là baseline tốt

Nhược điểm:

- Khó bắt quan hệ phức tạp

### Random Forest

Model ensemble nhiều cây quyết định.

Ưu điểm:

- Bắt quan hệ phi tuyến tốt
- Mạnh hơn Logistic Regression
- Có kết quả recall tốt

### HistGradientBoosting

Model boosting mạnh hơn, học tuần tự nhiều cây để sửa lỗi của cây trước.

Đây là model tốt nhất hiện tại.

Kết quả tốt nhất:

```text
HistGradientBoosting tuned F1
Accuracy  = 0.9932
Precision = 0.3557
Recall    = 0.9245
F1-score  = 0.5137
ROC-AUC   = 0.9976
PR-AUC    = 0.8701
```

So với Random Forest:

```text
Random Forest
Precision = 0.1802
Recall    = 0.9147
F1-score  = 0.3011
PR-AUC    = 0.7441
```

HistGradientBoosting tốt hơn rõ:

- Recall cao hơn
- Precision cao hơn
- F1 cao hơn nhiều
- PR-AUC cao hơn nhiều

## 7. Xử Lý Mất Cân Bằng

Đây là phần rất quan trọng khi bảo vệ.

Vì fraud chỉ chiếm `0.5789%`, nếu train bình thường model dễ thiên về class non-fraud.

Project dùng sampling:

```text
Giữ toàn bộ fraud
Lấy mẫu non-fraud theo tỷ lệ 5:1
```

Tức là trong training sample, model được nhìn thấy fraud nhiều hơn so với thực tế, để học được đặc điểm fraud.

Nhưng evaluation vẫn thực hiện trên toàn bộ test set, không phải test sample.

Nên nói:

> Nhóm chỉ dùng sampling ở training để giúp model học lớp thiểu số, còn đánh giá vẫn dùng full test set để phản ánh dữ liệu thực tế.

## 8. Threshold Tuning Là Gì

Model thường trả về xác suất fraud.

Ví dụ:

```text
transaction A -> fraud probability = 0.82
transaction B -> fraud probability = 0.31
```

Mặc định nếu probability >= `0.5` thì predict fraud.

Nhưng threshold `0.5` không phải lúc nào tối ưu, nhất là dữ liệu mất cân bằng.

Project tune threshold theo F1-score trên validation set.

Best model hiện tại:

```text
HistGradientBoosting tuned F1 threshold = 0.80
```

Có nghĩa:

```text
Nếu probability >= 0.80 -> fraud
Nếu probability < 0.80 -> non-fraud
```

Tại threshold này, model đạt:

```text
Precision = 0.3557
Recall = 0.9245
F1 = 0.5137
```

Tư duy cần hiểu:

- Hạ threshold -> bắt nhiều fraud hơn, recall tăng, nhưng false positive tăng.
- Tăng threshold -> cảnh báo ít hơn, precision có thể tăng, nhưng recall có thể giảm.
- F1 là điểm cân bằng giữa precision và recall.

## 9. Giải Thích Các Metric

### Accuracy

Tỷ lệ đoán đúng tổng thể.

Nhưng với fraud detection, accuracy dễ gây ảo giác vì non-fraud quá nhiều.

### Precision

Trong các giao dịch model báo fraud, bao nhiêu cái thật sự là fraud?

```text
Precision = TP / (TP + FP)
```

Precision `0.3557` nghĩa là trong 100 giao dịch bị flag fraud, khoảng 36 giao dịch thật sự fraud.

Nghe chưa quá cao, nhưng với fraud rate chỉ `0.58%`, đây là cải thiện lớn.

### Recall

Trong toàn bộ fraud thật, model bắt được bao nhiêu?

```text
Recall = TP / (TP + FN)
```

Recall `0.9245` nghĩa là model bắt được khoảng 92.45% fraud.

Đây là điểm rất mạnh.

### F1-score

Trung bình điều hòa giữa precision và recall.

```text
F1 = cân bằng giữa precision và recall
```

F1 `0.5137` là khá tốt với dữ liệu fraud cực lệch.

### ROC-AUC

Đo khả năng phân biệt fraud/non-fraud trên nhiều threshold.

`0.9976` là rất cao.

### PR-AUC

Đo chất lượng precision-recall curve.

Với imbalanced data, PR-AUC quan trọng hơn ROC-AUC.

`PR-AUC = 0.8701` là điểm rất đáng khoe.

## 10. Câu Chuyện Chính Của Project

Nếu phải tóm project bằng một câu chuyện logic:

> Dữ liệu giao dịch có hơn 1.8 triệu bản ghi và cực kỳ mất cân bằng, fraud chỉ khoảng 0.58%. Nhóm xây dựng pipeline Big Data gồm EDA, preprocessing, Spark aggregation, machine learning và evaluation. Sau khi so sánh Logistic Regression, Random Forest và HistGradientBoosting, model HistGradientBoosting cho kết quả tốt nhất với Recall 92.45%, F1-score 0.5137 và PR-AUC 0.8701. Điều này cho thấy model có khả năng phát hiện phần lớn giao dịch gian lận trong điều kiện dữ liệu rất mất cân bằng.

## 11. Gợi Ý Cấu Trúc Báo Cáo

Bố cục nên tách riêng phần Spark thành một chương riêng. Lý do: Spark không chỉ là mô tả dữ liệu, mà là phần thể hiện năng lực xử lý dữ liệu lớn của project. Nếu để Spark trong chương dữ liệu thì vai trò Big Data sẽ bị chìm.

### 1. Giới Thiệu

Mục này đặt vấn đề và giải thích vì sao bài toán đáng làm.

Nội dung nên có:

- Gian lận giao dịch tài chính gây thiệt hại lớn cho ngân hàng, ví điện tử, thương mại điện tử và hệ thống thanh toán số.
- Số lượng giao dịch ngày càng lớn, khiến kiểm tra thủ công không còn phù hợp.
- Cần xây dựng hệ thống hỗ trợ phát hiện giao dịch rủi ro dựa trên dữ liệu lịch sử.
- Bài toán khó vì fraud là lớp rất hiếm, dữ liệu bị mất cân bằng mạnh.
- Mục tiêu project: xây dựng pipeline Big Data + Machine Learning để phát hiện giao dịch gian lận.

Câu có thể dùng:

> Mục tiêu của đề tài là xây dựng một pipeline phân tích dữ liệu lớn và học máy nhằm phát hiện giao dịch gian lận, đồng thời đánh giá mô hình bằng các metric phù hợp với dữ liệu mất cân bằng như Recall, F1-score và PR-AUC.

### 2. Dữ Liệu Và Phân Tích Ban Đầu

Chương này tập trung vào việc hiểu dữ liệu trước khi xử lý bằng Spark và train model.

#### 2.1. Nguồn Dữ Liệu

Nội dung nên có:

- Dataset lấy từ Kaggle Fraud Detection.
- Hai file sử dụng:
  - `fraudTrain.csv`
  - `fraudTest.csv`
- Biến mục tiêu:
  - `is_fraud = 0`: giao dịch hợp lệ
  - `is_fraud = 1`: giao dịch gian lận
- Train có `1,296,675` giao dịch.
- Test có `555,719` giao dịch.

Mục đích của phần này là cho người đọc biết dữ liệu đến từ đâu, gồm những file nào, và bài toán dự đoán cột nào.

#### 2.2. Mô Tả Dữ Liệu

Nội dung nên có:

- Số dòng, số cột của train/test.
- Các nhóm thuộc tính chính:
  - Thông tin giao dịch: `amt`, `category`, `merchant`, `trans_date_trans_time`
  - Thông tin khách hàng: `gender`, `job`, `dob`, `city`, `state`
  - Thông tin vị trí: `lat`, `long`, `merch_lat`, `merch_long`
  - Thông tin định danh: `cc_num`, `trans_num`, `first`, `last`, `street`
- Phân bố target:
  - Fraud train: `7,506`
  - Fraud rate: khoảng `0.5789%`

Ý chính cần nhấn:

> Dataset có kích thước lớn và mất cân bằng rất mạnh. Đây là lý do accuracy không đủ để đánh giá mô hình.

#### 2.3. Đặc Trưng Big Data 5V

Phần này rất nên có vì đây là môn Big Data.

Nội dung nên viết theo 5V:

- Volume: dữ liệu có hơn 1.85 triệu giao dịch trên cả train và test.
- Velocity: trong thực tế, giao dịch tài chính phát sinh liên tục và cần cảnh báo nhanh.
- Variety: dữ liệu gồm nhiều kiểu thông tin như số tiền, thời gian, category, merchant, nghề nghiệp, giới tính, vị trí địa lý.
- Veracity: cần kiểm tra missing values, duplicate, kiểu dữ liệu và loại bỏ các cột định danh trực tiếp để giảm nhiễu/học thuộc.
- Value: mô hình giúp phát hiện giao dịch rủi ro, giảm thiệt hại tài chính và hỗ trợ quyết định kiểm tra giao dịch.

Thông điệp:

> Dữ liệu giao dịch tài chính có đầy đủ đặc trưng của bài toán dữ liệu lớn, nên cần một pipeline xử lý có tổ chức thay vì thao tác thủ công.

#### 2.4. Phân Tích Khám Phá Dữ Liệu

Nội dung nên có:

- Phân tích phân bố `is_fraud`.
- Phân tích phân bố số tiền giao dịch `amt`.
- Fraud rate theo `category`.
- Merchant phổ biến và merchant rủi ro cao.
- Fraud rate theo giờ giao dịch, ngày trong tuần, tháng.
- Phân tích theo gender, age, state.
- Phân tích khoảng cách giữa khách hàng và merchant.

Hình nên chèn:

- `f1.1_target_distribution.png`
- `f1.2_amount_distribution_log_scale.png`
- `f1.4_category_transaction_count_and_fraud_rate.png`
- `f1.5_merchant_transaction_count_and_fraud_rate.png`
- `f1.6_fraud_rate_by_hour.png`
- `f1.12_correlation_heatmap.png`

Insight nên nói:

- Fraud rất hiếm.
- Amount bị lệch phải.
- Fraud rate khác nhau theo category và merchant.
- Thời gian giao dịch có ảnh hưởng đến nguy cơ fraud.
- Khoảng cách địa lý có thể là dấu hiệu hữu ích.

#### 2.5. Tiền Xử Lý Và Xây Dựng Đặc Trưng

Phần này giải thích raw data được biến thành dữ liệu model-ready như thế nào.

Nội dung nên có:

- Parse cột thời gian `trans_date_trans_time`.
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
- Loại bỏ các cột định danh trực tiếp:
  - `cc_num`
  - `first`
  - `last`
  - `street`
  - `trans_num`
  - `trans_date_trans_time`
  - `dob`
  - `merchant`
  - `job`
  - `city`

Kết quả sau preprocessing:

```text
Train: 1,296,675 dòng, 85 cột
Test: 555,719 dòng, 85 cột
Missing values: 0
```

Điểm cần nhấn:

> Các cột định danh trực tiếp được loại bỏ để giảm nguy cơ model học thuộc từng cá nhân hoặc từng giao dịch cụ thể.

### 3. Xử Lý Dữ Liệu Lớn Với Apache Spark

Chương này nên tách riêng để làm nổi bật yếu tố Big Data.

#### 3.1. Mục Đích Sử Dụng Spark

Nội dung nên có:

- Spark được dùng để đọc và xử lý dữ liệu giao dịch bằng Spark DataFrame.
- Mục tiêu là thực hiện các phép aggregation trên dữ liệu lớn.
- Spark phù hợp với dữ liệu lớn vì có khả năng xử lý song song và mở rộng lên cluster.

Thông điệp:

> Trong project này, Spark đóng vai trò là công cụ xử lý và phân tích dữ liệu lớn, còn scikit-learn được dùng cho phần huấn luyện mô hình.

#### 3.2. Các Phân Tích Bằng Spark

Nội dung nên có:

- Tổng quan dataset:
  - tổng số dòng
  - số fraud
  - số non-fraud
  - fraud rate
- Fraud theo category.
- Fraud theo state.
- Top merchant rủi ro cao.
- Fraud theo giờ giao dịch.
- Thống kê amount theo target.

Các bảng output có thể nhắc:

- `t3.1_dataset_summary.csv`
- `t3.2_fraud_by_category.csv`
- `t3.3_fraud_by_state.csv`
- `t3.4_high_risk_merchants.csv`
- `t3.5_fraud_by_hour.csv`
- `t3.6_amount_by_target.csv`

#### 3.3. So Sánh Pandas Và Spark

Nội dung nên có:

- So sánh thời gian aggregation theo category.
- Pandas xử lý trên một máy, phù hợp dữ liệu vừa và nhỏ.
- Spark có overhead ban đầu, nhưng có lợi thế khi dữ liệu lớn hơn hoặc cần xử lý phân tán.

Thông điệp:

> Kết quả so sánh cho thấy Spark phù hợp hơn cho các tác vụ aggregation có khả năng mở rộng, đặc biệt khi dữ liệu tăng kích thước hoặc chạy trên cluster.

#### 3.4. Ý Nghĩa Của Spark Trong Pipeline

Nội dung nên có:

- Spark giúp project không chỉ là bài sklearn đơn giản.
- Spark thể hiện phần Data Management và Big Data Analytics.
- Các kết quả Spark hỗ trợ hiểu dữ liệu và tạo insight cho báo cáo.

### 4. Xây Dựng Mô Hình Phát Hiện Gian Lận

Chương này nói về chiến lược modeling.

#### 4.1. Bài Toán Phân Loại Nhị Phân

Nội dung nên có:

- Input: các feature sau preprocessing.
- Output: dự đoán `is_fraud`.
- Đây là bài toán binary classification:
  - `0`: non-fraud
  - `1`: fraud

#### 4.2. Xử Lý Mất Cân Bằng Dữ Liệu

Nội dung nên có:

- Fraud chỉ chiếm khoảng `0.5789%`.
- Nếu train trực tiếp, model dễ thiên về non-fraud.
- Project dùng sampling:
  - giữ toàn bộ fraud
  - lấy mẫu non-fraud theo tỷ lệ `5:1`
- Evaluation vẫn dùng full test set.

Câu quan trọng:

> Sampling chỉ được dùng ở training để model học tốt hơn lớp thiểu số; kết quả đánh giá vẫn được tính trên toàn bộ test set để phản ánh phân bố dữ liệu thực tế.

#### 4.3. Các Mô Hình Sử Dụng

Nội dung nên có:

- Logistic Regression:
  - baseline tuyến tính
  - nhanh, dễ diễn giải
- Random Forest:
  - ensemble nhiều cây quyết định
  - bắt quan hệ phi tuyến tốt hơn
- HistGradientBoosting:
  - boosting model
  - học tuần tự để sửa lỗi của mô hình trước
  - là mô hình tốt nhất trong project

#### 4.4. Threshold Tuning

Nội dung nên có:

- Model trả về xác suất fraud.
- Threshold mặc định thường là `0.50`.
- Vì dữ liệu mất cân bằng, threshold mặc định chưa chắc tối ưu.
- Project tune threshold theo F1-score trên validation split.
- HistGradientBoosting tuned F1 dùng threshold `0.80`.

Giải thích trade-off:

- Threshold thấp: recall cao hơn, nhưng false positive nhiều hơn.
- Threshold cao: cảnh báo ít hơn, nhưng có thể bỏ sót fraud.
- F1 giúp cân bằng precision và recall.

### 5. Kết Quả Thực Nghiệm Và Đánh Giá

Chương này trình bày metric, bảng kết quả, biểu đồ và phân tích.

#### 5.1. Các Metric Đánh Giá

Nội dung nên có:

- Accuracy: tỷ lệ đoán đúng tổng thể, nhưng không đủ với dữ liệu mất cân bằng.
- Precision: trong các giao dịch bị báo fraud, bao nhiêu giao dịch thật sự fraud.
- Recall: trong toàn bộ fraud thật, model bắt được bao nhiêu.
- F1-score: cân bằng precision và recall.
- ROC-AUC: khả năng phân biệt hai lớp trên nhiều threshold.
- PR-AUC: đặc biệt quan trọng với dữ liệu mất cân bằng.

Thông điệp:

> Vì fraud là lớp thiểu số, Recall, F1-score và PR-AUC quan trọng hơn Accuracy.

#### 5.2. So Sánh Kết Quả Các Mô Hình

Bảng metric chính:

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| HistGradientBoosting tuned | 0.3557 | 0.9245 | 0.5137 | 0.9976 | 0.8701 |
| Random Forest | 0.1802 | 0.9147 | 0.3011 | 0.9899 | 0.7441 |
| Logistic Regression tuned | 0.1577 | 0.6699 | 0.2553 | 0.9055 | 0.1189 |

Nhận xét nên có:

- Logistic Regression là baseline, kết quả thấp nhất.
- Random Forest cải thiện recall rõ rệt.
- HistGradientBoosting tốt nhất ở F1-score, Recall, ROC-AUC và PR-AUC.
- PR-AUC của HistGradientBoosting đạt `0.8701`, rất mạnh với dữ liệu fraud mất cân bằng.

#### 5.3. Phân Tích Mô Hình Tốt Nhất

Mô hình tốt nhất:

```text
HistGradientBoosting tuned F1
Threshold = 0.80
Precision = 0.3557
Recall = 0.9245
F1-score = 0.5137
ROC-AUC = 0.9976
PR-AUC = 0.8701
```

Ý nghĩa:

- Bắt được khoảng `92.45%` giao dịch fraud.
- F1-score cao hơn nhiều so với Random Forest.
- PR-AUC cao chứng minh model xử lý tốt lớp thiểu số.
- Precision vẫn chưa tuyệt đối cao, nhưng đã cải thiện rất mạnh so với fraud rate gốc `0.5789%`.

#### 5.4. Biểu Đồ Và Nhận Xét

Hình nên chèn:

- Confusion matrix của HistGradientBoosting.
- ROC curve.
- Precision-Recall curve.
- Model metrics comparison.

Nội dung nên giải thích:

- Confusion matrix cho thấy số fraud bắt được và số fraud bỏ sót.
- ROC curve thể hiện khả năng phân biệt hai lớp.
- PR curve quan trọng hơn trong bài toán fraud vì dữ liệu mất cân bằng.
- Biểu đồ comparison cho thấy HistGradientBoosting vượt các model còn lại.

### 6. Kết Luận Và Hướng Phát Triển

#### 6.1. Kết Luận

Nội dung nên có:

- Project đã hoàn thành pipeline Big Data + Machine Learning.
- Có EDA, preprocessing, Spark processing, modeling và evaluation.
- Spark được dùng để xử lý aggregation dữ liệu lớn.
- HistGradientBoosting là mô hình tốt nhất.
- Mô hình đạt:
  - Recall `92.45%`
  - F1-score `0.5137`
  - PR-AUC `0.8701`
- Project phù hợp làm hệ thống flag giao dịch rủi ro để kiểm tra tiếp.

#### 6.2. Hướng Phát Triển

Nội dung nên có:

- Thử LightGBM, XGBoost hoặc CatBoost.
- Hyperparameter tuning sâu hơn.
- Cost-sensitive learning để tối ưu theo chi phí fraud/false positive.
- Batch scoring pipeline bằng Spark.
- Real-time streaming với Spark Streaming hoặc Kafka.
- Explainability bằng SHAP để giải thích vì sao giao dịch bị flag.
- Xây dựng dashboard giám sát fraud.

### 7. Nguồn Dữ Liệu Và Tài Liệu Tham Khảo

Nội dung nên có:

- Link dataset Kaggle.
- Tài liệu Apache Spark/PySpark.
- Tài liệu scikit-learn.
- Tài liệu về fraud detection, imbalanced classification, ROC-AUC, PR-AUC nếu cần.

Gợi ý ghi:

- Kaggle Credit Card Transactions Fraud Detection Dataset.
- Apache Spark Documentation.
- scikit-learn Documentation.
- Các tài liệu/slide môn học về Big Data 5V và Machine Learning.

## 12. Gợi Ý Cấu Trúc Slide

Slide nên bám sát cấu trúc báo cáo, nhưng ngắn hơn. Mỗi slide chỉ nên có một thông điệp chính.

### Slide 1: Title

Tiêu đề:

```text
Fraud Detection System using Big Data Analytics
```

Nên có:

- Tên môn học
- Tên nhóm/thành viên
- Giảng viên
- Ngày trình bày

### Slide 2: Problem Statement

Thông điệp: vì sao bài toán quan trọng.

Nội dung:

- Gian lận giao dịch gây thiệt hại tài chính.
- Dữ liệu giao dịch lớn và phát sinh liên tục.
- Fraud rất hiếm, nên khó phát hiện.
- Cần hệ thống hỗ trợ phát hiện giao dịch rủi ro.

### Slide 3: Dataset Overview

Thông điệp: dữ liệu lớn và mất cân bằng.

Nội dung:

- Nguồn: Kaggle Fraud Detection.
- Train: `1,296,675` giao dịch.
- Test: `555,719` giao dịch.
- Target: `is_fraud`.
- Fraud rate train: `0.5789%`.

Hình nên chèn:

- `f1.1_target_distribution.png`

### Slide 4: Big Data 5V

Thông điệp: dataset phù hợp bối cảnh Big Data.

Nội dung:

- Volume: hơn 1.85 triệu giao dịch.
- Velocity: giao dịch phát sinh liên tục.
- Variety: amount, time, merchant, category, location, customer profile.
- Veracity: cần xử lý dữ liệu và loại bỏ định danh.
- Value: hỗ trợ giảm thiệt hại fraud.

### Slide 5: System Pipeline

Thông điệp: project có pipeline đầy đủ.

Vẽ flow:

```text
Raw Data -> EDA -> Preprocessing -> Spark Processing -> Modeling -> Evaluation -> Report
```

Nên nói:

- Dữ liệu raw được phân tích, xử lý, đưa qua Spark aggregation, sau đó train model và đánh giá.

### Slide 6: EDA Highlights

Thông điệp: EDA chỉ ra fraud là bài toán khó và có pattern theo nhiều chiều.

Nội dung:

- Fraud chiếm dưới 1%.
- Amount bị lệch phải.
- Fraud rate khác nhau theo category/merchant.
- Fraud thay đổi theo thời gian giao dịch.

Hình nên chèn:

- `f1.2_amount_distribution_log_scale.png`
- `f1.4_category_transaction_count_and_fraud_rate.png`
- `f1.6_fraud_rate_by_hour.png`

### Slide 7: Preprocessing & Feature Engineering

Thông điệp: raw data được biến thành feature sạch cho model.

Nội dung:

- Time features: hour, day of week, month, weekend flag.
- Customer age.
- Customer-merchant distance.
- One-hot encoding.
- Frequency encoding.
- Numeric scaling.
- Remove direct identifiers.

### Slide 8: Apache Spark Processing

Thông điệp: project có phần xử lý dữ liệu lớn.

Nội dung:

- Dùng Spark DataFrame đọc dữ liệu giao dịch.
- Fraud by category/state/merchant/hour.
- Amount statistics by target.
- Xuất bảng kết quả vào `reports/tables`.
- Spark có thể scale lên cluster/cloud.

### Slide 9: Pandas vs Spark / Big Data Value

Thông điệp: Spark có ý nghĩa khi dữ liệu lớn và cần mở rộng.

Nội dung:

- So sánh aggregation theo category.
- Pandas phù hợp dữ liệu nhỏ/vừa.
- Spark phù hợp xử lý song song và phân tán.
- Trong project, Spark giúp tạo insight và chứng minh khả năng Big Data processing.

Nếu muốn slide ngắn hơn, có thể gộp Slide 8 và Slide 9.

### Slide 10: Modeling Strategy

Thông điệp: cách train model được thiết kế cho dữ liệu mất cân bằng.

Nội dung:

- Binary classification: fraud/non-fraud.
- Sampling:
  - giữ toàn bộ fraud
  - sample non-fraud theo tỷ lệ `5:1`
- Models:
  - Logistic Regression
  - Random Forest
  - HistGradientBoosting
- Evaluation trên full test set.

### Slide 11: Threshold Tuning & Metrics

Thông điệp: không dùng accuracy đơn thuần.

Nội dung:

- Accuracy dễ gây ảo giác vì fraud rất hiếm.
- Recall: giảm bỏ sót fraud.
- Precision: giảm cảnh báo nhầm.
- F1-score: cân bằng precision/recall.
- PR-AUC: rất quan trọng với imbalanced data.
- Threshold tuning theo F1-score.

### Slide 12: Model Results

Thông điệp: HistGradientBoosting là model tốt nhất.

Bảng kết quả:

| Model | Precision | Recall | F1 | PR-AUC |
|---|---:|---:|---:|---:|
| HistGradientBoosting | 0.3557 | 0.9245 | 0.5137 | 0.8701 |
| Random Forest | 0.1802 | 0.9147 | 0.3011 | 0.7441 |
| Logistic Regression | 0.1577 | 0.6699 | 0.2553 | 0.1189 |

Nhấn:

```text
HistGradientBoosting:
Recall = 0.9245
F1-score = 0.5137
PR-AUC = 0.8701
```

### Slide 13: Evaluation Visualization

Thông điệp: kết quả được kiểm chứng bằng biểu đồ.

Hình nên chèn:

- Confusion matrix của HistGradientBoosting.
- ROC curve.
- Precision-Recall curve.
- Model metrics comparison.

Nên nói:

- PR curve quan trọng vì fraud là lớp thiểu số.
- Confusion matrix giúp nhìn rõ số fraud bắt được và bỏ sót.

### Slide 14: Key Findings

Thông điệp: kết quả chính của project.

Nội dung:

- Fraud detection là bài toán mất cân bằng mạnh.
- Logistic Regression là baseline.
- Random Forest đã đạt recall tốt.
- HistGradientBoosting cải thiện mạnh cả F1-score và PR-AUC.
- Recall cao phù hợp mục tiêu giảm bỏ sót fraud.

### Slide 15: Conclusion & Future Work

Thông điệp: chốt đóng góp và hướng phát triển.

Kết luận:

- Hoàn thành pipeline Big Data + ML.
- Có Spark processing, EDA, preprocessing, modeling, evaluation.
- HistGradientBoosting là model tốt nhất.
- Recall đạt `92.45%`, PR-AUC đạt `0.8701`.

Hướng phát triển:

- LightGBM/XGBoost/CatBoost.
- Hyperparameter tuning sâu hơn.
- Batch scoring bằng Spark.
- Real-time streaming.
- Explainability bằng SHAP.

## 13. Câu Hỏi Có Thể Bị Hỏi Và Cách Trả Lời

### Tại sao accuracy cao mà vẫn cần metric khác?

Trả lời:

> Vì fraud chỉ chiếm 0.58%, model đoán toàn bộ là non-fraud vẫn có accuracy rất cao. Do đó nhóm dùng Recall, F1-score và PR-AUC để đánh giá tốt hơn trên lớp fraud.

### Tại sao precision chưa quá cao?

Trả lời:

> Vì fraud là lớp cực kỳ hiếm. Nhóm ưu tiên recall để giảm bỏ sót fraud. Precision 0.3557 nghĩa là model đã cải thiện đáng kể so với fraud baseline 0.58%, và hệ thống phù hợp để flag giao dịch rủi ro cho bước kiểm tra tiếp theo.

### Model nào tốt nhất?

Trả lời:

> HistGradientBoosting tuned F1 là tốt nhất, vì đạt Recall 0.9245, F1-score 0.5137, ROC-AUC 0.9976 và PR-AUC 0.8701, vượt Random Forest và Logistic Regression.

### Spark dùng để làm gì?

Trả lời:

> Spark được dùng cho xử lý và aggregation dữ liệu lớn: fraud theo category, state, merchant, hour, amount statistics và so sánh với Pandas. Đây là phần thể hiện khả năng Big Data processing của pipeline.

### Dự án có thể triển khai thực tế không?

Trả lời:

> Có thể phát triển thành batch scoring pipeline: mỗi ngày lấy giao dịch mới, chạy preprocessing giống train, dùng model đã lưu để scoring, sau đó flag giao dịch có xác suất fraud cao cho hệ thống kiểm tra.

## 14. Ba Ý Sống Còn Cần Nắm

1. Dữ liệu cực mất cân bằng, nên không được khoe accuracy đơn thuần.
2. Pipeline đầy đủ Big Data + ML, không phải chỉ train model.
3. HistGradientBoosting là điểm sáng, đạt Recall `92.45%`, F1 `0.5137`, PR-AUC `0.8701`.
