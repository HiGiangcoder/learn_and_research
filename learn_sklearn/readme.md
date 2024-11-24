<h1>Learning <b>Scikit Learn</b> for Machine Learning<h1>

<h2> Table of Contents <h2>

<!-- TOC -->
- [1. Introduction to scikit-learn](#1-introduction-to-scikit-learn)
- [2. Basic Machine Learning Workflow(quy trinh lam viec) with scikit-learn](#2-basic-machine-learning-workflowquy-trinh-lam-viec-with-scikit-learn)
  - [2.1: Supervised Learning](#21-supervised-learning)
  - [2.2: Unsupervised Learning](#22-unsupervised-learning)
    - [More about Linear regression](#more-about-linear-regression)
- [3. Logistic Regression](#3-logistic-regression)
- [4. K-Nearest Neighbor](#4-k-nearest-neighbor)
- [5. Decision Tree](#5-decision-tree)
- [6. Common metrics for classification models (e.g., confusion matrix, accuracy, precision, recall, f1-score)](#6-common-metrics-for-classification-models-eg-confusion-matrix-accuracy-precision-recall-f1-score)
- [7. Try different the model's hyperparameters to improve performance](#7-try-different-the-models-hyperparameters-to-improve-performance)
- [8. Repeat the process with different models and compare their performance](#8-repeat-the-process-with-different-models-and-compare-their-performance)

<!-- \TOC -->

## 1. Introduction to scikit-learn
- [link](https://www.tutorialspoint.com/scikit_learn/scikit_learn_introduction.htm)

- Scikit-learn (Sklearn) is the most useful and robust(manh me) library for machine learning in Python. It provides a selection of efficient(co hieu qua) tools for machine learning and statistical(thong ke) modeling including classification(phan loai), regression, clustering(phan cum) and dimensionality_reduction(giam kich thuoc) via a consistence(tinh nhat quan) interface in Python. This library, which is largely written in Python, is built upon NumPy, SciPy and Matplotlib.

## 2. Basic Machine Learning Workflow(quy trinh lam viec) with scikit-learn
### 2.1: Supervised Learning

- [link](https://www.tutorialspoint.com/scikit_learn/scikit_learn_estimator_api.htm)

```python
%matplotlib inline
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

iris = sns.load_dataset('iris')
X_iris = iris.drop('species', axis = 1)
X_iris.shape
y_iris = iris['species']
y_iris.shape

rng = np.random.RandomState(35)
x = 10 * rng.rand(40)
y = 2 * x - 1 + rng.randn(40)
plt.scatter(x, y);

model = LinearRegression(fit_intercept=True)
model
X = x[:, np.newaxis]
X.shape

model.fit(X, y)
model.coef_
model.intercept_

xfit = np.linspace(-1, 11)
Xfit = xfit[:, np.newaxis]
yfit = model.predict(Xfit)
plt.scatter(x, y)
plt.plot(xfit, yfit);
```
Câu lệnh `LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)` được sử dụng để tạo một đối tượng **LinearRegression** từ thư viện **scikit-learn**. Đây là một mô hình hồi quy tuyến tính được dùng để dự đoán giá trị liên tục. 

**Ý nghĩa từng tham số:**

1. **`copy_X=True`**:
   - Quy định có sao chép dữ liệu đầu vào hay không.
   - Nếu `True`, dữ liệu đầu vào (ma trận đặc trưng `X`) sẽ được sao chép trước khi thực hiện bất kỳ thao tác nào, nhằm bảo toàn dữ liệu gốc.
   - Nếu `False`, dữ liệu gốc có thể bị thay đổi trong quá trình xử lý.

2. **`fit_intercept=True`**:
   - Quy định có tính toán **hệ số chặn** (intercept) trong mô hình hay không.
   - Nếu `True`, mô hình sẽ thêm một hệ số chặn \( b \) trong phương trình:
     \[
     y = wX + b
     \]
   - Nếu `False`, mô hình sẽ không tính hệ số chặn và giả định dữ liệu đã được chuẩn hóa (normalize) sao cho đường hồi quy đi qua gốc tọa độ \( (0,0) \).

3. **`n_jobs=None`**:
   - Xác định số luồng (threads) CPU được sử dụng để tính toán.
   - Nếu `n_jobs=None`, mô hình sử dụng mặc định một luồng duy nhất.
   - Nếu là một số nguyên (ví dụ: `n_jobs=4`), mô hình sẽ sử dụng 4 luồng.
   - Nếu `n_jobs=-1`, tất cả các luồng CPU có sẵn sẽ được sử dụng để tăng tốc tính toán.

4. **`normalize=False`** (deprecated từ scikit-learn 0.24):
   - Quy định có chuẩn hóa dữ liệu đầu vào hay không.
   - Nếu `True`, dữ liệu đầu vào sẽ được chuẩn hóa sao cho tất cả các đặc trưng có độ lớn trung bình bằng 0 và độ lệch chuẩn bằng 1.
   - Nếu `False`, dữ liệu được sử dụng như nguyên bản.
   - **Lưu ý**: Từ phiên bản scikit-learn 0.24 trở đi, tham số này không còn được khuyến khích sử dụng, và bạn nên chuẩn hóa dữ liệu bằng cách sử dụng `StandardScaler` hoặc các công cụ khác trước khi đưa vào mô hình.


```python
from sklearn.linear_model import LinearRegression
import numpy as np

# Dữ liệu mẫu
X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
y = np.dot(X, np.array([1, 2])) + 3  # y = 1*x1 + 2*x2 + 3

# Tạo mô hình hồi quy tuyến tính
model = LinearRegression(copy_X=True, fit_intercept=True, n_jobs=-1)

# Huấn luyện mô hình
model.fit(X, y)

# Hệ số của mô hình
print("Hệ số hồi quy(anh huong cua features):", model.coef_)
print("Hệ số chặn (intercept = bias):", model.intercept_)
```

---

**Kết quả:**

```
Hệ số hồi quy: [1. 2.]
Hệ số chặn (intercept): 3.0
```

### 2.2: Unsupervised Learning 
- Here, as an example of this process we are taking common case of reducing the dimensionality of the Iris dataset so that we can visualize it more easily. For this example, we are going to use principal component analysis (PCA), a fast-linear dimensionality reduction technique.

- Like the above given example, we can load and plot the random data from iris dataset. After that we can follow the steps as below −

```python
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

iris = sns.load_dataset('iris')
X_iris = iris.drop('species', axis = 1)
X_iris.shape
y_iris = iris['species']
y_iris.shape
rng = np.random.RandomState(35)
x = 10*rng.rand(40)
y = 2*x-1+rng.randn(40)
plt.scatter(x,y);

from sklearn.decomposition import PCA

model = PCA(n_components=2)
model
model.fit(X_iris)
X_2D = model.transform(X_iris)
iris['PCA1'] = X_2D[:, 0]
iris['PCA2'] = X_2D[:, 1]
sns.lmplot("PCA1", "PCA2", hue='species', data=iris, fit_reg=False);
```

![image](/learn_sklearn/image.png)

#### More about Linear regression

- [link](https://www.tutorialspoint.com/scikit_learn/scikit_learn_linear_regression.htm)

```python
import numpy as np
from sklearn.linear_model import LinearRegression
X = np.array([[1,1],[1,2],[2,2],[2,3]])
y = np.dot(X, np.array([1,2])) + 3
regr = LinearRegression(
   fit_intercept = True, normalize = True, copy_X = True, n_jobs = 2
).fit(X,y)
regr.predict(np.array([[3,5]]))
regr.score(X,y)
regr.coef_ # Muc do anh huong cua moi feature
regr.intercept_ # bias
```

## 3. Logistic Regression
- [link](https://www.geeksforgeeks.org/ml-logistic-regression-using-python/?ref=header_outind)

```python
# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
```

```python
# Load the diabetes dataset
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target

# Convert the target variable to binary (1 for diabetes, 0 for no diabetes)
y_binary = (y > np.median(y)).astype(int)
```

```python
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
	X, y_binary, test_size=0.2, random_state=42)
```

```python
# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

```
# Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)
```

```python
# evaluate the model
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
```

```python
# Visualize the decision boundary with accuracy information
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_test[:, 2], y=X_test[:, 8], hue=y_test, palette={
				0: 'blue', 1: 'red'}, marker='o')
plt.xlabel("BMI")
plt.ylabel("Age")
plt.title("Logistic Regression Decision Boundary\nAccuracy: {:.2f}%".format(
	accuracy * 100))
plt.legend(title="Diabetes", loc="upper right")
plt.show()
```
![image2](/learn_sklearn/image2.png)

```python
# Plot ROC Curve
y_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve\nAccuracy: {:.2f}%'.format(accuracy * 100))
plt.legend(loc="lower right")
plt.show()
```
![image3](/learn_sklearn/image3.png)

## 4. K-Nearest Neighbor
- [link](https://www.geeksforgeeks.org/ml-implementation-of-knn-classifier-using-sklearn/)

## 5. Decision Tree
- [link](https://www.tutorialspoint.com/scikit_learn/scikit_learn_decision_trees.htm)

## 6. Common metrics for classification models (e.g., confusion matrix, accuracy, precision, recall, f1-score)
- [link](https://towardsdatascience.com/20-popular-machine-learning-metrics-part-1-classification-regression-evaluation-metrics-1ca3e282a2ce)

## 7. Try different the model's hyperparameters to improve performance 
- link does not exist

## 8. Repeat the process with different models and compare their performance 
- link does not exist