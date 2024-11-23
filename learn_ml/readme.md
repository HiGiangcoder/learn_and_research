<h1>Learning basic data preprocessing<h1>

<h2>Mục lục<h2>

<!-- TOC -->
- [1. Overview of Data Cleaning](#1-overview-of-data-cleaning)
- [2. One Hot Encoding](#2-one-hot-encoding)
- [3. Feature Engineering: Scaling, Normalization and Standardization](#3-feature-engineering-scaling-normalization-and-standardization)
  - [3.1 Feature Scaling](#31-feature-scaling)
  - [3.2 Normalization(chuẩn hóa)](#32-normalizationchuẩn-hóa)
  - [3.3 Standardization(tiêu chuẩn hóa)](#33-standardizationtiêu-chuẩn-hóa)
- [4. Label Encoding](#4-label-encoding)
- [5. Woking with Missing Data in Pandas](#5-woking-with-missing-data-in-pandas)
<!-- /TOC -->
---

## 1. Overview of Data Cleaning

```python
import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('titanic.csv')
df.head()
```

**Output**

```
PassengerId    Survived    Pclass    Name    Sex    Age    SibSp    Parch    Ticket    Fare    Cabin    Embarked
0    1    0    3    Braund, Mr. Owen Harris    male    22.0    1    0    A/5 21171    7.2500    NaN    S
1    2    1    1    Cumings, Mrs. John Bradley (Florence Briggs Th...    female    38.0    1    0    PC 17599    71.2833    C85    C
2    3    1    3    Heikkinen, Miss. Laina    female    26.0    0    0    STON/O2. 3101282    7.9250    NaN    S
3    4    1    1    Futrelle, Mrs. Jacques Heath (Lily May Peel)    female    35.0    1    0    113803    53.1000    C123    S
4    5    0    3    Allen, Mr. William Henry    male    35.0    0    0    373450    8.0500    NaN    S
```

```python
df.info()
```

- In ra các features là **categorical** và **numberical**.

```python
# Categorical columns
cat_col = [col for col in df.columns if df[col].dtype == 'object']
print('Categorical columns :',cat_col)
# Numerical columns
num_col = [col for col in df.columns if df[col].dtype != 'object']
print('Numerical columns :',num_col)
```

- Drop Name and Ticket Columns

```python
df1 = df.drop(columns=['Name','Ticket'])
df1.shape
```

**Output**

```
(891, 10)
```

- Cách xóa những hàng(data) tại cột 'Embarked' có chứa `Nan`

```python
df2 = df1.drop(columns='Cabin')
df2.dropna(subset=['Embarked'], axis=0, inplace=True)
df2.shape
```

> inplace: tại chỗ
> tác dụng: thay đổi trên chính data frame đó

- Cách không xóa giá trị `Nan` mà thay thế giá trị đó bằng trung bình cộng của feature đó

```python
# Mean imputation
df3 = df2.fillna(df2.Age.mean())
# Let's check the null values again
df3.isnull().sum()
```

## 2. One Hot Encoding

- ***One hot encoding*** là hành động chia 1 features thuộc dạng categorical(phân loại)
  thành nhiều (ở đây là `số loại - 1`) features dạng numberical(loại số).
- **Lý do:** để tránh khi train model sẽ ra kết quả không chính xác.
- Dưới đây là cách *one hot encoding* bằng **pandas**

```python
import pandas as pd
# from sklearn.preprocessing import OneHotEncoder

data = {
    'Employee id': [10, 20, 15, 25, 30],
    'Gender': ['M', 'F', 'F', 'M', 'F'],
    'Remarks': ['Good', 'Nice', 'Good', 'Great', 'Nice']
}

# Convert the data into a pandas DataFrame

df = pd.DataFrame(data)
print(f"Original Employee Data:\n{df}\n")

# Use pd.get_dummies() to one-hot encode the categorical columns
df_pandas_encoded = pd.get_dummies(df, columns=['Gender', 'Remarks'], drop_first=True)
print(f"One-Hot Encoded Data using Pandas:\n{df_pandas_encoded}\n")
```

**Output**

```
Original Employee Data:
   Employee id Gender Remarks
0           10      M    Good
1           20      F    Nice
2           15      F    Good
3           25      M   Great
4           30      F    Nice

One-Hot Encoded Data using Pandas:
   Employee id  Gender_M  Remarks_Great  Remarks_Nice
0           10      True          False         False
1           20     False          False          True
2           15     False          False         False
3           25      True           True         False
4           30     False          False          True
```

## 3. Feature Engineering: Scaling, Normalization and Standardization
- Khi xử lý dữ liệu trước khi train data, cần phải chuẩn hóa.
- Nói cho đơn giản thì đưa giá trị về [-1, 1] tránh tràn số,...
- Giúp cho train nhanh, chuẩn hơn.


```python
import pandas as pd
df = pd.read_csv('SampleFile.csv')
print(df.head())
```

```
   LotArea  MSSubClass
0     8450          60
1     9600          20
2    11250          60
3     9550          70
4    14260          60
```
### 3.1 Feature Scaling
1. Absolute Maximum Scaling
```python 
max_vals = np.max(np.abs(df))
print((df - max_vals) / max_vals)
```
- Công thức cụ thể: $X_{scaled} = \frac{X_i - max(|X_i|)}{max(|X_i|)}$
- Với cách trên, data sẽ nằm trong đoạn [-1, 1] nhưng tôi không thích cách này lắm, đơn giản vì nó không được tốt cho một số data có đoạn [min-max] có mid lệch ra xa khỏi điểm số 0.

2. Min-Max Scaling
```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_data, columns=df.columns)
scaled_df.head()
```
- công thức cụ thể: $X_{scaled} = \frac{X_i - X_{min}}{X_{max}-X_{min}}$
- Đây là cách tốt hơn, nó sẽ đưa giá trị về trong đoạn [0-1]
- Thay vì tính theo công thức, ta dùng hàm MinMaxScaler từ thư viện sklearn 

### 3.2 Normalization(chuẩn hóa)
```python
from sklearn.preprocessing import Normalizer

scaler = Normalizer()
scaled_data = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_data, columns=df.columns)
print(scaled_df.head())
```

- Formula: $X_{scaled} = \frac{X_i - X_{mean}}{X_{max} - X_{min}}$
- Có gia' tri. tu` [-1, 1]

### 3.3 Standardization(tiêu chuẩn hóa)

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_data, columns=df.columns)
print(scaled_df.head())
```

- Formula: $X_{scaled} = \frac{X_i - X_{mean}}{\sigma}$
- with $\sigma$ as the standard deviation(do lech chuan). 

## 4. Label Encoding

- In machine learning projects, we usually deal with datasets having different categorical columns where some columns have their elements in the ordinal variable category for e.g a column income level having elements as low, medium, or high in this case we can replace these elements with 1,2,3. where 1 represents `low`  2  `medium`  and 3` high`. Through this type of encoding, we try to preserve the meaning of the element where higher weights are assigned to the elements having higher priority.

- Label Encoding is a technique that is used to convert categorical columns into numerical ones so that they can be fitted by machine learning models which only take numerical data. It is an important pre-processing step in a machine-learning project.

- Suppose we have a column Height in some dataset that has elements as Tall, Medium, and short. To convert this categorical column into a numerical column we will apply label encoding to this column. After applying label encoding, the Height column is converted into a numerical column having elements 0,1, and 2 where 0 is the label for tall, 1 is the label for medium, and 2 is the label for short height.

| Height | Height |
| ------ | ------ |
| Tail | 0 |
| Medium | 1 |
| Short | 2 |

```python
# Import libraries  
import numpy as np 
import pandas as pd 
  
# Import dataset 
df = pd.read_csv('../../data/Iris.csv') 
  
df['species'].unique() 
```

**Output**
```
array(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], dtype=object)
```

- After applying Label Encoding with LabelEncoder() our categorical value will replace with the numerical value[int].

```python
# Import label encoder 
from sklearn import preprocessing 

# label_encoder object knows 
# how to understand word labels. 
label_encoder = preprocessing.LabelEncoder() 

# Encode labels in column 'species'. 
df['species']= label_encoder.fit_transform(df['species']) 

df['species'].unique() 
```

**Output**
```
array([0, 1, 2], dtype=int64)
```

## 5. Woking with Missing Data in Pandas