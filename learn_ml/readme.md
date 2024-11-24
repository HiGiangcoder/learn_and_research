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
- [5. Woking with Missing Data in Pandas:](#5-woking-with-missing-data-in-pandas)
- [6. Imputing Missing Data with Simple and Advanced Techniques](#6-imputing-missing-data-with-simple-and-advanced-techniques)
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

## 5. Woking with Missing Data in Pandas:


In pandas missing data is represented by two value:

- `None`: none is a Python singleton(đơn đoc) object that is often used for missing data in Python code.
- `NaN`: NaN(an acronym(tu viet tat) for Not a Number), is a special floating-point value recognized(duo.c cong nha.n) by all systems that uuse the standard IEEE floating-point represention.

```python
isnull()

notnull()

dropna()

fillna()

replace()

interpolate()
```

1. Checking missing data using `isnull()` and `notnull()`
```python
import pandas as pd 
data = pd.read_csv(&quot;employees.csv&quot;) 
  
# creating bool series True for NaN values 
bool_series = pd.isnull(data[&quot;Gender&quot;]) 
  
# filtering data 
# displaying data only with Gender = NaN 
data[bool_series]
```

- **Output:** As shown in the output image, only the rows having Gender = NULL are displayed.

2. Filling missing values using `fullna()`, `replace()` and `interpolate()`
2.1 Filling a null value using `fullna()`:
2.1.1. Filling null values with a single value
```python
import pandas as pd
import numpy as np

# dictionary of lists
dict = {'First Score':[100, 90, np.nan, 95],
        'Second Score': [30, 45, 56, np.nan],
        'Third Score':[np.nan, 40, 80, 98]}

# creating a dataframe from dictionary
df = pd.DataFrame(dict)

# filling missing value using fillna()  
df.fillna(0)
```

**Output:**
|     | First Score | Second Score | Third Score |
| --- | ---         | ---          | ---         |
| 0   | 100.0       | 30.0         | 0.0         |
| 1   | 90.0        | 45.0         | 40.0        |
| 2   | 0.0         | 56.0         | 80.0        |
| 3   | 95.0        | 0.0          | 98.0        |


2.1.2. filling a missing value with previous ones  
```python
df.fillna(method ='pad')
```

2.1.3. filling  null value with the next ones
```python
df.fillna(method ='bfill')
```

2.2. Filling a null values using `replace()` method
```python
data = pd.read_csv(&quot;employees.csv&quot;) 

# will replace  Nan value in dataframe with value -99  
data.replace(to_replace = np.nan, value = -99)
```

2.3 With `interpolate()` method
interpolate(Xen va`o): trung binh cong cua 2 so ben canh
```python
# to interpolate the missing values 
df.interpolate(method ='linear', limit_direction ='forward')
```

3. Dropping missing values using `dropna()`
```python
# importing pandas as pd
import pandas as pd

# importing numpy as np
import numpy as np

# dictionary of lists
dict = {'First Score':[100, 90, np.nan, 95],
        'Second Score': [30, np.nan, 45, 56],
        'Third Score':[52, 40, 80, 98],
        'Fourth Score':[np.nan, np.nan, np.nan, 65]}

# creating a dataframe from dictionary
df = pd.DataFrame(dict)

# using dropna() function  
df.dropna()
```
**Output:**
|     | First Score | Second Score | Third Score | Fourth Score |
| --- | ---         |        ---   |      ---    | ---          |
| 3   |   95.0      |      56.0    |    98       |      65.0    |

- Drop a columns
```python
# using dropna() function     
df.dropna(axis = 1)
```

## 6. Imputing Missing Data with Simple and Advanced Techniques
```python
df.dropna(axis=0, how='any', subset=None, inplace=False)
```

Đoạn code `df.dropna(axis=0, how='any', subset=None, inplace=False)` là một câu lệnh trong thư viện pandas của Python, dùng để loại bỏ các hàng hoặc cột trong một DataFrame chứa giá trị bị thiếu (NaN). Cụ thể:

- **`axis=0`**: Quy định hướng thực hiện thao tác. 
  - `axis=0`: Áp dụng trên các hàng.
  - `axis=1`: Áp dụng trên các cột.
  

- **`how='any'`**: Xác định điều kiện để loại bỏ.
  - `'any'`: Loại bỏ hàng nếu có **bất kỳ** giá trị nào trong hàng là NaN.
  - `'all'`: Loại bỏ hàng chỉ khi **tất cả** các giá trị trong hàng đều là NaN.
  

- **`subset=None`**: Xác định các cột cần kiểm tra.
  - Nếu `subset` được cung cấp (ví dụ: `['column1', 'column2']`), chỉ kiểm tra giá trị NaN trong các cột được chỉ định.
  - Nếu `subset=None`, kiểm tra toàn bộ các cột.


- **`inplace=False`**: Quy định cách thực hiện thay đổi.
  - Nếu `inplace=True`, DataFrame gốc (`df`) sẽ được thay đổi trực tiếp.
  - Nếu `inplace=False`, phương thức trả về một bản sao của DataFrame sau khi đã loại bỏ các hàng/cột.

1. Basic Imputation Techniques

   1.1 Mean and Mode Imputation

   We can use SimpleImputer function from scikit-learn to replace missing values with a fill value. SimpleImputer function has a parameter called strategy that gives us four possibilities(kha nang) to choose the imputation method:

   - `strategy='mean'` replaces missing values using the mean of the column.
   - `strategy='median'` replaces missing values using the median of the column.
   - `strategy='most_frequent'` replaces missing values using the most frequent(thuo`ng xuyen) (or mode) of the column.
   - `strategy='constant'` replaces missing values using a defined fill value.

```python
# Mean Imputation

df_mean = df.copy()
mean_imputer = SimpleImputer(strategy='mean')
df_mean['MaxSpeed'] = mean_imputer.fit_transform(df_mean['MaxSpeed'].values.reshape(-1,1))
```

   Let’s plot a scatter plot with AvgSpeed on the x-axis and MaxSpeed on the y-axis. As we know AvgSpeed column doesn’t have missing values, and we replaced missing values in the MaxSpeed column with column mean. In the plot below, green points are transformed data and blue points are original non-missing data.

```python
# Scatter plot

fig = plt.Figure()
null_values = df['MaxSpeed'].isnull()
fig = df_mean.plot(x="AvgSpeed", y='MaxSpeed', kind='scatter', c=null_values, cmap='winter', title='Mean Imputation', colorbar=False)

```

   1.2. Time Series Imputation

```python
df['MaxSpeed'][:100].plot(title="MaxSpeed", marker="o")
```

   - While loading the dataset, we defined the index with the combination of Date and StartTime columns, if that is not clear, see the Data part above.☝️

   - One way to impute missing values in a time series data is to fill them with either the last or the next observed values. Pandas have fillna() function which has method parameter where we can choose “ffill” to fill with the next observed value or “bfill” to fill with the previously observed value.

```python
# Ffill imputation
ffill_imputation = df.fillna(method='ffill')

# Plot imputed data
ffill_imp['MaxSpeed'][:100].plot(color='red', marker='o', linestyle='dotted')
df['MaxSpeed'][:100].plot(title='MaxSpeed', marker='o')
```
   
![missing values filled with next observed value](/image.png)


2. Advanced Techniques

   2.1. K-Nearest Neighbour (KNN) Imputation

   2.2. Multivariate Imputation by Chained Equation — MICE