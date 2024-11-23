# Learning basic data preprocessing

## Mục lục

[1. Overview of DataCleaning](#1-overview-of-data-cleaning)

[2. One Hot Encoding](#2-one-hot-encoding)

[3. Feature engineering: Scaling, Normalization and Standardization](3-feature-engineering-scaling-normalization-and-standardization)

---

## 1 Overview of Data Cleaning

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

## 2 One Hot Encoding

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
