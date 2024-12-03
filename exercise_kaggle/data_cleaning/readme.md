<h1> Exercise Kaggle: Cleaning data</h1>

<h2> Table of Contents </h2>

<!-- TOC -->
- [I. Handling missing value](#i-handling-missing-value)
  - [1. Percent missing data](#1-percent-missing-data)
  - [2. Drop missing values: Rows](#2-drop-missing-values-rows)
  - [3. Drop missing values: Columns (cal num of drop col)](#3-drop-missing-values-columns-cal-num-of-drop-col)
  - [4. Fill in missing values automatically(mot cach tu dong)](#4-fill-in-missing-values-automaticallymot-cach-tu-dong)
- [II. Scaling and Normalization](#ii-scaling-and-normalization)
  - [1. Scaling:](#1-scaling)
  - [2. Normalization:](#2-normalization)
- [III Parsing(phan tich cu phap) Dates](#iii-parsingphan-tich-cu-phap-dates)
- [IV. Character Encodings](#iv-character-encodings)
- [V. Inconsistent(khong nhat quan) data entry(coongr vaof)](#v-inconsistentkhong-nhat-quan-data-entrycoongr-vaof)
<!-- /TOC -->
## I. Handling missing value
### 1. Percent missing data

```python
missing_value_count = sf_permits.isnull().sum()

total_cells = np.product(sf_permits.shape)
total_missing = missing_value_count.sum()


percent_missing = total_missing / total_cells * 100
```

### 2. Drop missing values: Rows
```python
sf_permits.dropna(axis=0, how='any',inplace=True, subset=None)
```

### 3. Drop missing values: Columns (cal num of drop col)
```python
# remove all columns with at least one missing value
sf_permits_with_na_dropped = sf_permits.dropna(axis=1)

# calculate number of dropped columns
sf_permits_with_na_dropped = sf_permits.dropna(axis=1, how='any', inplace=False, subset=None)

dropped_columns = len(sf_permits.columns) - sf_permits_with_na_dropped.shape[1]
# the two methods yield(mang la.i) the same result
```

### 4. Fill in missing values automatically(mot cach tu dong)
```python
sf_permits_with_na_imputed = sf_permits.fillna(method='bfill').fillna(0)
```

## II. Scaling and Normalization 
- Import library.
```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats
from mlxtend.preprocessing import minmax_scaling

np.random.seed(0)
```

### 1. Scaling:
- This means that you're transforming your data so that it fits within(owr trong) a specific(cu the?) scale, like 0-100 or 0-1. You want to scale data when you're using methods based on measures(ddo, can nhac) of how far apart(rieng biet) data points are, like support vector machines (SVM) or k-nearest neighbors (KNN). With these algorithms, a change of "1" in any numeric feature is given the same importance(tam quan trong).

```python
# generate 1000 data points randomly drawn from an exponential distribution
original_data = np.random.exponential(size=1000)

# mix-max scale the data between 0 and 1
scaled_data = minmax_scaling(original_data, columns=[0])

# plot both together to compare
fig, ax = plt.subplots(1, 2, figsize=(15, 3))
sns.histplot(original_data, ax=ax[0], kde=True, legend=False)
ax[0].set_title("Original Data")
sns.histplot(scaled_data, ax=ax[1], kde=True, legend=False)
ax[1].set_title("Scaled data")
plt.show()
```

1. **`fig, ax = plt.subplots(1, 2, figsize=(15, 3))`**
- **`plt.subplots`**: Tạo một lưới (grid) các biểu đồ (axes) trong một hình vẽ (figure).  
  - **`1, 2`**: Sắp xếp lưới với 1 hàng và 2 cột (tức là tạo ra 2 biểu đồ nằm cạnh nhau).
  - **`figsize=(15, 3)`**: Kích thước tổng thể của hình vẽ (figure) là rộng **15 đơn vị** và cao **3 đơn vị**.
- Kết quả:
  - **`fig`**: Đối tượng của hình vẽ toàn cục (figure).
  - **`ax`**: Danh sách các trục (axes) tương ứng với các ô (plots) trong lưới. Ở đây, **`ax`** có hai phần tử: **`ax[0]`** (biểu đồ đầu tiên) và **`ax[1]`** (biểu đồ thứ hai).

---

2. **`sns.histplot(original_data, ax=ax[0], kde=True, legend=False)`**
- **`sns.histplot`**: Hàm của Seaborn dùng để vẽ biểu đồ histogram (biểu đồ tần suất) cho dữ liệu.
- **`original_data`**: Dữ liệu đầu vào để vẽ biểu đồ.
- **`ax=ax[0]`**: Biểu đồ được vẽ trên **trục đầu tiên** trong lưới (trục nằm ở cột thứ nhất).
- **`kde=True`**: Vẽ thêm đường **KDE (Kernel Density Estimate)** lên biểu đồ histogram để biểu diễn mật độ xác suất (dạng đường cong trơn thay vì chỉ các thanh tần suất).
- **`legend=False`**: Ẩn phần chú thích (legend) trong biểu đồ.

---

<h4>Ý nghĩa tổng quát:</h4>

1. Tạo một hình vẽ với **2 ô biểu đồ** (trục).
2. Vẽ **biểu đồ phân phối dữ liệu** trên ô đầu tiên (**`ax[0]`**) bằng cách sử dụng histogram và KDE.
3. Không sử dụng chú thích trên biểu đồ.
---

![image scaled data](/exercise_kaggle/data_cleaning/image_scaled_data.png)

### 2. Normalization:
- Scaling just changes the range of your data. 
Normalization is a more radical(nguyen ly can ban) transformation. 
The point of normalization is to change your observations(su quan sat) 
so that they can be described as a normal distribution(su phan bo).

```python
# normalize the exponential data with boxcox
normalized_data = stats.boxcox(original_data)

# plot both together to compare
fig, ax=plt.subplots(1, 2, figsize=(15, 3))
sns.histplot(original_data, ax=ax[0], kde=True, legend=False)
ax[0].set_title("Original Data")
sns.histplot(normalized_data[0], ax=ax[1], kde=True, legend=False)
ax[1].set_title("Normalized data")
plt.show()
```
![image normalization data](/exercise_kaggle/data_cleaning/image_normalization_data.png)

## III Parsing(phan tich cu phap) Dates
- environment set up
```python
# modules we'll use
import pandas as pd
import numpy as np
import seaborn as sns
import datetime

# read in our data
landslides = pd.read_csv("../input/landslide-events/catalog.csv")

# set seed for reproducibility
np.random.seed(0)
```

- check data type
```python
landslides.head()
```

| ID  | Date     | Time  | Continent Code | Country Name | Country Code | State/Province | Population | City/Town      | Distance | Geolocation                     | Hazard Type | Landslide Type        | Landslide Size | Trigger  | Storm Name | Injuries | Fatalities | Source Name                   | Source Link                                                                                 |
|-----|----------|-------|----------------|--------------|--------------|----------------|------------|----------------|----------|---------------------------------|-------------|-----------------------|----------------|----------|------------|----------|------------|--------------------------------|---------------------------------------------------------------------------------------------|
| 34  | 3/2/07   | Night | NaN            | United States| US           | Virginia       | 16000      | Cherry Hill    | 3.40765  | (38.6009, -77.2682)            | Landslide   | Landslide             | Small          | Rain     | NaN        | NaN      | NaN        | NBC 4 news                   | [Link](http://www.nbc4.com/news/11186871/detail.html)                                       |
| 42  | 3/22/07  | NaN   | NaN            | United States| US           | Ohio           | 17288      | New Philadelphia | 3.33522 | (40.5175, -81.4305)            | Landslide   | Landslide             | Small          | Rain     | NaN        | NaN      | NaN        | Canton Rep.com               | [Link](http://www.cantonrep.com/index.php?ID=345054&C...)                                  |
| 56  | 4/6/07   | NaN   | NaN            | United States| US           | Pennsylvania   | 15930      | Wilkinsburg    | 2.91977  | (40.4377, -79.9160)            | Landslide   | Landslide             | Small          | Rain     | NaN        | NaN      | NaN        | The Pittsburgh Channel.com   | [Link](https://web.archive.org/web/20080423132842/http://www.thepittsburghchannel.com)     |
| 59  | 4/14/07  | NaN   | NaN            | Canada       | CA           | Quebec         | 42786      | Châteauguay    | 2.98682  | (45.3226, -73.7771)            | Landslide   | Riverbank collapse    | Small          | Rain     | NaN        | NaN      | NaN        | Le Soleil                    | [Link](http://www.hebdos.net/lsc/edition162007/article.html)                               |
| 61  | 4/15/07  | NaN   | NaN            | United States| US           | Kentucky       | 6903       | Pikeville      | 5.66542  | (37.4325, -82.4931)            | Landslide   | Landslide             | Small          | Downpour | NaN        | NaN      | 0.0        | Matthew Crawford (KGS)       | NaN                                                                                         |

```python
# print the first few rows of the date column
print(landslides['date'].head())
```

**Output**

```
0     3/2/07
1    3/22/07
2     4/6/07
3    4/14/07
4    4/15/07
Name: date, dtype: object
```

<h3>Convert our date columns to datetime</h3>

- `1/17/07` has the format `"%m/%d/%y"`
- `17-1-2007` has the format `"%d-%m-%Y"`

```python
# create a new column, date_parsed, with the parsed dates
landslides['date_parsed'] = pd.to_datetime(landslides['date'], format="%m/%d/%y")
```

```python
# print the first few rows
landslides['date_parsed'].head()
```

**Output**

```
0   2007-03-02
1   2007-03-22
2   2007-04-06
3   2007-04-14
4   2007-04-15
Name: date_parsed, dtype: datetime64[ns]
```

<h3> Select the day of the month</h3>

```python
# get the day of the month from the date_parsed column
day_of_month_landslides = landslides['date_parsed'].dt.day
day_of_month_landslides.head()
```

**Output**

```
0     2.0
1    22.0
2     6.0
3    14.0
4    15.0
Name: date_parsed, dtype: float64
```

<h3> Plot the day of the month to check the date parsing</h3>

```python
# remove na's
day_of_month_landslides = day_of_month_landslides.dropna()

# plot the day of the month
sns.distplot(day_of_month_landslides, kde=False, bins=31)
```

<b>Output</b>

```
<AxesSubplot:xlabel='date_parsed'>
```

![image date parsed](/exercise_kaggle/data_cleaning/image_date_distplot.png)

## IV. Character Encodings



## V. Inconsistent(khong nhat quan) data entry(coongr vaof)



