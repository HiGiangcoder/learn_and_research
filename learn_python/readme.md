# Sumary python tutorial

## 1. OOP

### 1.1. Polymorphism(Tính đa hình):
### 1.2. Inheritance
### 1.3. Abstract
### 1.4. Encapsulation
### 1.5. Iterators

## 2. File handling

- đọc từng line từ file `text.txt`
```python
file = open('text.txt', 'r')

for each in file:
    print(each)
```


## 3. Exception handling

#### Diffirent types of exceptions in python
##### 1. Syntax error
    - Sai cú pháp.
##### 2. Type error
    - Sai kiểu.
##### 3. Name error
    - Không tìm thấy tên biến/hàm đó.
##### 4. Index error
    - index vượt quá dãy list, tuple, sequence.
##### 5. Key error
    - Không tìm thấy key trong dictionary.
##### 6. Value error
    - Lỗi đối số(argument) invalid.
    - such as trying convert a string to an integer when the string does not represent a valid integer.
##### 7. Attribute error
    - Thuộc tính của phương thức không được tìm thấy trong object.
    - Ví dụ như cố gắng truy cập một thuộc tính không tồn tại trong class nào đó.
##### 8. IO error
    - Lỗi xảy ra khi thao tác I/O bị lỗi.
    - Such as reading/writing a file, fails due an input/output error.
##### 9. Zero Division error
    - Lỗi do chia một số cho 0.
##### 10. Import Error
    - lỗi thư viện chưa được cài đặt nên không import được.
    - sửa lỗi bằng cách cài đặt `pip install include`

## 4. Numpy (python library)
### 4.1. Introduction

##### 1. Array 
```python
# Creating array object
arr = np.array( [[ 1, 2, 3],
                 [ 4, 2, 5]] )

# Printing type of arr object
print("Array is of type: ", type(arr))

# Printing array dimensions (axes)
print("No. of dimensions: ", arr.ndim)

# Printing shape of array
print("Shape of array: ", arr.shape)

# Printing size (total number of elements) of array
print("Size of array: ", arr.size)

# Printing type of elements in array
print("Array stores elements of type: ", arr.dtype)
```

##### 2. Using arrange()
```python
f = np.arange(0, 30, 5)
print(f)
```

##### 3. using np.linspace(l, r, n)
- **Chức năng:** tạo một array bao gồm n giá trị trong đoạn [l, r] cách đều nhau
```python
print(np.linspace(l, r, n))
```


##### 4. numpy array indexing
```python
# Python program to demonstrate
# indexing in numpy
import numpy as np

# An exemplar array
arr = np.array([[-1, 2, 0, 4],
                [4, -0.5, 6, 0],
                [2.6, 0, 7, 8],
                [3, -7, 4, 2.0]])

# Slicing array
temp = arr[:2, ::2]
print ("Array with first 2 rows and alternate" "columns(0 and 2):\n", temp)

# Integer array indexing example
temp = arr[[0, 1, 2, 3], [3, 2, 1, 0]]
print ("\nElements at indices (0, 3), (1, 2), (2, 1)," "(3, 0):\n", temp)

# boolean array indexing example
cond = arr > 0 # cond is a boolean array
temp = arr[cond]
print ("\nElements greater than 0:\n", temp)
```

##### 5. Unary operators(toán tử ngôi I)
```python
# Python program to demonstrate
# unary operators in numpy
import numpy as np

arr = np.array([[1, 5, 6],
                [4, 7, 2],
                [3, 1, 9]])

# maximum element of array
print ("Largest element is:", arr.max())
print ("Row-wise maximum elements:", arr.max(axis = 1))

# minimum element of array
print ("Column-wise minimum elements:", arr.min(axis = 0))

# sum of array elements
print ("Sum of all array elements:", arr.sum())

# cumulative sum along each row
print ("Cumulative sum along each row:\n", arr.cumsum(axis = 1))
```

## Pandas library

## Matplotlib library

## Seaborn library

