<h1>TensorFlow tutorial</h1>

<h2>Table of Content</h2>

<!-- TOC -->

- [Quickstart](#quickstart)

<!-- /TOC -->

# Quickstart

[link](https://www.tensorflow.org/tutorials/quickstart/beginner#load_a_dataset)

```python
import tensorflow as tf
```


### 1. Load a dataset

- **MNIST** (Modified National Institute of Standards and Technology) là một tập dữ liệu phổ biến được sử dụng trong học máy, đặc biệt là trong các bài toán nhận dạng chữ viết tay. Đây là một tập dữ liệu cổ điển dành cho các bài tập khởi đầu về mạng nơ-ron và học sâu.
- Dạng dữ liệu:
  + Ảnh đầu vào là ma trận số thực (0 -> 255) đại diện cho độ sáng của mỗi pixel (size of each picture: 28x28).
  + Nhãn là các số nguyên từ 0 -> 9

```python
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Data Scaling
x_train, x_test = x_train / 255.0, x_test / 255.0
```

### 2. Build a machine learning(deep learning) model

```python
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)), 
  ft.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])
```

- For each example, the model returns a vector of **logits** or **log-odds** scores, one for each class.

```python
predictions = model(x_train[:1]).numpy()
predictions
```

**Output:**

```
array([[ 0.68130803, -0.03935227, -0.53304887,  0.22200397, -0.3079031, -0.6267688 ,  0.43393654,  0.5691322 ,  0.31098977,  0.32141146]], dtype=float32)
```

- The `tf.nn.softmax` function converts these logits to probabilities(xac suat) for each class:

```python
tf.nn.softmax(predictions).numpy()
```

**Output:**

```python
array([[0.16339162, 0.07947874, 0.04851112, 0.10321827, 0.06076043, 0.0441712 , 0.12758444, 0.14605366, 0.11282429, 0.11400625]], dtype=float32)
```

- Define a **loss function** for training using ```losses.SparseCategoricalCrossentropy```
  
```python
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

loss_fn(y_train[:1], predictions).numpy()
```

**Output:** `3.1196823`

- Before you start training, configure(caaus hinhf) and compile the model using Keras `Model.compile`. Set the optimizer class to `adam`, set the loss to the `loss_fn` function you defined earlier, and specify a metric(thuowcs ddo) to be evaluated for the model by setting the `metrics` parameter to `accuracy`.

```python
model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
```

