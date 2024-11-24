<h1>Learning <b>Scikit Learn</b> for Machine Learning<h1>

<h2> Table of Contents <h2>

<!-- TOC -->
- [1. Introduction to scikit-learn](#1-introduction-to-scikit-learn)
- [2. Basic Machine Learning Workflow(quy trinh lam viec) with scikit-learn](#2-basic-machine-learning-workflowquy-trinh-lam-viec-with-scikit-learn)
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

## 3. Logistic Regression
- [link](https://www.tutorialspoint.com/scikit_learn/scikit_learn_linear_regression.htm)

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