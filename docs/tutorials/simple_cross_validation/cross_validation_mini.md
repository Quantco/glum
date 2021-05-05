---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.11.2
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Cross Validation: Mini Tutorial with Boston Housing Data

This tutorial shows how to use cross validation with `quantcore.glm` using the [sklearn boston housing dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html).

*Note:* if you wish to explore this dataset further, there are a handful of resources online. For example, [this blog](https://medium.com/@amitg0161/sklearn-linear-regression-tutorial-with-boston-house-dataset-cde74afd460a). 

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from quantcore.glm import GeneralizedLinearRegressorCV
```

## Load the data

```python
boston = datasets.load_boston()
df_bos = pd.DataFrame(boston.data, columns = boston.feature_names)
df_bos['PRICE'] = boston.target
df_bos = df_bos[df_bos['PRICE'] <= 40] # remove outliers
df_bos.head()
```

## Fit model

We fit our `GeneralizedLinearRegressorCV` model using typical regularized least squares (Normal family). As the name implies, the best model is selected by cross-validation.

Some important parameters:

- **alphas**: For each model, `alpha` is the constant multiplying penalty termdetermines regularization strength. For `GeneralizedLinearRegressorCV()`, `alphas` is list of alphas for which to compute the models. If `None`, (preferred) the alphas are set automatically. The best value is chosen and stored as `self.alpha_`
- **l1_ratio**: For each model, the `l1_ratio` is the elastic net mixing parameter (`0 <= l1_ratio <= 1`). For `l1_ratio = 0`, the penalty is an L2 penalty. ``For l1_ratio = 1``, it is an L1 penalty.  For ``0 < l1_ratio < 1``, the penalty is a combination of L1 and L2. For `GeneralizedLinearRegressorCV()`, if you pass ``l1_ratio`` as an array, the `fit` method will choose the best value of `l1_ratio` and store it as `self.l1_ratio_`

```python
X = df_bos[["CRIM", "ZN", "CHAS", "NOX", "RM", "AGE", "TAX", "B", "LSTAT"]]
y = df_bos["PRICE"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state=5)

glmcv = GeneralizedLinearRegressorCV(
    family='normal',
    alphas=None,
    l1_ratio=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
    fit_intercept=True,
    max_iter=150
)
glmcv.fit(X_train, y_train)
print(f"Chosen alpha:    {glmcv.alpha_}")
print(f"Chosen l1 ratio: {glmcv.l1_ratio_}")
```

## Test

```python
print(f"Train RMSE: {mean_squared_error(glmcv.predict(X_train), y_train, squared=False)}")
print(f"Test  RMSE: {mean_squared_error(glmcv.predict(X_test), y_test, squared=False)}")
```
