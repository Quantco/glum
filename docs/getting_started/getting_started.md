---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.12.0
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Getting Started: fitting a Lasso model 

Welcome to `quantcore.glm`! Generalized linear models (GLMs) are core statistical tools that include many common methods like least-squares regression, Poisson regression, and logistic regression as special cases. At QuantCo, we have developed `quantcore.glm`, a fast Python-first GLM library. 

The purpose of this tutorial is to show the basics of `quantcore.glm`. It assumes a working knowledge of python, regularized linear models, and machine learning. The API is very similar to sklearn. After all, `quantcore.glm` is based on a fork of scikit-learn.

If you have not done so already, please refer to our [installation instructions](../install.rst) for installing `quantcore.glm`.

*Note:* We use the [sklearn boston housing dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html) throughout the tutorial. If you wish to explore this dataset further, there are a handful of resources online. For example, [this blog](https://medium.com/@amitg0161/sklearn-linear-regression-tutorial-with-boston-house-dataset-cde74afd460a). 

```python
import pandas as pd
import sklearn
from quantcore.glm import GeneralizedLinearRegressor, GeneralizedLinearRegressorCV
```

## Data

We start by loading the scikit-learn Boston housing dataset and splitting it into training and test sets. For simplicity, we don't go into any details regarding exploration or data cleaning.

```python
from sklearn import datasets

boston = sklearn.datasets.load_boston()
df_bos = pd.DataFrame(boston.data, columns = boston.feature_names)
df_bos['PRICE'] = boston.target
df_bos = df_bos[df_bos['PRICE'] <= 40] # remove outliers
df_bos.head(3)
```

```python
X = df_bos[["CRIM", "ZN", "CHAS", "NOX", "RM", "AGE", "TAX", "B", "LSTAT"]]
y = df_bos["PRICE"]
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1, random_state=5)
```

## GLM basics: fitting and predicting using the normal family

We'll use `quantcore.glm.GeneralizedLinearRegressor` to predict the house prices using the available predictors. 

We set three key parameters:

- **`family`**: the family parameter specifies the distributional assumption of the GLM and, as a consequence, the loss function to be minimized. Accepted strings are 'normal', 'poisson', 'gamma', 'inverse.gaussian', and 'binomial'. You can also pass in an instantiated `quantcore.glm` distribution (e.g. `quantcore.glm.TweedieDistribution(1.5)` )
- **`alpha`**: the constant multiplying the penalty term that determines regularization strength. (*Note*: `GeneralizedLinearRegressor` also has an alpha-search option. See the `GeneralizedLinearRegressorCV` example below for details on how alpha-search works).
- **`l1_ratio`**: the elastic net mixing parameter (`0 <= l1_ratio <= 1`). For `l1_ratio = 0`, the penalty is the L2 penalty (ridge). ``For l1_ratio = 1``, it is an L1 penalty (lasso).  For ``0 < l1_ratio < 1``, the penalty is a combination of L1 and L2.

To be precise, we will be minimizing the function with respect to the parameters, $\beta$:

\begin{equation}
\frac{1}{N}(\mathbf{X}\beta - y)^2 + \alpha\|\beta\|_1
\end{equation}

```python
glm = GeneralizedLinearRegressor(family='normal', alpha=0.1, l1_ratio=1)
```

The `GeneralizedLinearRegressor.fit()` method follows typical sklearn API style and accepts two primary inputs:

1. `X`: the design matrix with shape `(n_samples, n_features)`.
2. `y`: the `n_sample` length array of target data.

```python
glm.fit(X_train, y_train)
```

The `predict()` method is also similar to sklearn. It accepts an `(n_samples, n_feature)` shaped design matrix as its input

```python
print(f"Train RMSE: {sklearn.metrics.mean_squared_error(glm.predict(X_train), y_train, squared=False)}")
print(f"Test  RMSE: {sklearn.metrics.mean_squared_error(glm.predict(X_test), y_test, squared=False)}")
```

## Fitting a GLM with cross validation

Now, we fit using automatic cross validation with `quantcore.glm.GeneralizedLinearRegressorCV`. This mirrors the commonly used `cv.glmnet` function. 

Some important parameters:

- **`alphas`**: for `GeneralizedLinearRegressorCV`, the best `alpha` will be found by searching along the regularization path. The regularization path is determined as follows:
    1. If ``alpha`` is an iterable, use it directly. All other parameters
        governing the regularization path are ignored.
    2. If ``min_alpha`` is set, create a path from ``min_alpha`` to the
        lowest alpha such that all coefficients are zero.
    3. If ``min_alpha_ratio`` is set, create a path where the ratio of
        ``min_alpha / max_alpha = min_alpha_ratio``.
    4. If none of the above parameters are set, use a ``min_alpha_ratio``
        of 1e-6.      
- **l1_ratio**: for `GeneralizedLinearRegressorCV`, if you pass ``l1_ratio`` as an array, the `fit` method will choose the best value of `l1_ratio` and store it as `self.l1_ratio_`.

```python
glmcv = GeneralizedLinearRegressorCV(
    family='normal',
    alphas=None,  # default
    min_alpha=None,  # default
    min_alpha_ratio=None,  # default
    l1_ratio=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
    fit_intercept=True,
    max_iter=150
)
glmcv.fit(X_train, y_train)
print(f"Chosen alpha:    {glmcv.alpha_}")
print(f"Chosen l1 ratio: {glmcv.l1_ratio_}")

print(f"Train RMSE: {sklearn.metrics.mean_squared_error(glmcv.predict(X_train), y_train, squared=False)}")
print(f"Test  RMSE: {sklearn.metrics.mean_squared_error(glmcv.predict(X_test), y_test, squared=False)}")
```

Congratulations! You have finished our getting started tutorial. If you wish to learn more, please see our other tutorials for more advanced topics like Poisson, Gamma, and Tweedie regression, high dimensional fixed effects, and spatial smoothing using Tikhonov regularization.
