---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.0
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

<!-- #region tags=[] -->
# Getting Started: fitting a Lasso model 

The purpose of this tutorial is to show the basics of `glum`. It assumes a working knowledge of python, regularized linear models, and machine learning. The API is very similar to scikit-learn. After all, `glum` is based on a fork of scikit-learn.

If you have not done so already, please refer to our [installation instructions](../install.rst) for installing `glum`.
<!-- #endregion -->

```python
import pandas as pd
import sklearn
from sklearn.datasets import fetch_openml
from glum import GeneralizedLinearRegressor, GeneralizedLinearRegressorCV
```

## Data

We start by loading the King County housing dataset from openML and splitting it into training and test sets. For simplicity, we don't go into any details regarding exploration or data cleaning.

```python
house_data = fetch_openml(name="house_sales", version=3, as_frame=True)

# Use only select features
X = house_data.data[
    [
        "bedrooms",
        "bathrooms",
        "sqft_living",
        "floors",
        "waterfront",
        "view",
        "condition",
        "grade",
        "yr_built",
    ]
].copy()

# Targets
y = house_data.target
```

```python
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X, y, test_size = 0.3, random_state=5
)
```

## GLM basics: fitting and predicting using the normal family

We'll use `glum.GeneralizedLinearRegressor` to predict the house prices using the available predictors. 

We set three key parameters:

- `family`: the family parameter specifies the distributional assumption of the GLM and, as a consequence, the loss function to be minimized. Accepted strings are 'normal', 'poisson', 'gamma', 'inverse.gaussian', and 'binomial'. You can also pass in an instantiated `glum` distribution (e.g. `glum.TweedieDistribution(1.5)` )
- `alpha`: the constant multiplying the penalty term that determines regularization strength.
- `l1_ratio`: the elastic net mixing parameter (`0 <= l1_ratio <= 1`). For `l1_ratio = 0`, the penalty is the L2 penalty (ridge). ``For l1_ratio = 1``, it is an L1 penalty (lasso).  For ``0 < l1_ratio < 1``, the penalty is a combination of L1 and L2.

To be precise, we will be minimizing the function with respect to the parameters, $\beta$:

\begin{equation}
\frac{1}{N}(\mathbf{X}\beta - y)^2 + \alpha\|\beta\|_1
\end{equation}

```python
glm = GeneralizedLinearRegressor(family="normal", alpha=0.1, l1_ratio=1)
```

The `GeneralizedLinearRegressor.fit()` method follows typical sklearn API style and accepts two primary inputs:

1. `X`: the design matrix with shape `(n_samples, n_features)`.
2. `y`: the `n_samples` length array of target data.

```python
glm.fit(X_train, y_train)
```

Once the model has been estimated, we can retrieve useful information using an sklearn-style syntax.

```python
# retrieve the coefficients and the intercept
coefs = glm.coef_
intercept = glm.intercept_

# use the model to predict on our test data
preds = glm.predict(X_test)

preds[0:5]
```

## Regularization

In the example above, the `alpha` and `l1_ratio` parameters specify the level of regularization, i.e. the amount by which fitted model coefficients are biased towards zero.
The advantage of the regularized model is that one avoids overfitting by controlling the tradeoff between the bias and the variance of the coefficient estimator.
An optimal level of regularization can be obtained data-adaptively through cross-validation. In the `GeneralizedLinearRegressorCV` example below, we show how this can be done by specifying an `alpha_search` parameter.

To fit an unregularized GLM we set `alpha=0`. Note that the default level `alpha=None` results in regularization at the level `alpha=1.0`, which is the default in the scikit-learn's [ElasticNet](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html).

A basic unregularized GLM object is obtained as
```python
glm = GeneralizedLinearRegressor(family="normal", alpha=0)
```
which we interact with as in the example above.

## Fitting a GLM with cross validation

Now, we fit using automatic cross validation with `glum.GeneralizedLinearRegressorCV`. This mirrors the commonly used `cv.glmnet` function. 

Some important parameters:

- `alphas`: for `GeneralizedLinearRegressorCV`, the best `alpha` will be found by searching along the regularization path. The regularization path is determined as follows:
    1. If `alpha` is an iterable, use it directly. All other parameters
        governing the regularization path are ignored.
    2. If `min_alpha` is set, create a path from `min_alpha` to the
        lowest alpha such that all coefficients are zero.
    3. If `min_alpha_ratio` is set, create a path where the ratio of
        `min_alpha / max_alpha = min_alpha_ratio`.
    4. If none of the above parameters are set, use a `min_alpha_ratio`
        of 1e-6.      
- `l1_ratio`: for `GeneralizedLinearRegressorCV`, if you pass `l1_ratio` as an array, the `fit` method will choose the best value of `l1_ratio` and store it as `self.l1_ratio_`.

```python
glmcv = GeneralizedLinearRegressorCV(
    family="normal",
    alphas=None,  # default
    min_alpha=None,  # default
    min_alpha_ratio=None,  # default
    l1_ratio=[0, 0.5, 1.0],
    fit_intercept=True,
    max_iter=150
)
glmcv.fit(X_train, y_train)
print(f"Chosen alpha:    {glmcv.alpha_}")
print(f"Chosen l1 ratio: {glmcv.l1_ratio_}")
```

Congratulations! You have finished our getting started tutorial. If you wish to learn more, please see our other tutorials for more advanced topics like Poisson, Gamma, and Tweedie regression, high dimensional fixed effects, and spatial smoothing using Tikhonov regularization.

```python

```
