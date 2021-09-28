# quantcore.glm

![CI](https://github.com/Quantco/glm_benchmarks/workflows/CI/badge.svg)

[Documentation](https://docs.dev.quantco.cloud/qc-github-artifacts/Quantco/quantcore.glm/latest/index.html)

Generalized linear models (GLM) are a core statistical tool that include many common methods like least-squares regression, Poisson regression and logistic regression as special cases. At QuantCo, we have used GLMs in e-commerce pricing, insurance claims prediction and more. We have developed `quantcore.glm`, a fast Python-first GLM library. `quantcore.glm` is starting to be used at DIL and will soon be used by DIL actuaries. The development was based on [a fork of scikit-learn](https://github.com/scikit-learn/scikit-learn/pull/9405), so it has a scikit-learn-like API. We are thankful for the starting point provided by Christian Lorentzen in that PR!

`quantcore.glm` is at least as feature-complete as existing GLM libraries like `glmnet` or `h2o`. It supports

* Built-in cross validation for optimal regularization, efficiently exploiting a “regularization path”
* L1 regularization, which produces sparse and easily interpretable solutions
* L2 regularization, including variable matrix-valued (Tikhonov) penalties, which are useful in modeling correlated effects
* Elastic net regularization
* Normal, Poisson, logistic, gamma, and Tweedie distributions, plus varied and customizable link functions
* Box constraints, linear inequality constraints, sample weights, offsets

This repo also includes  tools for benchmarking GLM implementations in the `quantcore.glm_benchmarks` module. For details on the benchmarking, [see here](src/quantcore/glm_benchmarks/README.md).

For more information on `quantcore.glm`, including tutorials and API reference, please see [the documentation](https://docs.dev.quantco.cloud/qc-github-artifacts/Quantco/quantcore.glm/latest/index.html).

![](docs/_static/headline_benchmark.png)

# An example: predicting car insurance claim frequency using Poisson regression.

This example uses a public French car insurance dataset.
```python
import pandas as pd
import numpy as np

from quantcore.glm_benchmarks.problems import load_data, generate_narrow_insurance_dataset
from quantcore.glm_benchmarks.util import get_obj_val
from quantcore.glm import GeneralizedLinearRegressor

# Load the French Motor Insurance dataset
dat = load_data(generate_narrow_insurance_dataset)
X, y, weights = dat['X'], dat['y'], dat['weights']

# Model the number of claims per year as Poisson and regularize using a L1-penalty.
model = GeneralizedLinearRegressor(
    family='poisson',
    l1_ratio=1.0,
    alpha=0.001
)

model.fit(X=X, y=y, sample_weight=weights)

# .report_diagnostics shows details about the steps taken by the iterative solver
model._report_diagnostics(full_report=True)

print(pd.DataFrame(dict(name=X.columns, coef=model.coef_)).set_index('name'))

print('Percent of coefficients non-zero', 100 * np.mean(np.abs(model.coef_) > 0))
print('Zeros RMSE', np.sqrt(np.mean((0 - y) ** 2)))
print('Model RMSE', np.sqrt(np.mean((model.predict(X) - y) ** 2)))
print('Zeros log-likelihood', get_obj_val(dat, 'poisson', 0.0, 0.0, 0, np.zeros_like(model.coef_)))
print('Model log-likelihood', get_obj_val(dat, 'poisson', 0.0, 0.0, model.intercept_, model.coef_))


>>> Percent of coefficients non-zero 24.074074074074073
>>> Zeros RMSE 4.593120173102336
>>> Model RMSE 4.584480161172895
>>> Zeros log-likelihood 0.9999999999996729
>>> Model log-likelihood 0.3167597964655323
```

# Installation

Please install the package through conda-forge:
```bash
conda config --prepend channels conda-forge
conda install quantcore.glm
```
