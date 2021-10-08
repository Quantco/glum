# glum

![CI](https://github.com/Quantco/glm_benchmarks/workflows/CI/badge.svg)

[Documentation](https://glum.readthedocs.io/en/latest/)

Generalized linear models (GLM) are a core statistical tool that include many common methods like least-squares regression, Poisson regression and logistic regression as special cases. At QuantCo, we have used GLMs in e-commerce pricing, insurance claims prediction and more. We have developed `glum`, a fast Python-first GLM library. The development was based on [a fork of scikit-learn](https://github.com/scikit-learn/scikit-learn/pull/9405), so it has a scikit-learn-like API. We are thankful for the starting point provided by Christian Lorentzen in that PR!

`glum` is at least as feature-complete as existing GLM libraries like `glmnet` or `h2o`. It supports

* Built-in cross validation for optimal regularization, efficiently exploiting a “regularization path”
* L1 regularization, which produces sparse and easily interpretable solutions
* L2 regularization, including variable matrix-valued (Tikhonov) penalties, which are useful in modeling correlated effects
* Elastic net regularization
* Normal, Poisson, logistic, gamma, and Tweedie distributions, plus varied and customizable link functions
* Box constraints, linear inequality constraints, sample weights, offsets

This repo also includes tools for benchmarking GLM implementations in the `glum_benchmarks` module. For details on the benchmarking, [see here](src/glum_benchmarks/README.md). Although the performance of `glum` relative to `glmnet` and `h2o` depends on the specific problem, we find that it is consistently much faster for a wide range of problems.

![](docs/_static/headline_benchmark.png)

For more information on `glum`, including tutorials and API reference, please see [the documentation](https://glum.readthedocs.io/en/latest/).

# An example: predicting car insurance claim frequency using Poisson regression.

This example uses a public French car insurance dataset.
```python
>>> import pandas as pd
>>> import numpy as np
>>> from glum_benchmarks.problems import load_data, generate_narrow_insurance_dataset
>>> from glum_benchmarks.util import get_obj_val
>>> from glum import GeneralizedLinearRegressor
>>>
>>> # Load the French Motor Insurance dataset
>>> dat = load_data(generate_narrow_insurance_dataset)
>>> X, y, sample_weight = dat['X'], dat['y'], dat['sample_weight']
>>>
>>> # Model the number of claims per year as Poisson and regularize using a L1-penalty.
>>> model = GeneralizedLinearRegressor(
...     family='poisson',
...     l1_ratio=1.0,
...     alpha=0.001
... )
>>>
>>> _ = model.fit(X=X, y=y, sample_weight=sample_weight)
>>>
>>> # .report_diagnostics shows details about the steps taken by the iterative solver
>>> diags = model.get_formatted_diagnostics(full_report=True)
>>> diags[['objective_fct']]
        objective_fct
n_iter               
0            0.331670
1            0.328841
2            0.319605
3            0.318660
4            0.318641
5            0.318641

```

# Installation

Please install the package through conda-forge:
```bash
conda install glum -c conda-forge
```
