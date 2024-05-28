# glum

[![CI](https://github.com/Quantco/glm_benchmarks/workflows/CI/badge.svg)](https://github.com/Quantco/glum/actions)
[![Docs](https://readthedocs.org/projects/pip/badge/?version=latest&style=flat)](https://glum.readthedocs.io/)
[![Conda-forge](https://img.shields.io/conda/vn/conda-forge/glum?logoColor=white&logo=conda-forge)](https://anaconda.org/conda-forge/glum)
[![PypiVersion](https://img.shields.io/pypi/v/glum.svg?logo=pypi&logoColor=white)](https://pypi.org/project/glum)
[![PythonVersion](https://img.shields.io/pypi/pyversions/glum?logoColor=white&logo=python)](https://pypi.org/project/glum)


[Documentation](https://glum.readthedocs.io/en/latest/)

Generalized linear models (GLM) are a core statistical tool that include many common methods like least-squares regression, Poisson regression and logistic regression as special cases. At QuantCo, we have used GLMs in e-commerce pricing, insurance claims prediction and more. We have developed `glum`, a fast Python-first GLM library. The development was based on [a fork of scikit-learn](https://github.com/scikit-learn/scikit-learn/pull/9405), so it has a scikit-learn-like API. We are thankful for the starting point provided by Christian Lorentzen in that PR!

The goal of `glum` is to be at least as feature-complete as existing GLM libraries like `glmnet` or `h2o`. It supports

* Built-in cross validation for optimal regularization, efficiently exploiting a “regularization path”
* L1 regularization, which produces sparse and easily interpretable solutions
* L2 regularization, including variable matrix-valued (Tikhonov) penalties, which are useful in modeling correlated effects
* Elastic net regularization
* Normal, Poisson, logistic, gamma, and Tweedie distributions, plus varied and customizable link functions
* Box constraints, linear inequality constraints, sample weights, offsets

This repo also includes tools for benchmarking GLM implementations in the `glum_benchmarks` module. For details on the benchmarking, [see here](src/glum_benchmarks/README.md). Although the performance of `glum` relative to `glmnet` and `h2o` depends on the specific problem, we find that when N >> K (there are more observations than predictors), it is consistently much faster for a wide range of problems.

![Performance benchmarks](docs/_static/headline_benchmark.png#gh-light-mode-only)
![Performance benchmarks](docs/_static/headline_benchmark_dark.png#gh-dark-mode-only)

For more information on `glum`, including tutorials and API reference, please see [the documentation](https://glum.readthedocs.io/en/latest/).

Why did we choose the name `glum`? We wanted a name that had the letters GLM and wasn't easily confused with any existing implementation. And we thought glum sounded like a funny name (and not glum at all!). If you need a more professional sounding name, feel free to pronounce it as G-L-um. Or maybe it stands for "Generalized linear... ummm... modeling?"

# A classic example predicting housing prices

```python
>>> from sklearn.datasets import fetch_openml
>>> from glum import GeneralizedLinearRegressor
>>>
>>> # This dataset contains house sale prices for King County, which includes
>>> # Seattle. It includes homes sold between May 2014 and May 2015.
>>> house_data = fetch_openml(name="house_sales", version=3, as_frame=True)
>>>
>>> # Use only select features
>>> X = house_data.data[
...     [
...         "bedrooms",
...         "bathrooms",
...         "sqft_living",
...         "floors",
...         "waterfront",
...         "view",
...         "condition",
...         "grade",
...         "yr_built",
...         "yr_renovated",
...     ]
... ].copy()
>>>
>>>
>>> # Model whether a house had an above or below median price via a Binomial
>>> # distribution. We'll be doing L1-regularized logistic regression.
>>> price = house_data.target
>>> y = (price < price.median()).values.astype(int)
>>> model = GeneralizedLinearRegressor(
...     family='binomial',
...     l1_ratio=1.0,
...     alpha=0.001
... )
>>>
>>> _ = model.fit(X=X, y=y)
>>>
>>> # .report_diagnostics shows details about the steps taken by the iterative solver.
>>> diags = model.get_formatted_diagnostics(full_report=True)
>>> diags[['objective_fct']]
        objective_fct
n_iter               
0            0.693091
1            0.489500
2            0.449585
3            0.443681
4            0.443498
5            0.443497
>>>
>>> # Models can also be built with formulas from formulaic.
>>> model_formula = GeneralizedLinearRegressor(
...     family='binomial',
...     l1_ratio=1.0,
...     alpha=0.001,
...     formula="bedrooms + np.log(bathrooms + 1) + bs(sqft_living, 3) + C(waterfront)"
... )
>>> _ = model_formula.fit(X=house_data.data, y=y)

```

# Installation

Please install the package through conda-forge:
```bash
conda install glum -c conda-forge
```

# Performance

For optimal performance on an x86_64 architecture, we recommend using the MKL library
(`conda install mkl`). By default, conda usually installs the openblas version, which
is slower, but supported on all major architecture and OS.
