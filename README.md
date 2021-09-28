# quantcore.glm

![CI](https://github.com/Quantco/glm_benchmarks/workflows/CI/badge.svg)

[Documentation](https://docs.dev.quantco.cloud/qc-github-artifacts/Quantco/quantcore.glm/latest/index.html)

Generalized linear models (GLM) are a core statistical tool that include many common methods like least-squares regression, Poisson regression and logistic regression as special cases. At QuantCo, we have used GLMs in e-commerce pricing, insurance claims prediction and more. We have developed `quantcore.glm`, a fast Python-first GLM library. `quantcore.glm` is starting to be used at DIL and will soon be used by DIL actuaries. The development was based on [a fork of scikit-learn](https://github.com/scikit-learn/scikit-learn/pull/9405), so it has a scikit-learn-like API. We are thankful for the starting point provided by Christian Lorentzen in that PR!

`quantcore.glm` is at least as feature-complete as existing GLM libraries like `glmnet` or `h2o`. It supports

* Built-in cross validation for optimal regularization, efficiently exploiting a “regularization path”
* L1 and elastic net regularization, which produce sparse and easily interpretable solutions
* L2 regularization, including variable matrix-valued (Tikhonov) penalties, which are useful in modeling correlated effects
* Normal, Poisson, logistic, gamma, and Tweedie distributions, plus varied and customizable link functions
* Box constraints, sample weights, offsets.

This repo also includes  tools for benchmarking GLM implementations in the `quantcore.glm_benchmarks` module. For details on the benchmarking, [see here](src/quantcore/glm_benchmarks/README.md).

Table of Contents
=================

   * [quantcore.glm](#quantcoreglm)
   * [Installation](#installation)
   * [A quick usage example](#a-quick-usage-example)
   * [A more extensive introduction to GLM modeling via the sklearn interface](#a-more-extensive-introduction-to-glm-modeling-via-the-sklearn-interface)
   * [Testing/Continuous integration](#testingcontinuous-integration)
      * [Golden master tests](#golden-master-tests)
      * [Skipping the slow tests](#skipping-the-slow-tests)
      * [Artificial golden master](#artificial-golden-master)
      * [Benchmarks golden master](#benchmarks-golden-master)
   * [Building a conda package](#building-a-conda-package)
   * [The algorithm](#the-algorithm)
      * [What kind of problems can we solve?](#what-kind-of-problems-can-we-solve)
      * [Solvers overview](#solvers-overview)
      * [IRLS](#irls)
      * [Active set tracking](#active-set-tracking)
            * [Hessian approximation.](#hessian-approximation)
      * [Approximate Hessian updating](#approximate-hessian-updating)
      * [References](#references)
   * [Matrix Types](#matrix-types)
   * [Standardization](#standardization)


# Installation

Assuming you have access to the QuantCo DIL conda repository, you can install the package through conda:
```bash
# Set up the quantco_main conda channel. For the password, substitute in the correct password. You should be able to get the password by searching around on slack or asking on the glm_benchmarks slack channel!
conda config --system --prepend channels quantco_main
conda config --system --set custom_channels.quantco_main https://dil_ro:password@conda.quantco.cloud

conda install quantcore.glm
```

For development, you should do an editable installation:

```bash
# First, make sure you have conda-forge as your primary conda channel:
conda config --add channels conda-forge
# And install pre-commit
conda install -y pre-commit

git clone git@github.com:Quantco/quantcore.glm.git
cd quantcore.glm

# Set up our pre-commit hooks for black, mypy, isort and flake8.
pre-commit install

# Set up the quantco_main conda channel. For the password, substitute in the correct password. You should be able to get the password by searching around on slack or asking on the glm_benchmarks slack channel!
conda config --system --prepend channels quantco_main
conda config --system --set custom_channels.quantco_main https://dil_ro:password@conda.quantco.cloud

# Set up a conda environment with name "quantcore.glm"
conda install mamba=0.2.12
mamba env create

# Install this package in editable mode.
conda activate quantcore.glm
pip install --no-use-pep517 --disable-pip-version-check -e .
```

# A quick usage example

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

# A more extensive introduction to GLM modeling via the sklearn interface

[This is an excellent tutorial walking through modeling the French Motor Insurance dataset. It is based on the sklearn fork that `quantcore.glm` was originally based on.](https://scikit-learn.org/stable/auto_examples/linear_model/plot_tweedie_regression_insurance_claims.html)

[See here for a Jupyter Notebook of a similar tutorial that has been converted from using the sklearn interface to using `quantcore.glm`](https://github.com/Quantco/french-motor-glm-tutorial/blob/master/glm_freMTPL2_example.ipynb)

[This is a brief tutorial on Tweedie Regression with L2 regularization from sklearn. `quantcore.glm` has many more features and capabilities but it can also replicate everything done here.](https://scikit-learn.org/stable/modules/linear_model.html#generalized-linear-regression)

## A quick note on 32-bit floating point precision
Note that a float32 data matrix is acceptable and will result in the entire algorithm being run in 32-bit precision.  However, for problems that are poorly conditioned, this might result in poor convergence or flawed parameter estimates. Poor conditioning can be caused by a large number of features with few observations, highly collinear features and by light regularization. In spite of this risk, it's common to see 1.5-2x improvements in runtime with 32-bit data.

# Testing/Continuous integration

## Golden master tests

We use golden master testing to preserve correctness. The results of many different GLM models have been saved. After an update, the tests will compare the new output to the saved models. Any significant deviation will result in a test failure. This doesn't strictly mean that the update was wrong. In case of a bug fix, it's possible that the new output will be more accurate than the old output. In that situation, the golden master results can be overwritten as explained below.

There are two sets of golden master tests, one with artificial data and one directly using the benchmarking problems from `quantcore.glm_benchmarks`. For both sets of tests, creating the golden master and the tests definition are located in the same file. Calling the file with pytest will run the tests while calling the file as a python script will generate the golden master result. When creating the golden master results, both scripts accept the `--overwrite` command line flag. If set, the existing golden master results will be overwritten. Otherwise, only the new problems will be run.

## Skipping the slow tests

If you want to skip the slow tests, add the `-m "not slow"` flag to any pytest command. The "wide" problems (all marked as slow tests) are especially poorly conditioned. This means that even for estimation with 10k observations, it might still be very slow. Furthermore, we also have golden master tests for the "narrow" and "intermediate" problems, so adding the "wide" problems do not add much coverage.

## Artificial golden master

To overwrite the golden master results:
```
python tests/glm/test_golden_master.py
```

Add the `--overwrite` flag if you want to overwrite already existing golden master results

## Benchmarks golden master

To create the golden master results:
```
python tests/glm/test_benchmark_golden_master.py
```

Add the `--overwrite` flag if you want to overwrite already existing golden master results.

# Building a conda package

To use the package in another project, we distribute it as a conda package.
For building the package locally, you can use the following command:

```
conda build conda.recipe
```

This will build the recipe using the standard compiler flags set by the conda-forge activation scripts.
Instead, we can override to build the architecture using a variant.

```
conda build conda.recipe --variants "{GLM_ARCHITECTURE: ['skylake']}"
```

By default, `conda` will always install the variant with the default compiler flags.
To explicitly install a version optimised for your CPU, you need to specify it as part of the build string:

```
conda install quantcore.glm=*=*skylake
```

# The algorithm

## What kind of problems can we solve?

This package is intended to fit L1 and L2-norm penalized Generalized Linear Models. Bounce over to [the Jupyter notebook for an introduction to GLMs](docs/glms.ipynb).

To summarize from that notebook, given a number of observations, indexed by `i`, and coefficients, indexed by `j`, we optimize the objective function:

```
sum_i weight_i * -log_likelihood_i + sum_j alpha_j * [l1_ratio * abs(coef_j) + (1 - l1_ratio) * coef_j ** 2]
```

In words, we minimize the log likelihood plus a L1 and/or L2 penalty term.

## Solvers overview

There are three solvers implemented in the sklearn-fork subpackage.

The first solver, `lbfgs` uses the scipy `fmin_l_bfgs_b` optimizer to minimize L2-penalized GLMs. The L-BFGS solver does not work with L1-penalties. Because L-BFGS does not store the full Hessian, it can be particularly effective for very high dimensional problems with several thousand or more columns. For more details, see: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html

The second and third solver are both based on Iteratively Reweighted Least Squares (IRLS). IRLS proceeds by iteratively approximating the objective function with a quadratic, then solving that quadratic for the optimal update. For purely L2-penalized settings, the `irls-ls` uses a least squares inner solver for each quadratic subproblem. For problems that have any L1-penalty component, the `irls-cd` uses a coordinate descent inner solver for each quadratic subproblem.

The IRLS-LS and IRLS-CD implementations largely follow the algorithm described in `newglmnet` (see references below).

## IRLS

In the `irls-cd` and `irls-ls` solvers, the outer loop is an IRLS iteration that forms a quadratic approximation to the negative loglikelihood. That is, we find `w` and `z` so that the problem can be expressed as `min sum_i w_i (z_i - x_i beta)^2 + penalty`. We exit when either the gradient is small (`gradient_tol`) or the step size is small (`step_size_tol`).

Within the `irls-cd` solver, the inner loop involves solving for `beta` with coordinate descent. We exit the inner loop when the quadratic problem's gradient is small. See the `coordinate_descent` reference. `irls-cd` is the only current algorithm implemented here that is able to solve L1-penalized GLMs.

The "inner loop" of the `irls-ls` solver is simply a direct least squares solve.

## Active set tracking

When penalizing with an L1-norm, it is common for many coefficients to be exactly zero. And, it is possible to predict during a given iteration which of those coefficients will stay zero. As a result, we track the "active set" consisting of all the coefficients that are either currently non-zero or likely to remain non-zero. We follow the outer loop active set tracking algorithm in the `newglmnet` reference. That paper refers to the same concept as "shrinkage", whereas the `glmnet` reference calls this the "active set". Currently, we have not yet implemented the inner loop active set tracking from the `newglmnet` reference.

#### Hessian approximation.

Depending on the distribution and link functions, we may not use the true Hessian. There are two potentially useful approximations:

1. [The Gauss-Newton approximation.](https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm) [Some interesting discussion and further links to literature on why the Gauss-Newton matrix can even outperform the true Hessian in some optimization problems:](https://math.stackexchange.com/questions/2733257/approximation-of-hessian-jtj-for-general-non-linear-optimization-problems)
2. The Fisher information matrix.  See [this discussion for an explanation of the BHHH algorithm.](https://github.com/Quantco/glm_benchmarks/pull/156#discussion_r434746239)

## Approximate Hessian updating

When we compute the Gauss-Newton approximation to the Hessian, the computation takes the form:

```
H = X^T @ diag(hessian_rows) @ X
```
where `hessian_rows` is a vector with length equal to the number of observations composed of the non-data-matrix components of the Hessian calculation (see above).

Instead of computing `H` directly, we will compute updates to `H`: `dH`

So, given `H0` from a previous iterations:
```
H0 = X^T @ diag(hessian_rows_0) @ X
```
we want to compute H1 from this iteration:
```
H1 = X^T @ diag(hessian_rows_1) @ X
```

However, we will instead compute:
```
H1 = H0 + dH
```
where
```
dH = X^T @ diag(hessian_rows_1 - hessian_rows_0) @ X
```

We will also refer to:
```
hessian_rows_diff = hessian_rows_1 - hessian_rows_0
```

The advantage of reframing the computation of `H` as an update is that the values in `hessian_rows_diff` will vary depending on how large the influence of the last coefficient update was on that row. As a result, in the majority of problems, many of the entries in `hessian_rows_diff` will be very very small.

The goal of the approximate update is to filter to a subset of `hessian_rows_diff` that we will use to compute the sandwich product for `dH`. Let's use the simple threshold where we only take rows where the update is similarly large to the largest row-wise update. If
```
abs(hessian_rows_diff[i]) >= T * max(abs(hessian_rows_diff)
```
then, we will include row `i` in the update. Essentially, this criteria ignores data matrix rows that have not seen the second derivatives of their predictions change very much in the last iteration. Smaller values of `T` result in a more accurate update, while larger values will result in a faster but less accurate update. If `T = 0`, then the update is exact. Thresholds (`T`) between 0.001 and 0.1 seem to work well.

It is critical to only update our `hessian_rows_0` for those rows that were included. That way, hessian_rows_diff is no longer the change since the last iteration, but instead, the change since the last iteration that a row was active. This ensures that we handle situations where a row changes a small amount over several iterations, eventually accumulating into a large change.

## References

`glmnet` - [Regularization Paths for Generalized Linear Models via Coordinate Descent](https://web.stanford.edu/~hastie/Papers/glmnet.pdf)

`newglmnet` - [An Improved GLMNET for L1-regularized LogisticRegression](https://www.csie.ntu.edu.tw/~cjlin/papers/l1_glmnet/long-glmnet.pdf)

`glmintro` - [Bryan Lewis on GLMs](https://bwlewis.github.io/GLM/)

`blismatmul` - [Anatomy of High-Performance Many-ThreadedMatrix Multiplication](http://www.cs.utexas.edu/~flame/pubs/blis3_ipdps14.pdf)

`coordinate_descent` - [Coordinate Descent Algorithms](http://www.optimization-online.org/DB_FILE/2014/12/4679.pdf)

# Matrix Types

Along with the GLM solvers, this package supports dense, sparse, categorical matrix types and mixtures of these types. Using the most efficient matrix representations massively improves performacne.

For more details, see the [README for quantcore.matrix](https://github.com/Quantco/quantcore.matrix)

We support dense matrices via standard numpy arrays.

We support sparse CSR and CSC matrices via standard `scipy.sparse` objects. However, we have extended these operations with custom matrix-vector and sandwich product routines that are optimized and parallelized. A user does not need to modify their code to take advantage of this optimization. If a `scipy.sparse.csc_matrix` object is passed in, it will be automatically converted to a `SparseMatrix` object. This operation is almost free because no data needs to be copied.

We implement a `CategoricalMatrix` object that efficiently represents these matrices without nearly as much overhead as a normal CSC or CSR sparse matrix.

Finally, `SplitMatrix` allows mixing different matrix types for different columns to minimize overhead.

# Standardization

Internal to `GeneralizedLinearRegressor`, all matrix types are wrapped in a `StandardizedMat` which offsets columns to have mean zero and standard deviation one without modifying the matrix data itself. This avoids situations where modifying a matrix to have mean zero would result in losing the sparsity structure and avoids ever needing to copy or modify the input data matrix. As a result, memory usage is very low.
