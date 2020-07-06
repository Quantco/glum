# glm_benchmarks

![CI](https://github.com/Quantco/glm_benchmarks/workflows/CI/badge.svg)

Python package to benchmark GLM implementations. 

[Link to Google Sheet that compares various existing implementations.](https://docs.google.com/spreadsheets/d/1C-n3YTzPR47Sf8M04eEaX4RbNomM13dk_BZaPHGgWXg/edit)

[Link to Google Doc that compares the top contenders for libraries to improve.](https://docs.google.com/document/d/1hjmagUAS-NkUnD1r9Oyc8yL5NpeLHUyxprAIWWaVNAs/edit)

[Link to google doc that discusses some of the optimizations and improvements we have made](https://docs.google.com/document/d/1wd6_bV9OUFjqc9WGtELDJ1Kdv1jrrticivd50POTeqo/edit)


## Installation

You can install the package in development mode using:

```bash
# First, make sure you have conda-forge as your primary conda channel:
conda config --add channels conda-forge
# And install pre-commit
conda install -y pre-commit

git clone git@github.com:Quantco/glm_benchmarks.git
cd glm_benchmarks

# Set up our pre-commit hooks for black, mypy, isort and flake8.
pre-commit install

# Set up the ***REMOVED*** conda channel. For the password, substitute in the correct password. You should be able to get the password by searching around on slack or asking on the glm_benchmarks slack channel!
conda config --system --prepend channels ***REMOVED***
conda config --system --set custom_channels.***REMOVED*** https://***REMOVED***:password@conda.***REMOVED***
  
# Set up a conda environment with name "quantcore.glm"
conda install mamba=0.2.12
mamba env create

# Install this package in editable mode. 
conda activate quantcore.glm
pip install --no-use-pep517 --disable-pip-version-check -e .
```

## Running the benchmarks

After installing the package, you should have two CLI tools: `glm_benchmarks_run` and `glm_benchmarks_analyze`. Use the `--help` flag for full details. Look in `src/quantcore/glm/problems.py` to see the list of problems that will be run through each library.

To run the full benchmarking suite, just run `glm_benchmarks_run` with no flags. 

For a more advanced example: `glm_benchmarks_run --problem_name narrow-insurance-no-weights-l2-poisson --library_name sklearn-fork --storage dense --num_rows 100 --output_dir mydatadirname` will run just the first 100 rows of the `narrow-insurance-no-weights-l2-poisson` problem through the `sklearn-fork` library and save the output to `mydatadirname`. This demonstrates several capabilities that will speed development when you just want to run a subset of either data or problems or libraries. 

The `--problem_name` and `--library_name` flags take comma separated lists. This mean that if you want to run both `sklearn-fork` and `glmnet-python`, you could run `glm_benchmarks_run --library_name sklearn-fork,glmnet-python`.

The `glm_benchmarks_analyze` tool is still more a sketch-up and will evolve as we identify what we care about.

Benchmarks can be sped up by enabling caching of generated data. If you don't do this, 
you will spend a lot of time repeatedly generating the same data set. If you are using
Docker, caching is automatically enabled. The simulated data is written to an unmapped
directory within Docker, so it will cease to exist upon exiting the container. If you
are not using Docker, to enable caching, set the GLM_BENCHMARKS_CACHE environment
variable to the directory you would like to write to.

We support several types of matrix storage, passed with the argument "--storage". 
"dense" is the default. "sparse" stores data as a csc sparse matrix. "cat" splits
the matrix into a dense component and categorical components. "split0.1" splits the
matrix into sparse and dense parts, where any column with more than 10% nonzero elements
is put into the dense part, and the rest is put into the sparse part.

## Docker

To build the image, make sure you have a functioning Docker and docker-compose installation. Then, `docker-compose build work`.

To run something, for example: `docker-compose run work python benchmarks/sklearn_fork.py`

---
**NOTE FOR MAC USERS**

On MacOS, docker cannot use the "host" `network_mode` and will therefore have no exposed port. To use a jupyter notebook, you can instead start the container with `docker-compose run -p 8888:8888 workmac`. Port 8888 will be exposed and you will be able to access Jupyter.

---

## Library examples:

glmnet_python: see https://bitbucket.org/quantco/wayfairelastpricing/tests/test_glmnet_numerical.py
H2O: https://github.com/h2oai/h2o-tutorials/blob/master/tutorials/glm/glm_h2oworld_demo.py

## Profiling

For line-by-line profiling, use line_profiler `kernprof -lbv src/quantcore/glm/cli_run.py --problem_name narrow-insurance-no-weights-l2-poisson --library_name sklearn-fork`

For stack sampling profiling, use py-spy: `py-spy top -- python src/quantcore/glm/cli_run.py --problem_name narrow-insurance-no-weights-l2-poisson --library_name sklearn-fork`

## Memory profiling

To create a graph of memory usage:
```
mprof run --python -o mprofresults.dat --interval 0.01 src/quantcore/glm/cli_run.py --problem_name narrow-insurance-no-weights-l2-poisson --library_name sklearn-fork --num_rows 100000
mprof plot mprofresults.dat -o prof2.png
```

To do line-by-line memory profiling, add a `@profile` decorator to the functions you care about and then run:
```
python -m memory_profiler src/quantcore/glm/cli_run.py --problem_name narrow-insurance-no-weights-l2-poisson --library_name sklearn-fork --num_rows 100000
```

## Golden master tests

There are two sets of golden master tests, one with artificial data and one directly using the benchmarks and the problems. For both sets of tests, creating the golden master and the tests definition are located in the same file. Calling the file with pytest will run the tests while calling the file as a python script will generate the golden master result. When creating the golden master results, both scripts accept the `--overwrite` command line flag. If set, the existing golden master results will be overwritten. Otherwise, only the new problems will be run.

### Skipping the slow tests

If you want to skip the slow tests, add the `-m "not slow"` flag to any pytest command. The "wide" problems (all marked as slow tests) are especially poorly conditioned. This means that even for estimation with 10k observations, it might still be very slow. Furthermore, we also have golden master tests for the "narrow" and "intermediate" problems, so adding the "wide" problems do not add much coverage.

### Artificial golden master

To overwrite the golden master results:
```
python tests/sklearn_fork/test_golden_master.py
```

Add the `--overwrite` flag if you want to overwrite already existing golden master results

### Benchmarks golden master
To create the golden master results:
```
python tests/sklearn_fork/test_benchmark_golden_master.py
```

Add the `--overwrite` flag if you want to overwrite already existing golden master results.

## Building a conda package

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

## The algorithm

#### What kind of problems can we solve? 

This package is intended to fit L1 and L2-norm penalized Generalized Linear Models. Bounce over to [the Jupyter notebook for an introduction to GLMs](docs/glms.ipynb).

To summarize from that notebook, given a number of observations, indexed by `i`, and coefficients, indexed by `j`, we optimize the objective function:

```
sum_i weight_i * -log_likelihood_i + sum_j alpha_j * [l1_ratio * abs(coef_j) + (1 - l1_ratio) * coef_j ** 2]
```

In words, we minimize the log likelihood plus a L1 and/or L2 penalty term.

#### Solvers overview

There are three solvers implemented in the sklearn-fork subpackage. 

The first solver, `lbfgs` uses the scipy `fmin_l_bfgs_b` optimizer to minimize L2-penalized GLMs. The L-BFGS solver does not work with L1-penalties. Because L-BFGS does not store the full Hessian, it can be particularly effective for very high dimensional problems with several thousand or more columns. For more details, see: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html

The second and third solver are both based on Iteratively Reweighted Least Squares (IRLS). IRLS proceeds by iteratively approximating the objective function with a quadratic, then solving that quadratic for the optimal update. For purely L2-penalized settings, the `irls-ls` uses a least squares inner solver for each quadratic subproblem. For problems that have any L1-penalty component, the `irls-cd` uses a coordinate descent inner solver for each quadratic subproblem. 

The IRLS-LS and IRLS-CD implementations largely follow the algorithm described in `newglmnet` (see references below).

#### IRLS

In the `irls-cd` and `irls-ls` solvers, the outer loop is an IRLS iteration that forms a quadratic approximation to the negative loglikelihood. That is, we find `w` and `z` so that the problem can be expressed as `min sum_i w_i (z_i - x_i beta)^2 + penalty`. We exit when either the gradient is small (`gradient_tol`) or the step size is small (`step_size_tol`).

Within the `irls-cd` solver, the inner loop involves solving for `beta` with coordinate descent. We exit the inner loop when the quadratic problem's gradient is small. See the `coordinate_descent` reference. `irls-cd` is the only current algorithm implemented here that is able to solve L1-penalized GLMs.

The "inner loop" of the `irls-ls` solver is simply a direct least squares solve.

#### Active set tracking

When penalizing with an L1-norm, it is common for many coefficients to be exactly zero. And, it is possible to predict during a given iteration which of those coefficients will stay zero. As a result, we track the "active set" consisting of all the coefficients that are either currently non-zero or likely to remain non-zero. We follow the outer loop active set tracking algorithm in the `newglmnet` reference. That paper refers to the same concept as "shrinkage", whereas the `glmnet` reference calls this the "active set". Currently, we have not yet implemented the inner loop active set tracking from the `newglmnet` reference.

#### Hessian approximation. 

Depending on the distribution and link functions, we may not use the true Hessian. There are two potentially useful approximations:

1. The Gauss-Newton approximation: (https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm) Some interesting discussion and further links to literature on why the Gauss-Newton matrix can even outperform the true Hessian in some optimization problems: https://math.stackexchange.com/questions/2733257/approximation-of-hessian-jtj-for-general-non-linear-optimization-problems
2. The Fisher information matrix.  See [this discussion for an explanation of the BHHH algorithm.](https://github.com/Quantco/glm_benchmarks/pull/156#discussion_r434746239)

#### Approximate Hessian updating

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

#### References

`glmnet` - [Regularization Paths for Generalized Linear Models via Coordinate Descent](https://web.stanford.edu/~hastie/Papers/glmnet.pdf)

`newglmnet` - [An Improved GLMNET for L1-regularized LogisticRegression](https://www.csie.ntu.edu.tw/~cjlin/papers/l1_glmnet/long-glmnet.pdf)

`glmintro` - [Bryan Lewis on GLMs](https://bwlewis.github.io/GLM/)

`blismatmul` - [Anatomy of High-Performance Many-ThreadedMatrix Multiplication](http://www.cs.utexas.edu/~flame/pubs/blis3_ipdps14.pdf)

`coordinate_descent` - [Coordinate Descent Algorithms](http://www.optimization-online.org/DB_FILE/2014/12/4679.pdf)

## Matrix Types

Along with the GLM solvers, this package supports dense, sparse, categorical matrix types and mixtures of these types. Using the most efficient matrix representations massively improves performacne. 

For more details [see here](src/quantcore/glm/matrix/README.md)

We support dense matrices via standard numpy arrays. 

We support sparse CSR and CSC matrices via standard `scipy.sparse` objects. These `scipy.sparse` matrices have been modified in the `MKLSparseMatrix` class to use MKL via the `sparse_dot_mkl` package. As a result, sparse matrix-vector and matrix-matrix multiplies are optimized and parallelized. A user does not need to modify their code to take advantage of this optimization. If a `scipy.sparse.csc_matrix` object is passed in, it will be automatically converted to a `MKLSparseMatrix` object. This operation is almost free because no data needs to be copied.

We implement a CategoricalMatrix object that efficiently represents these matrices without nearly as much overhead as a normal CSC or CSR sparse matrix.

Finally, SplitMatrix allows mixing different matrix types for different columns to minimize overhead.

## Standardization

Internal to `GeneralizedLinearRegressor`, all matrix types are wrapped in a `ColScaledMat` which offsets columns to have mean zero and standard deviation one without modifying the matrix data itself. This avoids situations where modifying a matrix to have mean zero would result in losing the sparsity structure and avoids ever needing to copy or modify the input data matrix. As a result, memory usage is very low. 
