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

Objective function: -sum_i weight_i LL_i + penalty

Outer loop: Form a quadratic approximation to -LL, as in iteratively reweighted least squares. That is, find w and z so that the problem can be expressed as “min sum_i w_i (z_i - x_i beta)^2 + penalty”. Exit when the gradient is small.

Inner loop: Use coordinate descent to find beta (holding w and z fixed). Exit when the gradient for the inner problem is small. 

#### Solvers overview

There are three solvers implemented in the sklearn-fork subpackage. 

The first solver, `lbfgs` uses the scipy `fmin_l_bfgs_b` optimizer to minimize L2-penalized GLMs. The L-BFGS solver does not work with L1-penalties. For more details, see: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html

The second and third solver are both based on Iteratively Reweighted Least Squares (IRLS). IRLS proceeds by iteratively approximating the objective function with a quadratic, then solving that quadratic for the optimal update. For purely L2-penalized settings, the `irls-ls` uses a least squares inner solver for each quadratic subproblem. For problems that have any L1-penalty component, the `irls-cd` uses a coordinate descent inner solver for each quadratic subproblem. 

#### Active set tracking

...

#### Gauss-Newton Approximation

The optimization algorithm used here is a type of Gauss-Newton method where the Hessian is approximated as the outer product of the gradient (https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm). The same approximation can be inspired via arguments relating to the Fisher information matrix (https://en.wikipedia.org/wiki/Information_matrix_test). For canonical link functions, the Hessian and gradient outer product should be exactly equal. The gradient outer product can be particularly valuable for non-canonical link functions because the gradient outer product (`J.T @ J`) is guaranteed to be symmetric and positive definite whereas the true Hessian is not. Some interesting discussion and further links to literature on why the Gauss-Newton matrix can even outperform the true Hessian in some optimization problems: https://math.stackexchange.com/questions/2733257/approximation-of-hessian-jtj-for-general-non-linear-optimization-problems

#### Approximate Hessian updating

...

#### References

...
