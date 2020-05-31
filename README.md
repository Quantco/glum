# glm_benchmarks

![CI](https://github.com/Quantco/glm_benchmarks/workflows/CI/badge.svg)

Python package to benchmark GLM implementations. 

[Link to Google Sheet that compares various existing implementations.](https://docs.google.com/spreadsheets/d/1C-n3YTzPR47Sf8M04eEaX4RbNomM13dk_BZaPHGgWXg/edit)

## Installation

You can install the package in development mode using:

```bash
git clone https://github.com/Quantco/glm_benchmarks
cd glm_benchmarks
pre-commit install
pip install --no-use-pep517 --disable-pip-version-check -e .
```

## Running the benchmarks

After installing the package, you should have two CLI tools: `glm_benchmarks_run` and `glm_benchmarks_analyze`. Use the `--help` flag for full details. Look in `src/glm_benchmarks/problems.py` to see the list of problems that will be run through each library. 

To run the full benchmarking suite, just run `glm_benchmarks_run` with no flags. 

For a more advanced example: `glm_benchmarks_run --problem_name narrow_insurance_l2_poisson --library_name sklearn_fork --storage dense --num_rows 100 --output_dir mydatadirname` will run just the first 100 rows of the `narrow_insurance_l2_poisson` problem through the `sklearn_fork` library and save the output to `mydatadirname`. This demonstrates several capabilities that will speed development when you just want to run a subset of either data or problems or libraries. 

The `--problem_name` and `--library_name` flags take comma separated lists. This mean that if you want to run both `sklearn_fork` and `glmnet_python`, you could run `glm_benchmarks_run --library_name sklearn_fork,glmnet_python`.

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

For line-by-line profiling, use line_profiler `kernprof -lbv src/glm_benchmarks/main.py --problem_name narrow_insurance_l2_poisson --library_name sklearn_fork`

For stack sampling profiling, use py-spy: `py-spy top -- python src/glm_benchmarks/main.py --problem_name narrow_insurance_l2_poisson --library_name sklearn_fork`

## Memory profiling

To create a graph of memory usage:
```
mprof run --python -o mprofresults.dat --interval 0.01 src/glm_benchmarks/main.py --problem_name narrow_insurance_l2_poisson --library_name sklearn_fork --num_rows 100000
mprof plot mprofresults.dat -o prof2.png
```

To do line-by-line memory profiling, add a `@profile` decorator to the functions you care about and then run:
```
python -m memory_profiler src/glm_benchmarks/main.py --problem_name narrow_insurance_l2_poisson --library_name sklearn_fork --num_rows 100000
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

