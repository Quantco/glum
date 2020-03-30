# glm_benchmarks

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

For a more advanced example: `glm_benchmarks_run --problem_names simple_insurance_l2 --library_names sklearn_fork --num_rows 100 --output_dir mydatadirname` will run just the first 100 rows of the `simple_insurance_l2` problem through the `sklearn_fork` library and save the output to `mydatadirname`. This demonstrates several capabilities that will speed development when you just want to run a subset of either data or problems or libraries. 

The `--problem_names` and `--library_names` flags take comma separated lists. This mean that if you want to run both `sklearn_fork` and `glmnet_python`, you could run `glm_benchmarks_run --library_names sklearn_fork,glmnet_python`.

The `glm_benchmarks_analyze` tool is still more a sketch-up and will evolve as we identify what we care about.

## Docker

To build the image, make sure you have a functioning Docker and docker-compose installation. Then, `docker-compose build work`.

To run something, for example: `docker-compose run work python benchmarks/sklearn_fork.py`

## Library examples:

glmnet_python: see https://bitbucket.org/quantco/wayfairelastpricing/tests/test_glmnet_numerical.py
H2O: https://github.com/h2oai/h2o-tutorials/blob/master/tutorials/glm/glm_h2oworld_demo.py

