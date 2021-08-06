Development
=========

Hello! And thanks for exploring quantcore.glm more deeply. 

Install for development
----------

::

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

   # If you want to install the dependencies necessary for benchmarking against other GLM packages:
   mamba env update -n quantcore.glm --file environment-benchmark.yml

   # Install this package in editable mode. 
   conda activate quantcore.glm
   pip install --no-use-pep517 --disable-pip-version-check -e .

Testing/Continuous integration
---------

Golden master tests
^^^^^^^^^

We use golden master testing to preserve correctness. The results of many different GLM models have been saved. After an update, the tests will compare the new output to the saved models. Any significant deviation will result in a test failure. This doesn't strictly mean that the update was wrong. In case of a bug fix, it's possible that the new output will be more accurate than the old output. In that situation, the golden master results can be overwritten as explained below. 

There are two sets of golden master tests, one with artificial data and one directly using the benchmarking problems from `quantcore.glm_benchmarks`. For both sets of tests, creating the golden master and the tests definition are located in the same file. Calling the file with pytest will run the tests while calling the file as a python script will generate the golden master result. When creating the golden master results, both scripts accept the `--overwrite` command line flag. If set, the existing golden master results will be overwritten. Otherwise, only the new problems will be run.

Skipping the slow tests
^^^^^^^^

If you want to skip the slow tests, add the `-m "not slow"` flag to any pytest command. The "wide" problems (all marked as slow tests) are especially poorly conditioned. This means that even for estimation with 10k observations, it might still be very slow. Furthermore, we also have golden master tests for the "narrow" and "intermediate" problems, so adding the "wide" problems do not add much coverage.

Artificial golden master
^^^^^^^^

To overwrite the golden master results:
```
python tests/glm/test_golden_master.py
```

Add the `--overwrite` flag if you want to overwrite already existing golden master results

Benchmarks golden master
^^^^^^^^^^

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

