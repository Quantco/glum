Contributing and Development
=========

Hello! And thanks for exploring quantcore.glm more deeply. Please see the issue tracker and pull requests tabs on Github for information about what is currently happening. Feel free to post an issue if you'd like to get involved in development and don't really know where to start -- we can give some advice. 

We welcome contributions of any kind!

- New features
- Feature requests
- Bug reports
- Documentation
- Tests
- Questions

Pull request process
----------

- Before working on a non-trivial PR, please first discuss the change you wish to make via issue, Slack, email or any other method with the owners of this repository. This is meant to prevent spending time on a feature that will not be merged.
- Please make sure that a new feature comes with adequate tests. If these require data, please check if any of our existing test data sets fits the bill.
- Please make sure that all functions come with proper docstrings. If you do extensive work on docstrings, please check if the Sphinx documentation renders them correctly. The CI system builds it on every commit and pushes the rendered HTMLs to `https://docs.dev.quantco.cloud/qc-github-artifacts/Quantco/quantcore.glm/{YOUR_COMMIT}/index.html`
- Please make sure you have our pre-commit hooks installed.
- If you fix a bug, please consider first contributing a test that _fails_ because of the bug and then adding the fix as a separate commit, so that the CI system picks it up.
- Please add an entry to the change log and increment the version number according to the type of change. We use semantic versioning. Update the major if you break the public API. Update the minor if you add new functionality. Update the patch if you fixed a bug. All changes that have not been released are collected under the date `UNRELEASED`.

Releases
--------

- We make package releases infrequently, but usually any time a new non-trivial feature is contributed or a bug is fixed. To make a release, just open a PR that updates the change log with the current date. Once that PR is approved and merged, you can create a new release on [GitHub](https://github.com/Quantco/quantcore.glm/releases/new). Use the version from the change log as tag and copy the change log entry into the release description. New releases on GitHub are automatically deployed to the QuantCo conda channel.

Install for development
----------

The first step is to set up a conda environment and install quantcore.glm in editable mode.
::

   # First, make sure you have conda-forge as your primary conda channel:
   conda config --add channels conda-forge

   # And install pre-commit
   conda install -y pre-commit

   # Clone the repository
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


Testing and continuous integration
---------
The test suite is in `tests/`. 

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

Building a conda package
--------

To use the package in another project, we distribute it as a conda package.
For building the package locally, you can use the following command:

```
conda build conda.recipe
```

This will build the recipe using the standard compiler flags set by the conda-forge activation scripts.
