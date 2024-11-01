Contributing and Development
====================================

Hello! And thanks for exploring glum more deeply. Please see the issue tracker and pull requests tabs on Github for information about what is currently happening. Feel free to post an issue if you'd like to get involved in development and don't really know where to start -- we can give some advice. 

We welcome contributions of any kind!

- New features
- Feature requests
- Bug reports
- Documentation
- Tests
- Questions

Pull request process
--------------------------------------------------

- Before working on a non-trivial PR, please first discuss the change you wish to make via issue, Slack, email or any other method with the owners of this repository. This is meant to prevent spending time on a feature that will not be merged.
- Please make sure that a new feature comes with adequate tests. If these require data, please check if any of our existing test data sets fits the bill.
- Please make sure that all functions come with proper docstrings. If you do extensive work on docstrings, please check if the Sphinx documentation renders them correctly. ReadTheDocs builds on every commit to an open pull request. You can see whether the documentation has successfully built in the "checks" section of the PR. Once the build finishes, your documentation should be accessible by clicking the "details" link next to the check in the GitHub interface and will appear at a URL like: ``https://glum--###.org.readthedocs.build/en/###/`` where ``###`` is the number of your PR.
- Please make sure you have our pre-commit hooks installed.
- If you fix a bug, please consider first contributing a test that _fails_ because of the bug and then adding the fix as a separate commit, so that the CI system picks it up.
- Please add an entry to the change log and increment the version number according to the type of change. We use semantic versioning. Update the major if you break the public API. Update the minor if you add new functionality. Update the patch if you fixed a bug. All changes that have not been released are collected under the date ``UNRELEASED``.

Releases
--------------------------------------------------

- We make package releases infrequently, but usually any time a new non-trivial feature is contributed or a bug is fixed. To make a release, just open a PR that updates the change log with the current date. Once that PR is approved and merged, you can create a new release on [GitHub](https://github.com/Quantco/glum/releases/new). Use the version from the change log as tag and copy the change log entry into the release description. 

Install for development
--------------------------------------------------

The first step is to set up a conda environment and install glum in editable mode.
The project uses [pixi](https://pixi.sh/latest/) for managing the environment. If you don't have pixi installed, start by [installing it](https://pixi.sh/latest/#installation).

::

   # Clone the repository
   git clone git@github.com:Quantco/glum.git
   cd glum

   # Install the pre-commit hooks
   pixi run pre-commit-install

   # Install the dependencies, as well as and glum in editable mode
   pixi run postinstall

   # If you want to install the dependencies necessary for benchmarking against other GLM packages:
   pixi run -e benchmark postinstall

   # If you want to work on the tutorial notebooks or the documentation:
   pixi run -e docs postinstall

   # You can run any command in the pixi environment with `pixi run <command>`. For example:
   pixi run [-e ENVIRONMENT] ipython

   # Alternatively, you can create a shell with the pixi environment activated:
   pixi shell

   # Alternatively, a number of pixi tasks are available for commonly used commands.
   # You can run them with `pixi run <task>`.
   # To get a list of available tasks, run:
   pixi task list


Testing and continuous integration
--------------------------------------------------
The test suite is in ``tests/``. A pixi task is available to run the tests:

::

   pixi run test


Golden master tests
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We use golden master testing to preserve correctness. The results of many different GLM models have been saved. After an update, the tests will compare the new output to the saved models. Any significant deviation will result in a test failure. This doesn't strictly mean that the update was wrong. In case of a bug fix, it's possible that the new output will be more accurate than the old output. In that situation, the golden master results can be overwritten as explained below. 

There are two sets of golden master tests, one with artificial data and one directly using the benchmarking problems from :mod:`glum_benchmarks`. For both sets of tests, creating the golden master and the tests definition are located in the same file. Calling the file with pytest will run the tests while calling the file as a python script will generate the golden master result. When creating the golden master results, both scripts accept the ``--overwrite`` command line flag. If set, the existing golden master results will be overwritten. Otherwise, only the new problems will be run.
 
Skipping the slow tests
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you want to skip the slow tests, add the ``-m "not slow"`` flag to any pytest command. The "wide" problems (all marked as slow tests) are especially poorly conditioned. This means that even for estimation with 10k observations, it might still be very slow. Furthermore, we also have golden master tests for the "narrow" and "intermediate" problems, so adding the "wide" problems do not add much coverage.

Storing and modifying
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To store the golden master results, use the following pixi tasks:

::

   pixi run store-golden-master
   pixi run store-benchmark-golden-master

Add the ``--overwrite`` flag if you want to overwrite already existing golden master results

Building a conda package
----------------------------------------

To use the package in another project, we distribute it as a conda package.
For building the package locally, you can use the following command:

:: 

   conda build conda.recipe

This will build the recipe using the standard compiler flags set by the conda-forge activation scripts.

Developing the documentation
----------------------------------------

The documentation is built with a mix of Sphinx, autodoc, and nbsphinx. To develop the documentation:

::

   pixi run serve-docs

Then, navigate to `<http://localhost:8000>`_ to view the documentation.

Alternatively, if you install `entr <http://eradman.com/entrproject/>`_, then you can auto-rebuild the documentation any time a file changes with:

:: 

   cd docs
   ./dev

.. note::
   The tutorial notebooks are not executed as part of the documentation build. If you want to modify them, make sure to execute them manually and save the output before committing. Also don't forget to install the extra dependencies for the tutorial notebooks as described above.

If you are a newbie to Sphinx, the links below may help get you up to speed on some of the trickier aspects:

* `An idiot's guide to Sphinx <https://samnicholls.net/2016/06/15/how-to-sphinx-readthedocs/>`_
* `Links between documents <https://stackoverflow.com/questions/37553750/how-can-i-link-reference-another-rest-file-in-the-documentation>`_
* `Cross-referencing python objects <http://certik.github.io/sphinx/markup/inline.html#cross-referencing-python-objects>`_ using things like ``:mod:`` and ``:meth:`` and ``:class:``.
* `autodoc is used for automatically converting docstrings to docs <https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#module-sphinx.ext.autodoc>`_
* `We follow the numpy docstring style guide <https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard>`_
* `To create links between ipynb files when using nbsphinx <https://nbsphinx.readthedocs.io/en/0.4.1/markdown-cells.html#Links-to-*.rst-Files-(and-Other-Sphinx-Source-Files)>`_

Where to start looking in the source?
-------------------------------------

The primary user interface of ``glum`` consists of the :class:`GeneralizedLinearRegressor <glum.GeneralizedLinearRegressor>` and :class:`GeneralizedLinearRegressorCV <glum.GeneralizedLinearRegressorCV>` classes via their constructors and the :meth:`fit() <glum.GeneralizedLinearRegressor.fit>` and :meth:`predict() <glum.GeneralizedLinearRegressor.predict>` functions. Those are the places to start looking if you plan to change the system in some way. 

What follows is a high-level summary of the source code structure. For more details, please look in the documentation and docstrings of the relevant classes, functions and methods.

* ``_glm.py`` - This is the main entrypoint and implements the core logic of the GLM. Most of the code in this file handles input arguments and prepares the data for the GLM fitting algorithm.
* ``_glm_cv.py`` - This is the entrypoint for the cross validated GLM implementation. It depends on a lot of the code in ``_glm.py`` and only modifies the sections necessary for running training many models with different regularization parameters.
* ``_solvers.py`` - This contains the bulk of the IRLS and L-BFGS algorithms for training GLMs.
* ``_cd_fast.pyx`` - This is a Cython implementation of the coordinate descent algorithm used for fitting L1 penalty GLMs. Note the ``.pyx`` extension indicating that it is a Cython source file.
* ``_distribution.py`` - definitions of the distributions that can be used. Includes Normal, Poisson, Gamma, InverseGaussian, Tweedie, Binomial and GeneralizedHyperbolicSecant distributions. 
* ``_link.py`` - definitions of the link functions that can be used. Includes identity, log, logit and Tweedie link functions.
* ``_functions.pyx`` - This is a Cython implementation of the log likelihoods, gradients and Hessians for several popular distributions.
* ``_util.py`` - This contains a few general purpose linear algebra routines that serve several other modules and don't fit well elsewhere.

The GLM benchmark suite
------------------------

Before deciding to build a library custom built for our purposes, we did an thorough investigation of the various open source GLM implementations available. This resulted in an extensive suite of benchmarks for comparing the correctness, runtime and availability of features for these libraries. 

The benchmark suite has two command line entrypoints:

* ``glm_benchmarks_run``
* ``glm_benchmarks_analyze``

Both of these CLI tools take a range of arguments that specify the details of the benchmark problems and which libraries to benchmark.

For more details on the benchmark suite, see the README in the source at ``src/glum_benchmarks/README.md``.

