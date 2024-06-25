.. Versioning follows semantic versioning, see also
   https://semver.org/spec/v2.0.0.html. The most important bits are:
   * Update the major if you break the public API
   * Update the minor if you add new functionality
   * Update the patch if you fixed a bug

Changelog
=========

3.0.2 - 2024-06-25
------------------

**Bug fix:**

- Fixed :meth:`~glum.GeneralizedLinearRegressor.wald_test` when using ``terms`` and no intercept.

**Other changes:**

- Moved the development infrastructure to pixi.
- Moved the linting and formatting to ruff.
- Removed libblas MKL from the development environment.
- Replaced deprecated 'oldest-supported-numpy' dependency with 'numpy' to support 2.0 release.

3.0.1 - 2024-05-23
------------------

**Bug fix:**

- We now support scikit-learn 1.5.


3.0.0 - 2024-04-27
------------------

**Breaking changes:**

- All arguments to :class:`~glum.GeneralizedLinearRegressorBase`, :class:`~glum.GeneralizedLinearRegressor` and :class:`GeneralizedLinearRegressorCV` are now keyword-only.
- All arguments to public methods of :class:`~glum.GeneralizedLinearRegressorBase`, :class:`~glum.GeneralizedLinearRegressor` or :class:`GeneralizedLinearRegressorCV` except ``X``, ``y``, ``sample_weight`` and ``offset`` are now keyword-only.
- :class:`~glum.GeneralizedLinearRegressor`'s default value for ``alpha`` is now ``0``, i.e. no regularization.
- :class:`~glum.GammaDistribution`, :class:`~glum.InverseGaussianDistribution`, :class:`~glum.NormalDistribution` and :class:`~glum.PoissonDistribution` no longer inherit from :class:`~glum.TweedieDistribution`.
- The power parameter of :class:`~glum.TweedieLink` has been renamed from ``p`` to ``power``, in line with :class:`~glum.TweedieDistribution`.
- :class:`~glum.TweedieLink` no longer instantiates :class:`~glum.IdentityLink` or :class:`~glum.LogLink` for ``power=0`` and ``power=1``, respectively. On the other hand, :class:`~glum.TweedieLink` is now compatible with ``power=0`` and ``power=1``.

**New features:**

- Added a formula interface for specifying models.
- Improved feature name handling. Feature names are now created for non-pandas input matrices too. Furthermore, the format of categorical features can be specified by the user.
- Term names are now stored in the model's attributes. This is useful for categorical features, where they refer to the whole variable, not just single levels.
- Added more options for treating missing values in categorical columns. They can either raise a ``ValueError`` (``"fail"``), be treated as all-zero indicators (``"zero"``) or represented as a new category (``"convert"``).
- `meth:GeneralizedLinearRegressor.wald_test` can now perform tests based on a formula string and term names.
- :class:`~glum.InverseGaussianDistribution` gains a :meth:`~glum.InverseGaussianDistribution.log_likelihood` method.


2.7.0 - 2024-02-19
------------------

**Bug fix:**

- Added cython compiler directive legacy_implicit_noexcept = True to fix performance regression with cython 3.

**Other changes:**

- Require Python>=3.9 in line with `NEP 29 <https://numpy.org/neps/nep-0029-deprecation_policy.html#support-table>`.
- Build and test with Python 3.12 in CI.
- Added line search stopping criterion for tiny loss improvements based on gradient information.
- Added warnings about breaking changes in future versions.


2.6.0 - 2023-09-05
------------------

**New features:**

- Added the complementary log-log (``cloglog``) link function.
- Added the option to store the covariance matrix after estimating it. In this case, the covariance matrix does not have to be recomputed when calling inference methods.
- Added methods for performing Wald tests based on a restriction matrix, feature names or term names.
- Added a method for creating a coefficient table with confidence intervals and p-values.

**Bug fix:**

- Fixed :meth:`~glum.GeneralizedLinearRegressorBase.covariance_matrix` mutating feature names when called with a data frame. See `here <https://github.com/Quantco/glum/issues/669>`_.

**Other changes:**

- When computing the covariance matrix, check whether the design matrix is ill-conditioned for all types of input. Furthermore, do it in a more efficient way.
- Pin ``tabmat<4.0.0`` (the new release will bring breaking changes).


2.5.2 - 2023-06-02
------------------

**Bug fix**

- Fix the ``glm_benchmarks_analyze`` command line tool. See `here <https://github.com/Quantco/glum/issues/642>`_.
- Fixed a bug in :class:`~glum.GeneralizedLinearRegressor` when fit on a data set with a constant column and ``warm_start=True``. See `here <https://github.com/Quantco/glum/issues/645>`_.

**Other changes:**

- Remove dev dependency on ``dask_ml``.
- We now pin ``llvm-openmp=11`` when creating the wheel for macOS in line with what scikit-learn does.


2.5.1 - 2023-05-19
------------------

**Bug fix:**

- We fixed a bug in the computation of :meth:`~glum.distribution.NegativeBinomialDistribution.log_likelihood`. Previously, this method just returned ``None``.


2.5.0 - 2023-04-28
------------------

**New feature:**

- Added Negative Binomial distribution by setting the ``'family'`` parameter of
  :class:`~glum.GeneralizedLinearRegressor` and :class:`~glum.GeneralizedLinearRegressorCV`
  to ``'negative.binomial'``.


2.4.1 - 2023-03-14
------------------

**Bug fixes:**

- Fixed an issue with :meth:`~glum.ExponentialDispersionModel._score_matrix` which failed when called with a tabmat matrix input.

**Other changes**:

- Removes unused scikit-learn cython imports.


2.4.0 - 2023-01-31
------------------

**Other changes**:

- :class:`~glum._link.LogitLink` has been made public.
- Apple Silicon wheels are now uploaded to PyPI.


2.3.0 - 2023-01-06
------------------

**Bug fixes:**

- A data frame with dense and sparse columns was transformed to a dense matrix instead of a split matrix by :meth:`~glum.GeneralizedLinearRegressor._set_up_and_check_fit_args`.
  Fixed by calling ``tabmat.from_pandas`` on any data frame.

**New features:**

- The following classes and functions have been made public:
  :class:`~glum._distribution.BinomialDistribution`,
  :class:`~glum._distribution.ExponentialDispersionModel`,
  :class:`~glum._distribution.GammaDistribution`,
  :class:`~glum._distribution.GeneralizedHyperbolicSecant`,
  :class:`~glum._distribution.InverseGaussianDistribution`,
  :class:`~glum._distribution.NormalDistribution`,
  :class:`~glum._distribution.PoissonDistribution`,
  :class:`~glum._link.IdentityLink`,
  :class:`~glum._link.Link`,
  :class:`~glum._link.LogLink`,
  :class:`~glum._link.TweedieLink`,
  :func:`~glum._glm.get_family` and
  :func:`~glum._glm.get_link`.
- The distribution and link classes now feature a more lenient equality check instead of the default identity check,
  so that, e.g., ``TweedieDistribution(1) == TweedieDistribution(1)`` now returns ``True``.


2.2.1 - 2022-11-25
------------------

**Other changes:**

- Fixing pypi upload issue. Version 2.2.0 will not be available through the standard distribution channels.


2.2.0 - 2022-11-25
------------------

**New features:**

- Add an argument to GeneralizedLinearRegressorBase to drop the first category in a Categorical column using [implementation in tabmat](https://github.com/Quantco/tabmat/pull/168)
- One may now request the Tweedie loss by setting the ``'family'`` parameter of
  :class:`~glum.GeneralizedLinearRegressor` and :class:`~glum.GeneralizedLinearRegressorCV`
  to ``'tweedie'``.

**Bug fixes:**

- Setting bounds for constant columns was not working (bounds were internally modified to 0).
  A similar issue was preventing inequalities from working with constant columns. This is now fixed.

**Other changes:**

- No more builds for 32-bit systems with python >= 3.8. This is due to scipy not supporting it anymore.


2.1.2 - 2022-07-01
------------------

**Other changes:**

- Next attempt to build wheel for PyPI without ``--march=native``.


2.1.1 - 2022-07-01
------------------

**Other changes:**

- We are now building the wheel for PyPI without ``--march=native`` to make it more portable across architectures.


2.1.0 - 2022-06-27
------------------

**New features:**

- Added :meth:`aic`, :meth:`aicc` and :meth:`bic` attributes to the :class:`~glum.GeneralizedLinearRegressor`.
  These attributes provide the information criteria based on the training data and the effective degrees of freedom
  of the maximum likelihood estimate for the model's parameters.
- :meth:`~glum.GeneralizedLinearRegressor.std_errors` and :meth:`~glum.GeneralizedLinearRegressor.covariance_matrix`
  of :class:`~glum.GeneralizedLinearRegressor` now accept data frames with categorical data.

**Bug fixes:**

- The :meth:`score` method of :class:`~glum.GeneralizedLinearRegressor` and :class:`~glum.GeneralizedLinearRegressorCV` now accepts offsets.
- Fixed the calculation of the information matrix for the Binomial distribution with logit link, which affected nonrobust standard errors.

**Other:**

- The CI now runs daily unit tests against the nightly builds of numpy, pandas and scikit-learn.
- The minimally required version of tabmat is now 3.1.0.


2.0.3 - 2021-11-05
------------------

**Other:**

- We are now specifying the run time dependencies in ``setup.py``, so that missing dependencies are automatically installed from PyPI when installing ``glum`` via pip.


2.0.2 - 2021-11-03
------------------

**Bug fix:**

- Fixed the sign of the log likelihood of the Gaussian distribution (not used for fitting coefficients).
- Fixed the wide benchmarks which had duplicated columns (categorical and numerical).

**Other:**

- The CI now builds the wheels and upload to pypi with every new release.
- Renamed functions checking for qc.matrix compliance to refer to tabmat.


2.0.1 - 2021-10-11
------------------

**Bug fix:**

- Fixed pyproject.toml. We now support installing through pip and pep517.


2.0.0 - 2021-10-08
------------------

**Breaking changes:**

- Renamed the package to ``glum``!! Hurray! Celebration.
- :class:`~glum.GeneralizedLinearRegressor` and :class:`~glum.GeneralizedLinearRegressorCV` lose the ``fit_dispersion`` parameter.
  Please use the :meth:`dispersion` method of the appropriate family instance instead.
- All functions now use ``sample_weight`` as a keyword instead of ``weights``, in line with scikit-learn.
- All functions now use ``dispersion`` as a keyword instead of ``phi``.
- Several methods :class:`~glum.GeneralizedLinearRegressor` and :class:`~glum.GeneralizedLinearRegressorCV` that should have been private have had an underscore prefixed on their names: :meth:`tear_down_from_fit`, :meth:`_set_up_for_fit`, :meth:`_set_up_and_check_fit_args`, :meth:`_get_start_coef`, :meth:`_solve` and :meth:`_solve_regularization_path`.
- :meth:`glum.GeneralizedLinearRegressor.report_diagnostics` and :meth:`glum.GeneralizedLinearRegressor.get_formatted_diagnostics` are now public.

**New features:**

- P1 and P2 now accepts 1d array with the same number of elements as the unexpanded design matrix. In this case,
  the penalty associated with a categorical feature will be expanded to as many elements as there are levels,
  all with the same value.
- :class:`ExponentialDispersionModel` gains a :meth:`dispersion` method.
- :class:`BinomialDistribution` and :class:`TweedieDistribution` gain a :meth:`log_likelihood` method.
- The :meth:`fit` method of :class:`~glum.GeneralizedLinearRegressor` and :class:`~glum.GeneralizedLinearRegressorCV`
  now saves the column types of pandas data frames.
- :class:`~glum.GeneralizedLinearRegressor` and :class:`~glum.GeneralizedLinearRegressorCV` gain two properties: ``family_instance`` and ``link_instance``.
- :meth:`~glum.GeneralizedLinearRegressor.std_errors` and :meth:`~glum.GeneralizedLinearRegressor.covariance_matrix` have been added and support non-robust, robust (HC-1), and clustered
  covariance matrices.
- :class:`~glum.GeneralizedLinearRegressor` and :class:`~glum.GeneralizedLinearRegressorCV` now accept ``family='gaussian'`` as an alternative to ``family='normal'``.

**Bug fix:**

- The :meth:`score` method of :class:`~glum.GeneralizedLinearRegressor` and :class:`~glum.GeneralizedLinearRegressorCV` now accepts data frames.
- Upgraded the code to use tabmat 3.0.0.

**Other:**

- A major overhaul of the documentation. Everything is better!
- The methods of the link classes will now return scalars when given scalar inputs. Under certain circumstances, they'd return zero-dimensional arrays.
- There is a new benchmark available ``glm_benchmarks_run`` based on the Boston housing dataset. See `here <https://github.com/Quantco/glum/pull/376>`_.
- ``glm_benchmarks_analyze`` now includes ``offset`` in the index. See `here <https://github.com/Quantco/glum/issues/346>`_.
- ``glmnet_python`` was removed from the benchmarks suite.
- The innermost coordinate descent was optimized. This speeds up coordinate descent dominated problems like LASSO by about 1.5-2x. See `here <https://github.com/Quantco/glum/pull/424>`_.


1.5.1 - 2021-07-22
------------------

**Bug fix:**

* Have the :meth:`linear_predictor` and :meth:`predict` methods of :class:`~glum.GeneralizedLinearRegressor` and :class:`~glum.GeneralizedLinearRegressorCV`
  honor the offset when ``alpha`` is ``None``.

1.5.0 - 2021-07-15
------------------

**New features:**

* The :meth:`linear_predictor` and :meth:`predict` methods of :class:`~glum.GeneralizedLinearRegressor` and :class:`~glum.GeneralizedLinearRegressorCV`
  gain an ``alpha`` parameter (in complement to ``alpha_index``). Moreover, they are now able to predict for multiple penalties.

**Other:**

* Methods of :class:`~glum._link.Link` now consistently return NumPy arrays, whereas they used to preserve pandas series in special cases.
* Don't list ``sparse_dot_mkl`` as a runtime requirement from the conda recipe.
* The minimal ``numpy`` pin should be dependent on the ``numpy`` version in ``host`` and not fixed to ``1.16``.


1.4.3 - 2021-06-25
------------------

**Bug fix:**

- ``copy_X = False`` will now raise a value error when ``X`` has dtype ``int32`` or ``int64``. Previously, it would only raise for dtype ``int64``.


1.4.2 - 2021-06-15
------------------

**Tutorials and documentation improvements:**

- Adding tutorials to the documentation.
- Additional documentation improvements.

**Bug fix:**

- Verbose progress bar now working again.

**Other:**

- Small improvement in documentation for the ``alpha_index`` argument to :meth:`~glum.GeneralizedLinearRegressor.predict`.
- Pinned pre-commit hooks versions.


1.4.1 - 2021-05-01
------------------

We now have Windows builds!


1.4.0 - 2021-04-13
------------------

**Deprecations:**

- Fusing the ``alpha`` and ``alphas`` arguments for :class:`~glum.GeneralizedLinearRegressor`. ``alpha`` now also accepts array like inputs. ``alphas`` is now deprecated but can still be used for backward compatibility. The ``alphas`` argument will be removed with the next major version.

**Bug fix:**

- We removed entry points to functions in ``glum_benchmarks`` from the conda package.


1.3.1 - 2021-04-12
------------------

**Bug fix:**

- :func:`glum._distribution.unit_variance_derivative` is
  evaluating a proper numexpr expression again (regression in 1.3.0).


1.3.0 - 2021-04-12
------------------

**New features:**

- We added a new solver based on ``scipy.optimize.minimize(method='trust-constr')``.
- We added support for linear inequality constraints of type ``A_ineq.dot(coef_) <= b_ineq``.


1.2.0 - 2021-02-04
------------------

We removed ``glum_benchmarks`` from the conda package.


1.1.1 - 2021-01-11
------------------

Maintenance release to get a fresh build for OSX.


1.1.0 - 2020-11-23
------------------

**New feature:**

- Direct support for pandas categorical types in ``fit`` and ``predict``. These will be converted into a :class:`CategoricalMatrix`.


1.0.1 - 2020-11-12
------------------

This is a maintenance release to be compatible with ``tabmat>=1.0.0``.


1.0.0 - 2020-11-11
------------------

**Other:**

- Renamed ``alpha_level`` attribute of :class:`~glum.GeneralizedLinearRegressor` and :class:`~glum.GeneralizedLinearRegressorCV` to ``alpha_index``.
- Clarified behavior of ``scale_predictors``.


0.0.15 - 2020-11-11
-------------------

**Other:**

- Pin ``tabmat<1.0.0`` as we are expecting a breaking change with version 1.0.0.


0.0.14 - 2020-08-06
-------------------

**New features:**

- Add Tweedie Link.
- Allow infinite bounds.

**Bug fixes:**

- Unstandardize regularization path.
- No copying in predict.

**Other:**

- Various memory and performance improvements.
- Update pre-commit hooks.


0.0.13 - 2020-07-23
-------------------

See git history.


0.0.12 - 2020-07-07
-------------------

See git history.


0.0.11 - 2020-07-02
-------------------

See git history.


0.0.10 - 2020-06-30
-------------------

See git history.


0.0.9 - 2020-06-26
-------------------

See git history.


0.0.8 - 2020-06-24
------------------

See git history.


0.0.7 - 2020-06-17
------------------

See git history.


0.0.6 - 2020-06-16
------------------

See git history.


0.0.5 - 2020-06-10
------------------

See git history.


0.0.4 - 2020-06-08
------------------

See git history.


0.0.3 - 2020-06-08
------------------

See git history.
