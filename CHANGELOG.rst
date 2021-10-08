.. Versioning follows semantic versioning, see also
   https://semver.org/spec/v2.0.0.html. The most important bits are:
   * Update the major if you break the public API
   * Update the minor if you add new functionality
   * Update the patch if you fixed a bug

Changelog
=========

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
