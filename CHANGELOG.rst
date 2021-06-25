.. Versioning follows semantic versioning, see also
   https://semver.org/spec/v2.0.0.html. The most important bits are:
   * Update the major if you break the public API
   * Update the minor if you add new functionality
   * Update the patch if you fixed a bug

Changelog
=========

1.4.3 - 2021-06-25
------------------

**Bug fix:**

- ``copy_X = False`` will now raise a value error when ``X`` has dtype ``int32`` or ``int64``. Previously, it would only raise for dtype ``int64``.

1.4.2 - 2021-06-15
------------------

**Tutorials and documenation improvements:**

- Adding tutorials to the documentation
- Additional documentation improvements

**Bug fix:**

- Verbose progress bar now working again.

**Other:**

- Small improvement in documentation for the ``alpha_index`` argument to :func:`quantcore.glm.GeneralizedLinearRegressor.predict`.
- Pinned pre-commit hooks versions.

1.4.1 - 2021-05-01
------------------

We now have Windows builds!

1.4.0 - 2021-04-13
------------------

**Deprecations:**

- Fusing the ``alpha`` and ``alphas`` arguments for :func:`quantcore.glm.GeneralizedLinearRegressor`. ``alpha`` now also accepts array like inputs. ``alphas`` is now deprecated but can still be used for backward compatibility. The ``alphas`` argument will be removed with the next major version.

**Bug fix:**

- We removed entry points to functions in ``quantcore.glm_benchmarks`` from the conda package.

1.3.1 - 2021-04-12
------------------

**Bug fix**:

- :func:`quantcore.glm._distribution.unit_variance_derivative` is
  evaluating a proper numexpr expression again (regression in 1.3.0).

1.3.0 - 2021-04-12
------------------

**New features:**

- We added a new solver based on ``scipy.optimize.minimize(method='trust-constr')``.
- We added support for linear inequality constraints of type ``A_ineq.dot(coef_) <= b_ineq``.

1.2.0 - 2021-02-04
------------------

We removed ``quantcore.glm_benchmarks`` from the conda package.

1.1.1 - 2021-01-11
------------------

Maintenance release to get a fresh build for OSX.

1.1.0 - 2020-11-23
------------------

**New features:**

- Direct support for pandas categorical types in ``fit`` and ``predict``. These will be converted into a ``CategoricalMatrix``.

1.0.1 - 2020-11-12
------------------

This is a maintenance release to be compatible with `quantcore.matrix>=1.0.0`.

1.0.0 - 2020-11-11
------------------

**New features:**

- Renamed `alpha_level` attribute of :func:`quantcore.glm.GeneralizedLinearRegressor` and :func:`quantcore.glm.GeneralizedLinearRegressorCV` to `alpha_index`.

**Other:**

- Clarified behavior of `scale_predictors`.

0.0.15 - 2020-11-11
-------------------

**Other:**

- Pin quantcore.matrix < 1.0.0 as we are expecting a breaking change with version 1.0.0.

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
