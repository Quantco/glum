Stepwise selection and matrix caching
=====================================

``glum`` ships two complementary tools for workflows that repeatedly fit
GLMs on the same DataFrame:

* :class:`~glum.TabmatCache` — a reusable, persistable cache for
  ``tabmat`` matrix construction and per-fold row-slicing.
* :class:`~glum.StepwiseGLM` — an accelerated wrapper around
  :class:`~glum.GeneralizedLinearRegressor` for stepwise variable
  selection, score-test screening, and cross-validated candidate ranking.

These cover use cases such as forward / backward stepwise selection,
grid search, nested cross-validation, ensemble fitting, and periodic
model refits where the underlying data stays mostly the same.


Why this is fast
----------------

Three sources of overhead dominate iterative GLM workflows:

1. **DataFrame → tabmat conversion** — ``tabmat.from_pandas`` runs
   narwhals dtype detection on every call. At 1M rows / 30 columns,
   this is ~250 ms each call.
2. **Standardize pass** — computing column means and standard
   deviations is O(n) per fit.
3. **Fold row-slicing** — for *k*-fold CV, slicing a 1M-row mixed
   ``SplitMatrix`` is ~330 ms per fold because each component matrix
   (numeric, sparse, categorical) must independently allocate its
   subset.

:class:`~glum.TabmatCache` eliminates (1) by memoizing per-column and
per-subset matrices, and (3) by caching per-fold row-slices with
incremental ``tabmat.hstack`` reuse when a new column set shares a
prefix with a cached one. :class:`~glum.StepwiseGLM` adds (2) by
caching standardize statistics per fold + column-set + sample-weight.


Quick example: two-stage forward selection
------------------------------------------

The recommended workflow is **score-test screening followed by
cross-validated refinement**:

.. code-block:: python

   import glum
   import pandas as pd

   df = pd.read_parquet("insurance.parquet")
   y  = df["ClaimNb"].to_numpy()
   offset = (df["Exposure"]).pipe(lambda s: s.clip(lower=1e-6)).pipe(lambda s: s.apply("log"))

   sglm = glum.StepwiseGLM(family="poisson", alpha=1e-4, drop_first=True)
   active = ["BonusMalus"]
   candidates = ["VehAge", "DrivAge", "Density", "VehPower",
                 "Area", "VehBrand", "VehGas", "Region"]

   sglm.fit(df[active], y, offset=offset.to_numpy())

   for step in range(6):
       # Stage 1: O(n) score test ranks all candidates.
       scores = sglm.screen_candidates(df, active, candidates)
       shortlist = [r.column for r in scores if r.pvalue < 0.05][:5]
       if not shortlist:
           break

       # Stage 2: CV on the shortlist only.
       cv_results = sglm.cv_select(
           df, active, shortlist, y, cv=5, n_alphas=15,
       )
       best = cv_results[0]
       if not best.selected:
           break

       active.append(best.column)
       candidates.remove(best.column)
       sglm.fit(df[active], y, offset=offset.to_numpy())

The score test costs one dot product per candidate (O(n)) vs a full
IRLS solve (O(n·p·iters)), so the shortlist stage scales to hundreds of
candidates cheaply. The CV stage then ranks the shortlist by hold-out
deviance using cached fold matrices.


Backward stepwise
-----------------

Backward elimination uses symmetric methods that score / CV-evaluate
each active column as a drop candidate:

.. code-block:: python

   # Score-test each active column as a drop candidate.
   # Sorted ascending by statistic: the lowest-impact column is first.
   drops = sglm.screen_drops(df, active)

   # CV-evaluate each "active \ {c}" reduced model.
   cv_drops = sglm.cv_select_drop(df, active, y, cv=5)
   worst    = cv_drops[0]   # smallest CV deviance increase → best to drop


Information criteria (AIC / BIC)
--------------------------------

Both :meth:`~glum.StepwiseGLM.cv_select` and
:meth:`~glum.StepwiseGLM.cv_select_drop` accept ``criterion=`` to sort
results by AIC or BIC instead of hold-out deviance:

.. code-block:: python

   results = sglm.cv_select(df, active, candidates, y,
                            cv=5, criterion="aic")
   # results[0] has the lowest cv_aic

The :class:`~glum.CVResult` objects always carry ``cv_deviance``,
``cv_aic`` and ``cv_bic`` regardless of the chosen criterion, so you
can re-rank without re-running the fit.


Looser tolerance for candidate screening
----------------------------------------

Candidate models inside ``cv_select`` only need enough convergence to
rank correctly, not full machine precision. Use ``screen_tol`` to cut
IRLS iterations on candidate fits:

.. code-block:: python

   cv_results = sglm.cv_select(
       df, active, shortlist, y,
       cv=5, n_alphas=15,
       screen_tol=1e-3,   # ~100x looser than glum's default
   )

Raise ``screen_tol`` if you observe ranking instability across runs.


Standalone use of TabmatCache
-----------------------------

:class:`~glum.TabmatCache` is independent of :class:`~glum.StepwiseGLM`
and can accelerate any workflow that calls ``tabmat.from_pandas``
multiple times on the same DataFrame:

.. code-block:: python

   from glum import TabmatCache, GeneralizedLinearRegressor

   cache = TabmatCache()
   cache.register_cols(df)

   X, _ = cache.get_subset(df, ["x0", "x1", "x2"])
   glm = GeneralizedLinearRegressor(family="poisson", alpha=0.01)
   glm.fit(X, y)

   # Persist for the next session:
   cache.save("model_cache.pkl")
   # ... next process
   cache = TabmatCache.load("model_cache.pkl")


Building a cache directly from parquet
--------------------------------------

For workflows that repeatedly fit a GLM on the same on-disk dataset —
daily model refits, scheduled scoring jobs, nested CV across sessions —
:meth:`~glum.TabmatCache.from_parquet` reads the file via :mod:`pyarrow`,
applies dictionary encoding to any ``cat_cols``, and binds the file's
fingerprint to the cache.  The fingerprint persists across
:meth:`~glum.TabmatCache.save` / :meth:`~glum.TabmatCache.load` so a
later session can verify that the underlying parquet hasn't changed.

.. code-block:: python

   from glum import TabmatCache, GeneralizedLinearRegressor, fingerprint_file

   # First session: build and persist the cache
   cache = TabmatCache.from_parquet(
       "data/insurance.parquet",
       columns=["ClaimNb", "VehAge", "DrivAge", "BonusMalus",
                "Area", "VehBrand", "VehGas", "Region"],
       cat_cols=["Area", "VehBrand", "VehGas", "Region"],
   )
   y = cache.source_df["ClaimNb"].to_numpy().astype(float)
   X, _ = cache.get_subset(
       cache.source_df,
       ["VehAge", "DrivAge", "BonusMalus", "Region"],
   )
   GeneralizedLinearRegressor(family="poisson", alpha=0.01).fit(X, y)
   cache.save("warm.pkl")

   # Next session: skip pyarrow + pandas + tabmat reconstruction entirely
   cache = TabmatCache.load("warm.pkl")
   cache.verify_source(fingerprint_file("data/insurance.parquet"))
   # ↑ raises SourceFingerprintError if the parquet was modified
   #   (size or mtime changed); otherwise returns True.

The fingerprint is ``("file", absolute_path, size_bytes, mtime_ns)`` —
sub-millisecond to compute, no file read.  Catches normal edits,
replacements, and appends.  Does **not** catch in-place rewrites that
preserve size and mtime; for those, hash the file contents yourself
and pass the tuple via :meth:`~glum.TabmatCache.set_source_fingerprint`.

Note that :attr:`~glum.TabmatCache.source_df` is a convenience
attribute set by :meth:`from_parquet` so you can immediately call
:meth:`get_subset`.  It is **not** pickled — on a re-load you must
re-read the parquet yourself if you want a pandas view of the data.
The cached tabmat matrices, however, are fully usable without ever
touching pandas again::

   cache = TabmatCache.load("warm.pkl")
   # No source_df, no pandas read — just use the cached matrices.
   X = cache._subset_cache[("VehAge", "DrivAge", "BonusMalus", "Region")]
   GeneralizedLinearRegressor(family="poisson", alpha=0.01).fit(X, y)


Limitations
-----------

* :class:`~glum.TabmatCache` detects DataFrame **reassignment**, **row
  count changes**, and **column rename / reorder** via a lightweight
  fingerprint (``id(df)``, ``df.shape``, column-name hash). It does
  **not** detect in-place value mutation (``df.loc[0, 'x'] = ...``).
  Call :meth:`~glum.TabmatCache.clear` after any such mutation.
* Concurrent :meth:`~glum.StepwiseGLM.fit` calls from multiple threads
  are serialized through a module-level lock around the standardize
  patch. Each individual fit retains its full caching speedup; the
  lock guarantees correctness, not parallelism.


API reference
-------------

See the :doc:`API Reference <glm>` for class signatures.
:class:`~glum.StepwiseGLM`, :class:`~glum.TabmatCache`,
:class:`~glum.ScoreTestResult`, :class:`~glum.CVResult` and
:class:`~glum.CacheVersionError` are all exported from the top-level
:mod:`glum` namespace.
