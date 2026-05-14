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


Cross-session caching with ``managed_cache``
--------------------------------------------

For workflows that repeatedly fit a GLM on the same on-disk dataset —
daily model refits, scheduled scoring jobs, nested CV across sessions —
:func:`~glum.managed_cache` wraps the entire load-or-build-then-save
lifecycle in a single ``with`` block.  On entry it loads a previously
saved cache and verifies that the source file is unchanged, or rebuilds
from scratch when needed.  On clean exit it persists the cache so the
next session starts warm.

.. code-block:: python

   from glum import managed_cache, GeneralizedLinearRegressor

   with managed_cache(
       "data/insurance.parquet",
       cat_cols=["Region", "Area", "VehBrand", "VehGas"],
       columns=["ClaimNb", "VehAge", "DrivAge", "BonusMalus", "Region"],
   ) as cache:
       y = cache.read_target("ClaimNb")
       X, _ = cache.get_subset(
           cache.source_df,
           ["VehAge", "DrivAge", "BonusMalus", "Region"],
       )
       GeneralizedLinearRegressor(family="poisson", alpha=0.01).fit(X, y)
   # Cache auto-persisted because the with-block exited cleanly.

The first invocation reads the parquet via :mod:`pyarrow`, builds tabmat
matrices, and persists the cache to ``./.tabmat_cache/insurance.pkl``
(the default location).  Subsequent invocations load the cache, verify
that ``data/insurance.parquet`` is unchanged (``size_bytes`` and
``mtime_ns``), and yield the warm cache — no pyarrow read, no tabmat
construction.  If the parquet has changed, ``managed_cache`` silently
rebuilds (or raises :class:`~glum.SourceFingerprintError` if you pass
``rebuild_on_mismatch=False``).

``save_on_exit`` controls persistence policy:

- ``"success"`` (default): persist only if the with-block exited cleanly.
- ``"always"``: persist regardless of exceptions.
- ``"never"``: discard whatever state accumulated inside the block.

Backend extensibility
---------------------

The default backend writes to a local directory, but
:class:`~glum.CacheBackend` is a four-method ``Protocol`` — any object
satisfying ``exists``, ``read``, ``write``, ``delete`` (all taking and
returning ``bytes``) is a valid backend.  This is the seam for future
distributed cache support (Redis, S3, Azure Blob).  The protocol is
deliberately bytes-only so backends don't need to depend on
:mod:`joblib` or pickle semantics — serialization is the cache's
responsibility.

.. code-block:: python

   from glum import managed_cache, LocalFileBackend

   # Custom backend (e.g. shared NFS mount)
   backend = LocalFileBackend("/mnt/shared/glum_caches")

   with managed_cache(
       "data/insurance.parquet",
       backend=backend,
       key="prod/insurance.pkl",
       cat_cols=["Region"],
   ) as cache:
       ...

To implement your own backend (e.g. for Redis), satisfy the four-method
protocol — see :class:`~glum.CacheBackend`.

Lower-level building blocks
---------------------------

``managed_cache`` is sugar over three lower-level primitives that
remain available for advanced use:

- :meth:`~glum.TabmatCache.from_parquet` — build a cache directly from
  a parquet file, returning an instance with the file fingerprint
  bound but no persistence applied.
- :meth:`~glum.TabmatCache.save_to` / :meth:`~glum.TabmatCache.load_from`
  — backend-aware versions of :meth:`~glum.TabmatCache.save` /
  :meth:`~glum.TabmatCache.load`.
- :func:`~glum.fingerprint_file` — return the
  ``("file", path, size, mtime_ns)`` tuple used by
  :meth:`~glum.TabmatCache.verify_source`.

The :attr:`~glum.TabmatCache.source_df` attribute is **lazy** after
:meth:`~glum.TabmatCache.load`: accessing it re-reads the bound parquet
on first use (with fingerprint verification) and caches the result for
subsequent calls.  Likewise :meth:`~glum.TabmatCache.read_target` reads
a single response column from the bound source without materializing a
DataFrame, useful for extracting ``y`` cheaply.


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
