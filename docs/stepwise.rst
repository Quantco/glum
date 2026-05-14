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

Caching behavior: cold start vs. warm start
-------------------------------------------

The :class:`~glum.TabmatCache` is a *value object* — a populated set of
tabmat matrices plus the fingerprint of the parquet that produced them.
On the first call, :func:`~glum.managed_cache` reads the source file,
constructs the matrices, and persists the bundle to the backend.  On
every subsequent call, it loads the bundle, verifies the source hasn't
changed, and yields the warm cache without touching tabmat construction
at all.

Cold start (cache file does not exist)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. ``backend.exists(key)`` returns ``False``.
2. :meth:`~glum.TabmatCache.from_parquet` runs:

   - ``pyarrow.parquet.read_table(path, columns=...)`` reads only the
     requested columns from disk.
   - Each ``cat_cols`` entry is dictionary-encoded in the Arrow table.
   - ``table.to_pandas()`` produces a DataFrame with proper
     ``pd.Categorical`` dtypes on the dictionary-encoded columns.
   - :func:`~glum.fingerprint_file` records the source identity via
     ``os.stat`` — no file content read.
   - :meth:`~glum.TabmatCache.register_cols` builds a single-column
     tabmat sub-matrix for each registerable column via
     ``tabmat.from_pandas``.

3. The yielded cache has ``source_df`` populated eagerly and
   ``_col_matrices`` populated for every registerable column.

4. ``cache.read_target("ClaimNb")`` runs an independent
   ``pyarrow.parquet.read_table(path, columns=["ClaimNb"])``.  This is
   *not* memoized — every call hits disk.

5. ``cache.source_df`` returns the in-memory frame immediately.

6. ``cache.get_subset(df, [...])`` finds the subset key absent from
   ``_subset_cache``.  Because every requested column is already in
   ``_col_matrices``, it assembles the subset via ``tabmat.hstack``
   (cheap; no narwhals dtype detection) and stores the result.

7. The GLM ``.fit(X, y)`` call runs IRLS on the assembled
   ``StandardizedMatrix``.

8. On clean exit, ``save_on_exit="success"`` invokes
   :meth:`~glum.TabmatCache.save_to`: build the state dict,
   ``joblib.dump`` into a bytes buffer with ``compress=3``, then
   ``backend.write(key, ...)``.

Warm start (cache file exists from a prior run)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. ``backend.exists(key)`` returns ``True``.
2. :meth:`~glum.TabmatCache.load_from` runs:

   - ``backend.read(key)`` returns the bytes.
   - ``joblib.load`` deserializes the state dict in memory.
   - ``_restore_from_state`` constructs a fresh cache and rebinds all
     the persisted dicts — ``_col_matrices``, ``_subset_cache``,
     ``_fold_mat``, ``_std_stats``, plus ``_source_fingerprint``,
     ``_source_columns``, and ``_source_cat_cols`` rehydration
     metadata.  **No tabmat conversion runs.**

3. :func:`~glum.fingerprint_file` re-stats the source file; the
   resulting tuple is compared by equality against
   ``cache.source_fingerprint``.  Match → yield.  Mismatch → silently
   rebuild from parquet (``rebuild_on_mismatch=True``) or raise.

4. ``cache.read_target("ClaimNb")`` still hits disk.  This is
   *intentional*: the target column may be updated daily even when the
   feature columns are stable, and a cache miss is cheaper than a
   stale ``y``.

5. ``cache.source_df`` is **lazy** post-load.  The backing field is
   ``None`` until first access; the property then re-reads the parquet
   using the recorded ``columns`` and ``cat_cols``, runs
   ``verify_source``, and caches the DataFrame in memory for the
   remainder of the session.  Skip this step entirely if you never
   touch ``source_df``.

6. ``cache.get_subset(df, [...])`` finds the subset key already
   present in ``_subset_cache``.  Returns the cached matrix in O(1)
   via a dict lookup.  No ``from_pandas``, no ``hstack``, no work.

7. GLM ``.fit(X, y)`` runs IRLS — same cost as cold-start.

8. On clean exit, ``save_on_exit="success"`` re-pickles and overwrites
   the cache file unconditionally.  There is no dirty-flag detection.

Flowchart
~~~~~~~~~

.. code-block:: text

                  managed_cache(parquet, ...) __enter__
                                 |
                                 v
                    backend.exists(key) ?
                          /            \
                       No /              \ Yes
                        /                  \
                       v                    v
            TabmatCache.from_parquet     load_from(backend, key)
                       |                    |
                       v                    v
            pq.read_table + to_pandas    joblib.load(state)
                       |                    |
                       v                    v
              register_cols(df, cols)    verify_source(file_fp)
                       |                    /      \
                       |                  ok       mismatch
                       |                  /            \
                       |                 v              v
                       |              return     rebuild_on_mismatch?
                       |              cache           /     \
                       |                            Yes      No
                       |                            /          \
                       |                           v            v
                       |                       (cold path)   raise
                       |                          |
                       v                          |
                  +-----<--------------------------+
                  |
                  v
              yield cache  ----->  user code:
                                     y     = cache.read_target("y")   [parquet]
                                     df    = cache.source_df          [O(1) cold | parquet warm]
                                     X, _  = cache.get_subset(df,...) [hstack cold | O(1) warm]
                                     GLM.fit(X, y)                    [IRLS]
                  |
                  v
              __exit__ (save_on_exit policy)
                  |
                  v
              cache.save_to(backend, key)   ->   re-pickle + overwrite

Cost summary
~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 25 25 20

   * - Operation
     - Cold (no cache)
     - Warm (cache hit)
     - Notes
   * - Backend exists / load
     - 0 (skipped)
     - ~100–300 ms
     - dominated by ``joblib.load``
   * - Source fingerprint check
     - 0 (skipped)
     - ~1 ms
     - ``os.stat`` + tuple eq
   * - Build all per-column matrices
     - ~10–50 ms × n_cols
     - 0 (loaded from pickle)
     - ``register_cols``
   * - ``read_target``
     - ~50–200 ms
     - ~50–200 ms
     - *every call* hits parquet
   * - ``source_df`` first access
     - ~0 (already in mem)
     - ~500–1000 ms
     - lazy re-read on warm
   * - ``get_subset`` for a known column set
     - ~5–20 ms (hstack)
     - <1 ms (dict hit)
     -
   * - GLM IRLS solve
     - same
     - same
     - the actual work
   * - ``__exit__`` save
     - ~400–1000 ms
     - ~400–1000 ms
     - re-pickles unconditionally

Two performance gotchas worth knowing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. :meth:`~glum.TabmatCache.read_target` reads from the parquet on
   every call.  If you're iterating over candidates in a stepwise loop
   and the target doesn't change, hoist ``y = cache.read_target(...)``
   out of the inner loop.

2. ``managed_cache(..., save_on_exit="success")`` re-pickles the cache
   file on every clean exit, even when nothing changed.  For
   high-frequency batch jobs (e.g., a scheduled inference task that
   runs the cache through a single fit and exits cleanly), pass
   ``save_on_exit="never"`` to skip the rewrite.

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
