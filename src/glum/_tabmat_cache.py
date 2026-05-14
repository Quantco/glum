"""
TabmatCache
===========

Reusable cache for ``tabmat`` matrix construction and fold subsetting.

Designed for workflows that repeatedly fit GLMs on the same DataFrame with
varying column subsets, row subsets (e.g. CV folds), or both — such as
stepwise selection, grid search, nested CV, ensemble fitting, or daily model
refits.

Two complementary caches live on a single instance:

1. **Column-set matrix cache** — ``get_subset(df, cols)`` memoizes the
   result of :func:`tabmat.from_pandas`.  The narwhals dtype-detection
   overhead is paid at most once per unique column tuple.  Single-column
   sub-matrices are also stored independently via ``register_cols`` for
   per-column operations such as score-test screening.

2. **Fold row-slice cache** — ``get_fold_slice(fold_id, cols)`` memoizes
   the row-subset of a column-set matrix.  When a new column set shares a
   *prefix* with a cached fold slice, the new slice is built via
   ``tabmat.hstack`` of the cached prefix and the new column's fold slice,
   which is ~10× cheaper than re-slicing the full matrix at large n.  The
   fold-slice store is bounded by an LRU policy (``fold_mat_maxsize``).

Persistence
-----------
``cache.save(path)`` writes the populated cache to disk via :mod:`joblib`.
``TabmatCache.load(path)`` returns a new cache populated from the file.
This enables sessions to start with a fully warm cache, eliminating
``from_pandas`` and fold-slice cost on repeated workflows.  Compatibility is
guarded by a ``__cache_version__`` sentinel — incompatible files raise
:class:`CacheVersionError`.

Limitations
-----------
- The cache keys by **column name tuples** and (for folds) integer fold
  indices.  If you mutate the underlying DataFrame between calls (changing
  values, adding rows, renaming columns), the cache does not detect the
  change and will return stale matrices.  Call :meth:`clear` after any
  such mutation, or use a fresh cache.
- Categorical matrices store integer codes, not level names.  Saving and
  re-loading preserves the codes; if you later need the human-readable
  category names you should keep the original DataFrame's category levels
  available alongside the cache.

Examples
--------
Stand-alone use with a vanilla GLM::

    from glum import TabmatCache, GeneralizedLinearRegressor

    cache = TabmatCache()
    cache.register_cols(df)                   # warm per-column store
    X_tab, names = cache.get_subset(df, ["x0", "x1", "cat_a"])

    glm = GeneralizedLinearRegressor(family="poisson", alpha=0.01)
    glm.fit(X_tab, y)

    cache.save("model_cache.pkl")             # persist for next session
"""

from __future__ import annotations

import logging
import os
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Sequence, Union

import joblib
import numpy as np
import pandas as pd
import tabmat as tm

__all__ = [
    "TabmatCache",
    "CacheVersionError",
    "SourceFingerprintError",
    "fingerprint_file",
]

_logger = logging.getLogger(__name__)

# Bumped whenever the on-disk layout changes incompatibly.
_CACHE_VERSION = 2   # bumped: added source_fingerprint to saved state


class CacheVersionError(RuntimeError):
    """Raised when a saved ``TabmatCache`` has an incompatible version."""


class SourceFingerprintError(RuntimeError):
    """Raised when a verify_source check fails — the underlying file changed."""


def fingerprint_file(path: Union[str, Path]) -> tuple:
    """
    Lightweight fingerprint of a file at ``path``.

    Returns ``("file", absolute_path, size_bytes, mtime_ns)`` — cheap
    (sub-millisecond, no read), portable across processes, and survives
    save/load.  Catches ordinary edits, file replacements, and appends.

    Does **not** catch malicious or buggy in-place rewrites that
    preserve size and mtime.  For those, hash the file contents
    separately and pass a tuple like ``("sha256", hex_digest)`` to
    :meth:`TabmatCache.set_source_fingerprint`.
    """
    p = Path(path).resolve()
    st = os.stat(p)
    return ("file", str(p), int(st.st_size), int(st.st_mtime_ns))


# Internal alias preserved so other modules can use the older name.
_file_fingerprint = fingerprint_file


def _df_fingerprint(df: pd.DataFrame) -> tuple:
    """
    Lightweight fingerprint for cache-invalidation purposes.

    Captures DataFrame **identity**, **shape**, and the **column names**.
    This detects:

    * reassignment (``df = df.copy()`` changes ``id(df)``),
    * row addition / removal (``df.shape`` changes), and
    * column rename / reorder (column-name tuple hash changes).

    It does **not** detect in-place value mutation
    (``df.loc[0, "x"] = ...``).  That would require hashing column data
    on every call, which is too expensive for a hot-path cache.  Users
    who mutate in place must call :meth:`TabmatCache.clear` manually.
    """
    return (id(df), df.shape, hash(tuple(df.columns)))


class TabmatCache:
    """
    Two-level tabmat matrix cache: column subsets + fold row-slices.

    Parameters
    ----------
    fold_mat_maxsize : int, default 256
        Maximum number of entries kept in the fold row-slice cache.
        When exceeded, the least-recently-inserted entry is evicted.
        Set to ``0`` to disable the bound (not recommended for very long
        workflows — the cache grows unboundedly).

    Attributes
    ----------
    fold_mat_maxsize : int
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, fold_mat_maxsize: int = 256):
        if fold_mat_maxsize < 0:
            raise ValueError("fold_mat_maxsize must be non-negative")
        self.fold_mat_maxsize = fold_mat_maxsize

        # ── Column-set store ────────────────────────────────────────────
        # col_name → single-column MatrixBase (for per-column lookups)
        self._col_matrices:   dict[str, tm.MatrixBase] = {}
        # col_name → expanded feature names for that column
        self._col_feat_names: dict[str, list[str]]     = {}
        # tuple(cols) → full column-subset MatrixBase
        self._subset_cache:   dict[tuple, tm.MatrixBase] = {}
        # tuple(cols) → expanded feature names for that subset
        self._subset_names:   dict[tuple, list[str]]    = {}

        # ── Fold row-slice store ────────────────────────────────────────
        # (fold_id, tuple(cols)) → fold-sliced MatrixBase  (LRU-bounded)
        self._fold_mat:       OrderedDict[tuple, tm.MatrixBase] = OrderedDict()
        # fold_id → training-row indices
        self._fold_idx:       dict[int, np.ndarray]    = {}
        # (fold_id, tuple(cols), sw_id) → (col_means, col_stds)
        self._std_stats:      dict[tuple, tuple]       = {}

        # ── Mutation-detection fingerprint ──────────────────────────────
        # Lightweight signature of the DataFrame that produced our cached
        # matrices.  Reset to None means "no DataFrame seen yet".  When a
        # subsequent call arrives with a different fingerprint, all cached
        # state is invalidated and the new DataFrame becomes the source of
        # truth.  See `_df_fingerprint` for what is (and isn't) detected.
        self._df_fingerprint: Optional[tuple] = None

        # ── Source-of-truth fingerprint ─────────────────────────────────
        # Optional cross-session identity tag for the data that produced
        # these matrices.  Typically a file fingerprint (``_file_fingerprint``)
        # set by :meth:`from_parquet`, but any opaque hashable tag is
        # accepted via :meth:`set_source_fingerprint`.  Survives
        # ``save()`` / ``load()`` so a freshly-loaded cache can be
        # verified against the original data file in a new session.
        self._source_fingerprint: Optional[tuple] = None

    # ------------------------------------------------------------------
    # Column / subset layer
    # ------------------------------------------------------------------

    def _check_fingerprint(self, df: pd.DataFrame) -> None:
        """
        Validate the cache fingerprint against ``df``.

        On first contact, record the fingerprint.  On any subsequent
        call with a mismatched fingerprint, clear all cached state and
        log an info-level message.  This catches DataFrame reassignment,
        row count changes, and column rename/reorder — but not in-place
        value mutation (see :func:`_df_fingerprint`).
        """
        fp = _df_fingerprint(df)
        if self._df_fingerprint is not None and fp != self._df_fingerprint:
            _logger.info(
                "TabmatCache: DataFrame fingerprint changed "
                "(was %r, now %r); invalidating cache.",
                self._df_fingerprint, fp,
            )
            self.clear()
        self._df_fingerprint = fp

    def register_cols(self, df: pd.DataFrame) -> None:
        """
        Build per-column tabmat sub-matrices for every column in ``df``
        that is not yet registered.  Subsequent ``get_col(name)`` calls
        return the cached single-column matrix in O(1).
        """
        self._check_fingerprint(df)
        for col in df.columns:
            if col not in self._col_matrices:
                mat = tm.from_pandas(df[[col]])
                self._col_matrices[col]   = mat
                self._col_feat_names[col] = mat.get_names(
                    type="column", missing_prefix=f"_{col}_"
                )

    def get_col(self, col: str) -> tm.MatrixBase:
        """Return the cached single-column matrix for ``col``."""
        return self._col_matrices[col]

    def col_feat_names(self, col: str) -> list[str]:
        """Return the expanded feature names for column ``col``."""
        return self._col_feat_names[col]

    def get_subset(
        self, df: pd.DataFrame, cols: list[str]
    ) -> tuple[tm.MatrixBase, list[str]]:
        """
        Return ``(MatrixBase, expanded_feature_names)`` for a column subset.

        Memoized by ``tuple(cols)``.  On a cache miss, the subset is
        assembled from the per-column store via :func:`tabmat.hstack` if
        every column is already registered (~4× faster than re-running
        :func:`tabmat.from_pandas`); otherwise it falls back to
        :func:`tabmat.from_pandas` on ``df[cols]`` and back-fills the
        per-column store for next time.  Underlying numpy arrays are
        shared by reference, so the subset cache adds no meaningful
        memory overhead.
        """
        self._check_fingerprint(df)
        key = tuple(cols)
        if key not in self._subset_cache:
            if all(c in self._col_matrices for c in cols):
                # Fast path: assemble from cached per-column matrices.
                mat = tm.hstack([self._col_matrices[c] for c in cols])
            else:
                # Cold path: build via from_pandas and back-fill the
                # per-column store so subsequent subsets are fast.
                mat = tm.from_pandas(df[cols])
                for c in cols:
                    if c not in self._col_matrices:
                        self._col_matrices[c] = tm.from_pandas(df[[c]])
                        self._col_feat_names[c] = self._col_matrices[c].get_names(
                            type="column", missing_prefix=f"_{c}_"
                        )
            self._subset_cache[key] = mat
            self._subset_names[key] = mat.get_names(
                type="column", missing_prefix="_col_"
            )
        return self._subset_cache[key], self._subset_names[key]

    def __contains__(self, col: str) -> bool:
        return col in self._col_matrices

    # ------------------------------------------------------------------
    # Fold layer
    # ------------------------------------------------------------------

    def set_fold_indices(self, fold_id: int, train_idx: np.ndarray) -> None:
        """Register training-row indices for a fold (call once per fold)."""
        self._fold_idx[fold_id] = np.asarray(train_idx)

    def get_fold_slice(
        self,
        fold_id: int,
        cols: list[str],
        full_mat: tm.MatrixBase,
    ) -> tm.MatrixBase:
        """
        Return the fold-training row-slice of ``full_mat`` for ``cols``.

        Resolution strategy:

        1. Exact cache hit → O(1) ``OrderedDict`` lookup (and the entry is
           bumped to the most-recently-used position).
        2. **Prefix hit** — the longest cached entry for this fold whose
           column tuple is a prefix of ``cols`` is reused; the missing
           columns are sliced from the per-column store and appended via
           ``tabmat.hstack``.  ~10× faster than a full re-slice.
        3. Full miss — calls ``full_mat[train_idx, :]``.

        After insertion, the LRU cache may evict the oldest entry if
        ``len(self._fold_mat) > self.fold_mat_maxsize`` (and maxsize > 0).
        """
        key = (fold_id, tuple(cols))
        if key in self._fold_mat:
            self._fold_mat.move_to_end(key)
            return self._fold_mat[key]

        train_idx = self._fold_idx[fold_id]

        # Find the longest cached prefix for this fold
        best_prefix:    Optional[tuple]           = None
        best_prefix_mat: Optional[tm.MatrixBase]  = None
        for cached_key, cached_mat in self._fold_mat.items():
            if cached_key[0] != fold_id:
                continue
            cached_cols = cached_key[1]
            n = len(cached_cols)
            if (
                tuple(cols[:n]) == cached_cols
                and n > (len(best_prefix) if best_prefix else -1)
            ):
                best_prefix     = cached_cols
                best_prefix_mat = cached_mat

        if best_prefix is not None and best_prefix_mat is not None:
            # Incrementally append the new columns
            new_cols = cols[len(best_prefix):]
            pieces:   list[tm.MatrixBase] = [best_prefix_mat]
            ok = True
            for c in new_cols:
                if c in self._col_matrices:
                    pieces.append(self._col_matrices[c][train_idx, :])
                else:
                    # Can't build incrementally — fall through to full slice
                    ok = False
                    break
            if ok:
                mat = tm.hstack(pieces) if len(pieces) > 1 else pieces[0]
                self._insert_fold_mat(key, mat)
                return mat

        # Full miss — row-slice the full matrix
        mat = full_mat[train_idx, :]
        self._insert_fold_mat(key, mat)
        return mat

    def _insert_fold_mat(self, key: tuple, mat: tm.MatrixBase) -> None:
        """Insert into fold-mat cache, evicting oldest entry if needed."""
        self._fold_mat[key] = mat
        if (
            self.fold_mat_maxsize > 0
            and len(self._fold_mat) > self.fold_mat_maxsize
        ):
            self._fold_mat.popitem(last=False)   # FIFO/LRU evict-oldest

    def get_std_stats(
        self, fold_id: int, cols: list[str], sw_id: int
    ) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """
        Return cached ``(col_means, col_stds)`` for a fold + column-set +
        sample-weight identity, or ``None`` on miss.
        """
        return self._std_stats.get((fold_id, tuple(cols), sw_id))

    def set_std_stats(
        self,
        fold_id: int,
        cols: list[str],
        sw_id: int,
        means: np.ndarray,
        stds: np.ndarray,
    ) -> None:
        """Store ``(col_means, col_stds)`` for a fold + column-set + sw_id."""
        self._std_stats[(fold_id, tuple(cols), sw_id)] = (
            np.atleast_1d(np.asarray(means, dtype=float).ravel()),
            np.atleast_1d(np.asarray(stds,  dtype=float).ravel()),
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """
        Return a snapshot of cache contents:

        - ``n_cols``: number of per-column matrices registered
        - ``n_subsets``: number of column-set matrices cached
        - ``n_fold_slices``: number of fold slices cached
        - ``n_std_stats``: number of cached fold standardize stats
        - ``n_folds``: number of folds with registered indices
        - ``fold_mat_maxsize``: configured LRU bound
        """
        return {
            "n_cols":           len(self._col_matrices),
            "n_subsets":        len(self._subset_cache),
            "n_fold_slices":    len(self._fold_mat),
            "n_std_stats":      len(self._std_stats),
            "n_folds":          len(self._fold_idx),
            "fold_mat_maxsize": self.fold_mat_maxsize,
        }

    def clear(self) -> None:
        """Drop all cached state (columns, subsets, fold slices, std stats)."""
        self._col_matrices.clear()
        self._col_feat_names.clear()
        self._subset_cache.clear()
        self._subset_names.clear()
        self._fold_mat.clear()
        self._fold_idx.clear()
        self._std_stats.clear()
        self._df_fingerprint = None
        self._source_fingerprint = None

    def clear_fold_slices(self) -> None:
        """Drop only the fold row-slice cache (keep column-set matrices)."""
        self._fold_mat.clear()
        self._std_stats.clear()

    # ------------------------------------------------------------------
    # Source-of-truth fingerprinting (cross-session)
    # ------------------------------------------------------------------

    def set_source_fingerprint(self, fingerprint: tuple) -> None:
        """
        Attach an opaque source fingerprint to this cache.

        The fingerprint is stored on the instance and persisted via
        :meth:`save`.  Use :meth:`verify_source` to check it against a
        fresh fingerprint of the original data.  Use whatever hashable
        tuple uniquely identifies your source — a file fingerprint, a
        content hash, a dataset version string, etc.
        """
        self._source_fingerprint = fingerprint

    @property
    def source_fingerprint(self) -> Optional[tuple]:
        """The currently-bound source fingerprint, or ``None`` if unset."""
        return self._source_fingerprint

    def verify_source(
        self,
        expected: tuple,
        strict: bool = True,
    ) -> bool:
        """
        Check the bound source fingerprint against ``expected``.

        Parameters
        ----------
        expected : tuple
            Fingerprint produced from the user's current source of
            truth.  For files, use :func:`_file_fingerprint` (exposed
            indirectly via :meth:`from_parquet`'s ``source_path``).
        strict : bool, default True
            If True, raise :class:`SourceFingerprintError` on mismatch.
            If False, return ``False`` instead so the caller can decide
            how to react.

        Returns
        -------
        bool
            ``True`` if the bound fingerprint matches ``expected``.
        """
        if self._source_fingerprint is None:
            if strict:
                raise SourceFingerprintError(
                    "No source fingerprint is bound to this cache; "
                    "call set_source_fingerprint() or build via "
                    "TabmatCache.from_parquet() first."
                )
            return False

        if self._source_fingerprint != expected:
            if strict:
                raise SourceFingerprintError(
                    f"Source fingerprint mismatch:\n"
                    f"  bound:    {self._source_fingerprint!r}\n"
                    f"  expected: {expected!r}\n"
                    "The underlying data appears to have changed since "
                    "this cache was built."
                )
            return False
        return True

    # ------------------------------------------------------------------
    # Parquet ingestion (cross-session warm start without pandas in user code)
    # ------------------------------------------------------------------

    @classmethod
    def from_parquet(
        cls,
        path: Union[str, Path],
        columns: Optional[Sequence[str]] = None,
        cat_cols: Optional[Sequence[str]] = None,
        fold_mat_maxsize: int = 256,
        register_cols: bool = True,
    ) -> "TabmatCache":
        """
        Build a :class:`TabmatCache` directly from a parquet file.

        Reads the parquet via :mod:`pyarrow`, applies dictionary encoding
        to any ``cat_cols`` (so they surface as ``pd.Categorical`` after
        ``to_pandas``), converts to pandas, and pre-populates the cache.
        The source file's fingerprint is bound to the cache so
        :meth:`verify_source` works across sessions.

        Parameters
        ----------
        path : str or Path
            Parquet file to ingest.
        columns : sequence of str, optional
            Subset of columns to load.  If ``None``, all columns are
            loaded.
        cat_cols : sequence of str, optional
            Columns to treat as categorical.  Each named column is
            dictionary-encoded inside the arrow Table before conversion,
            which becomes a ``pd.Categorical`` automatically.  Columns
            already stored as dictionary-encoded in parquet are detected
            by their arrow dtype and converted regardless.
        fold_mat_maxsize : int, default 256
            See :class:`TabmatCache`.
        register_cols : bool, default True
            If True, immediately call :meth:`register_cols` so the
            per-column cache is populated.  Set ``False`` if you intend
            to use only a subset of columns and want to defer the
            per-column build.

        Returns
        -------
        TabmatCache
            Cache instance with the file fingerprint bound.  Ready for
            :meth:`get_subset` calls passing the same DataFrame (which
            this method exposes via :attr:`source_df` for convenience).

        Notes
        -----
        ``pyarrow`` is a required dependency of glum, so this method has
        no extra imports.  We still go through pandas as the final
        narwhals-friendly DataFrame for ``tabmat.from_df``; the saving
        relative to a naive ``pd.read_parquet`` is that we can
        selectively dictionary-encode categorical columns without
        re-allocating string arrays.
        """
        import pyarrow.parquet as pq
        import pyarrow as pa

        p = Path(path).resolve()
        fp = _file_fingerprint(p)

        # Read only the requested columns (cheaper than full read).
        table = pq.read_table(p, columns=list(columns) if columns else None)

        # Dictionary-encode anything the user marked categorical AND any
        # column already stored as a parquet dictionary type.
        cat_set = set(cat_cols or [])
        for name in table.column_names:
            col = table.column(name)
            already_dict = pa.types.is_dictionary(col.type)
            if name in cat_set and not already_dict:
                table = table.set_column(
                    table.column_names.index(name),
                    name,
                    col.dictionary_encode(),
                )

        # Convert to pandas: dictionary columns surface as pd.Categorical.
        df = table.to_pandas()

        cache = cls(fold_mat_maxsize=fold_mat_maxsize)
        cache.set_source_fingerprint(fp)
        if register_cols:
            # Only register columns whose pandas dtype is tabmat-compatible:
            # numeric, boolean, or pd.Categorical.  Plain string / object
            # columns are silently skipped — the user opted out of treating
            # them as categorical by not naming them in ``cat_cols``.
            registerable = [
                c for c in df.columns
                if (
                    isinstance(df[c].dtype, pd.CategoricalDtype)
                    or pd.api.types.is_numeric_dtype(df[c])
                    or pd.api.types.is_bool_dtype(df[c])
                )
            ]
            if registerable:
                cache.register_cols(df[registerable])
        # Stash the DataFrame on the cache for convenience: users who
        # want to call get_subset can pull it off without re-reading the
        # parquet.  We deliberately do NOT pickle this on save().
        cache.source_df = df
        return cache

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """
        Serialize the cache to ``path`` via :func:`joblib.dump`.

        The format includes a version sentinel; :meth:`load` will raise
        :class:`CacheVersionError` if the layout is incompatible.
        """
        # We deliberately omit `_df_fingerprint`: id(df) is not meaningful
        # across processes, and df.shape / column-name hash should be
        # re-established by the user on first contact in the new session.
        # We DO persist `_source_fingerprint` — its whole purpose is
        # cross-session identity for the underlying data file/dataset.
        state = {
            "__cache_version__":   _CACHE_VERSION,
            "fold_mat_maxsize":    self.fold_mat_maxsize,
            "col_matrices":        self._col_matrices,
            "col_feat_names":      self._col_feat_names,
            "subset_cache":        self._subset_cache,
            "subset_names":        self._subset_names,
            "fold_mat":            dict(self._fold_mat),   # OrderedDict → dict
            "fold_idx":            self._fold_idx,
            "std_stats":           self._std_stats,
            "source_fingerprint":  self._source_fingerprint,
        }
        joblib.dump(state, str(path), compress=3)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "TabmatCache":
        """
        Load a previously-saved cache from ``path``.

        Raises
        ------
        CacheVersionError
            If the file was saved with an incompatible cache version.
        """
        state = joblib.load(str(path))
        version = state.get("__cache_version__")
        if version != _CACHE_VERSION:
            raise CacheVersionError(
                f"Saved cache version is {version!r}, "
                f"expected {_CACHE_VERSION!r}.  "
                "Re-build the cache from the source DataFrame."
            )

        cache = cls(fold_mat_maxsize=state["fold_mat_maxsize"])
        cache._col_matrices       = state["col_matrices"]
        cache._col_feat_names     = state["col_feat_names"]
        cache._subset_cache       = state["subset_cache"]
        cache._subset_names       = state["subset_names"]
        cache._fold_mat           = OrderedDict(state["fold_mat"])
        cache._fold_idx           = state["fold_idx"]
        cache._std_stats          = state["std_stats"]
        cache._source_fingerprint = state.get("source_fingerprint")
        return cache
