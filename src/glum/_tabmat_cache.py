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
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Union

import joblib
import numpy as np
import pandas as pd
import tabmat as tm

__all__ = ["TabmatCache", "CacheVersionError"]

_logger = logging.getLogger(__name__)

# Bumped whenever the on-disk layout changes incompatibly.
_CACHE_VERSION = 1


class CacheVersionError(RuntimeError):
    """Raised when a saved ``TabmatCache`` has an incompatible version."""


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

    # ------------------------------------------------------------------
    # Column / subset layer
    # ------------------------------------------------------------------

    def register_cols(self, df: pd.DataFrame) -> None:
        """
        Build per-column tabmat sub-matrices for every column in ``df``
        that is not yet registered.  Subsequent ``get_col(name)`` calls
        return the cached single-column matrix in O(1).
        """
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

        Memoized by ``tuple(cols)``.  On miss, calls
        :func:`tabmat.from_pandas` on ``df[cols]`` and stores the result.
        """
        key = tuple(cols)
        if key not in self._subset_cache:
            mat = tm.from_pandas(df[cols])
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

    def clear_fold_slices(self) -> None:
        """Drop only the fold row-slice cache (keep column-set matrices)."""
        self._fold_mat.clear()
        self._std_stats.clear()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """
        Serialize the cache to ``path`` via :func:`joblib.dump`.

        The format includes a version sentinel; :meth:`load` will raise
        :class:`CacheVersionError` if the layout is incompatible.
        """
        state = {
            "__cache_version__": _CACHE_VERSION,
            "fold_mat_maxsize":  self.fold_mat_maxsize,
            "col_matrices":      self._col_matrices,
            "col_feat_names":    self._col_feat_names,
            "subset_cache":      self._subset_cache,
            "subset_names":      self._subset_names,
            "fold_mat":          dict(self._fold_mat),   # OrderedDict → dict
            "fold_idx":          self._fold_idx,
            "std_stats":         self._std_stats,
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
        cache._col_matrices   = state["col_matrices"]
        cache._col_feat_names = state["col_feat_names"]
        cache._subset_cache   = state["subset_cache"]
        cache._subset_names   = state["subset_names"]
        cache._fold_mat       = OrderedDict(state["fold_mat"])
        cache._fold_idx       = state["fold_idx"]
        cache._std_stats      = state["std_stats"]
        return cache
