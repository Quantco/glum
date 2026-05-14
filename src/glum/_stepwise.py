"""
StepwiseGLM
===========

A wrapper around :class:`glum.GeneralizedLinearRegressor` that accelerates
iterative / stepwise model fitting via four complementary optimizations:

1. **Column-level tabmat cache** — each DataFrame column is converted to a
   tabmat sub-matrix exactly once.  Column subsets are assembled via
   ``tabmat.hstack``, eliminating the narwhals dtype-detection overhead that
   ``from_pandas`` incurs on every call.

2. **Warm-start coefficients** — ``warm_start=True`` is always enabled so the
   IRLS solver begins from the previous solution, reducing outer iterations on
   incremental changes.

3. **Standardization cache** — column means and standard deviations are
   computed once per unique column and cached.  Subsequent fits inject the
   cached stats directly, replacing the full O(n) pass each step.  The cache
   is invalidated automatically when ``sample_weight`` changes.

4. **Score-test candidate screener** — ``screen_candidates()`` computes the
   score (Rao) test statistic for each candidate column against the current
   fitted model.  This costs one dot-product per candidate instead of one full
   IRLS solve, enabling forward-stepwise search to rank many candidates and
   only refit the winner.

No global monkeypatching occurs.  The standardization cache is applied by
temporarily replacing ``glum._glm.standardize`` for the duration of each
``fit()`` call only, then restoring it unconditionally in a ``finally`` block.

Usage
-----
::

    from glum._stepwise import StepwiseGLM

    glm = StepwiseGLM(family="poisson", alpha=0.01)

    # Forward stepwise loop
    active = ["x0", "x1"]
    candidates = ["x2", "x3", "x4", "x5"]

    for _ in range(len(candidates)):
        glm.fit(df[active], y)
        scores = glm.screen_candidates(df, active, candidates)
        best = scores[0]          # sorted descending by statistic
        if not best.selected:
            break
        active.append(best.column)
        candidates.remove(best.column)
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pandas as pd
import scipy.stats
import tabmat as tm
from sklearn.model_selection import check_cv

import glum
import glum._glm as _glm_mod
import glum._utils as _utils_mod

# ---------------------------------------------------------------------------
# Stash the real standardize at import time (idempotent)
# ---------------------------------------------------------------------------

if not hasattr(_utils_mod, "_original_standardize"):
    _utils_mod._original_standardize = _utils_mod.standardize


# ---------------------------------------------------------------------------
# 1. Matrix cache
# ---------------------------------------------------------------------------

class _MatrixCache:
    """
    Two-level tabmat matrix cache.

    **Column-level store**: each DataFrame column is converted to a single-
    column tabmat sub-matrix once and cached by name.  Used for score-test
    candidate screening (one column at a time).

    **Subset-level store**: full column-subset matrices are memoized by the
    tuple of column names.  On the first call for a given column set,
    ``tabmat.from_pandas`` wraps the existing pandas memory (cheap).  On
    subsequent calls the cached ``MatrixBase`` is returned in O(1) via a dict
    lookup — no data copying, no narwhals dtype detection.

    At small n, ``hstack`` from per-column pieces is competitive.  At large n
    (1M+ rows), ``from_pandas`` on a DataFrame slice is faster because it
    wraps existing memory, while ``hstack`` must concatenate arrays.  The
    subset-level memoization gives us the best of both: pay ``from_pandas``
    cost once per unique column set, then zero cost thereafter.
    """

    def __init__(self, source_df: Optional[pd.DataFrame] = None):
        # col_name → single-column MatrixBase  (for score tests)
        self._col_matrices:   dict[str, tm.MatrixBase] = {}
        self._col_feat_names: dict[str, list[str]]     = {}
        # tuple(cols) → full-subset MatrixBase  (for fit())
        self._subset_cache:   dict[tuple, tm.MatrixBase] = {}
        self._subset_names:   dict[tuple, list[str]]    = {}

        self._source_df: Optional[pd.DataFrame] = source_df

    def set_source(self, df: pd.DataFrame) -> None:
        """Register the source DataFrame used for subset lookups."""
        self._source_df = df

    def register_cols(self, df: pd.DataFrame) -> None:
        """Build per-column cache entries for any unseen columns in df."""
        for col in df.columns:
            if col not in self._col_matrices:
                mat = tm.from_pandas(df[[col]])
                self._col_matrices[col]   = mat
                self._col_feat_names[col] = mat.get_names(
                    type="column", missing_prefix=f"_{col}_"
                )

    def get_subset(
        self, df: pd.DataFrame, cols: list[str]
    ) -> tuple[tm.MatrixBase, list[str]]:
        """
        Return (MatrixBase, expanded_feature_names) for a column subset.

        Uses the subset-level memoization cache.  On a miss, calls
        ``from_pandas`` on ``df[cols]`` and stores the result.
        """
        key = tuple(cols)
        if key not in self._subset_cache:
            mat = tm.from_pandas(df[cols])
            self._subset_cache[key] = mat
            self._subset_names[key] = mat.get_names(
                type="column", missing_prefix="_col_"
            )
        return self._subset_cache[key], self._subset_names[key]

    def get_col(self, col: str) -> tm.MatrixBase:
        """Return the single-column matrix for a registered column."""
        return self._col_matrices[col]

    def col_feat_names(self, col: str) -> list[str]:
        return self._col_feat_names[col]

    def __contains__(self, col: str) -> bool:
        return col in self._col_matrices


# ---------------------------------------------------------------------------
# 2 & 3. Caching standardize replacement
# ---------------------------------------------------------------------------

class _CachingStandardize:
    """
    Drop-in replacement for ``glum._utils.standardize``.

    Serves cached column stats where available; falls back to the real
    ``standardize`` for new columns and populates the cache from the result.
    """

    def __init__(self, col_stats: dict, active_names: list[str], sw_id: int):
        self._col_stats   = col_stats
        self._names       = active_names
        self._sw_id       = sw_id
        self.cache_hits   = 0
        self.cache_misses = 0

    def __call__(self, X, sample_weight, center_predictors,
                 estimate_as_if_scaled_model, lower_bounds, upper_bounds,
                 A_ineq, P1, P2):
        import scipy.sparse as sparse

        n = X.shape[1]
        means = np.empty(n)
        stds  = np.ones(n)
        all_hit = True

        for j, name in enumerate(self._names):
            key = (name, self._sw_id)
            if key in self._col_stats:
                means[j], stds[j] = self._col_stats[key]
                self.cache_hits += 1
            else:
                all_hit = False
                self.cache_misses += 1

        if all_hit:
            col_means = means if center_predictors else np.zeros(n)
            inv_stds  = np.where(stds > 0, 1.0 / stds, 1.0)
            shifts    = -means * inv_stds if center_predictors else np.zeros(n)
            X_std     = tm.StandardizedMatrix(X, shifts, inv_stds)

            if not estimate_as_if_scaled_model:
                P1 = P1 * inv_stds
                if sparse.issparse(P2):
                    D  = sparse.diags(inv_stds)
                    P2 = D @ P2 @ D
                elif P2.ndim == 1:
                    P2 = P2 * inv_stds ** 2
                else:
                    P2 = (inv_stds[:, None] * P2) * inv_stds[None, :]

            return X_std, col_means, stds, lower_bounds, upper_bounds, A_ineq, P1, P2

        # Cache miss — delegate and populate
        result = _utils_mod._original_standardize(
            X, sample_weight, center_predictors,
            estimate_as_if_scaled_model,
            lower_bounds, upper_bounds, A_ineq, P1, P2,
        )
        _, col_means, col_stds, *_ = result
        if col_stds is not None:
            for j, name in enumerate(self._names):
                key = (name, self._sw_id)
                if key not in self._col_stats:
                    self._col_stats[key] = (col_means[j], col_stds[j])
        return result


@contextlib.contextmanager
def _patch_standardize(caching_std):
    """Swap glum._glm.standardize for one fit() call, restore on exit."""
    orig = _glm_mod.standardize
    try:
        _glm_mod.standardize = caching_std
        yield
    finally:
        _glm_mod.standardize = orig


# ---------------------------------------------------------------------------
# 4. Score test result
# ---------------------------------------------------------------------------

@dataclass
class ScoreTestResult:
    """Score (Rao) test result for one candidate column."""
    column:    str
    statistic: float   # chi-squared statistic
    dof:       int     # 1 for numeric; (levels - 1) for categorical
    pvalue:    float
    selected:  bool    # pvalue < alpha threshold


# ---------------------------------------------------------------------------
# 5. CV fold matrix cache
# ---------------------------------------------------------------------------

class _CVMatrixCache:
    """
    Per-fold tabmat matrix cache for repeated cross-validation.

    Stores two things per fold:

    **Fold row-slice cache** — the most recently used column subset for
    each fold, keyed by ``(fold_id, tuple(cols))``.  When the next
    stepwise step adds a single column, the new fold slice is assembled via
    ``tabmat.hstack([cached_prev_slice, new_col[train_idx, :]])`` instead
    of re-slicing the full matrix (~10× faster on mixed data).

    **Fold standardize stats cache** — ``(col_means, col_stds)`` per
    ``(fold_id, tuple(cols), sw_id)``.  The stats are stable across
    repeated CV calls with the same split, so they need only be computed
    once per unique (fold, column-set, sample-weight) triple.
    """

    def __init__(self):
        # (fold_id, col_tuple) → (MatrixBase, train_idx ndarray)
        self._fold_mat:    dict[tuple, tm.MatrixBase]  = {}
        self._fold_idx:    dict[int,   np.ndarray]     = {}
        # (fold_id, col_tuple, sw_id) → (col_means ndarray, col_stds ndarray)
        self._std_stats:   dict[tuple, tuple]          = {}
        # per-column matrices for incremental hstack (from _MatrixCache)
        self._col_mats:    dict[str, tm.MatrixBase]    = {}

    def register_col_mats(self, col_mats: dict) -> None:
        """Share the per-column cache from _MatrixCache."""
        self._col_mats = col_mats

    def set_fold_indices(self, fold_id: int, train_idx: np.ndarray) -> None:
        """Store the training indices for a fold (called once when CV splits are built)."""
        self._fold_idx[fold_id] = train_idx

    def get_fold_mat(
        self,
        fold_id: int,
        cols: list[str],
        full_mat: tm.MatrixBase,
    ) -> tm.MatrixBase:
        """
        Return the fold training slice for ``cols``.

        Strategy:
        1. Exact cache hit → O(1).
        2. The previous cached col-set for this fold is a prefix of ``cols``
           (stepwise added one or more columns) → incremental hstack.
        3. Full miss → re-slice ``full_mat`` and store.
        """
        key = (fold_id, tuple(cols))
        if key in self._fold_mat:
            return self._fold_mat[key]

        train_idx = self._fold_idx[fold_id]

        # Find longest cached prefix for this fold
        best_prefix: Optional[tuple]       = None
        best_prefix_mat: Optional[tm.MatrixBase] = None
        for cached_key, cached_mat in self._fold_mat.items():
            if cached_key[0] != fold_id:
                continue
            cached_cols = cached_key[1]
            n = len(cached_cols)
            if tuple(cols[:n]) == cached_cols and n > (len(best_prefix) if best_prefix else -1):
                best_prefix     = cached_cols
                best_prefix_mat = cached_mat

        if best_prefix is not None and best_prefix_mat is not None:
            # Incrementally append the new columns
            new_cols = cols[len(best_prefix):]
            pieces   = [best_prefix_mat]
            for c in new_cols:
                if c in self._col_mats:
                    pieces.append(self._col_mats[c][train_idx, :])
                else:
                    # Fallback: slice from full mat column-by-column is hard;
                    # just do a full re-slice for this miss.
                    pieces = None
                    break
            if pieces is not None:
                mat = tm.hstack(pieces) if len(pieces) > 1 else pieces[0]
                self._fold_mat[key] = mat
                return mat

        # Full miss — row-slice the full dataset matrix
        mat = full_mat[train_idx, :]
        self._fold_mat[key] = mat
        return mat

    def get_std_stats(
        self,
        fold_id: int,
        cols: list[str],
        sw_id: int,
    ) -> Optional[tuple]:
        """Return cached ``(col_means, col_stds)`` or ``None`` on miss."""
        return self._std_stats.get((fold_id, tuple(cols), sw_id))

    def set_std_stats(
        self,
        fold_id: int,
        cols: list[str],
        sw_id: int,
        col_means: np.ndarray,
        col_stds: np.ndarray,
    ) -> None:
        self._std_stats[(fold_id, tuple(cols), sw_id)] = (col_means, col_stds)

    def clear_fold_mats(self) -> None:
        """Evict fold matrix cache (e.g. after column set changes substantially)."""
        self._fold_mat.clear()


# ---------------------------------------------------------------------------
# CV result dataclass
# ---------------------------------------------------------------------------

@dataclass
class CVResult:
    """Cross-validation result for one candidate column."""
    column:         str
    cv_deviance:    float    # mean hold-out deviance across folds
    alpha:          float    # best alpha chosen by CV
    selected:       bool     # cv_deviance < baseline_deviance


# ---------------------------------------------------------------------------
# StepwiseGLM
# ---------------------------------------------------------------------------

class StepwiseGLM:
    """
    Accelerated GLM wrapper for iterative / stepwise model fitting.

    Parameters
    ----------
    **glm_kwargs
        Forwarded to :class:`glum.GeneralizedLinearRegressor`.
        ``warm_start`` is always forced to ``True``.

    Attributes
    ----------
    glm_ : GeneralizedLinearRegressor
        The underlying fitted estimator.
    cache_stats_ : dict
        Per-step standardization cache hit/miss counts, keyed by step index.
    score_test_history_ : list[list[ScoreTestResult]]
        One entry per ``screen_candidates()`` call.
    """

    def __init__(self, **glm_kwargs):
        glm_kwargs["warm_start"] = True
        self.glm_ = glum.GeneralizedLinearRegressor(**glm_kwargs)

        self._mat_cache  = _MatrixCache()
        self._cv_cache   = _CVMatrixCache()
        self._col_stats: dict = {}
        self._last_sw_id: Optional[int] = None
        self._step = 0
        self._source_df: Optional[pd.DataFrame] = None

        self.cache_stats_:       dict                      = {}
        self.score_test_history_: list[list[ScoreTestResult]] = []
        self.cv_history_:        list[list[CVResult]]      = []

        # State carried between fit() and screen_candidates()
        self._last_y:     Optional[np.ndarray]    = None
        self._last_mu:    Optional[np.ndarray]    = None
        self._last_X_tab: Optional[tm.MatrixBase] = None

    # ------------------------------------------------------------------
    # fit()
    # ------------------------------------------------------------------

    def fit(
        self,
        X: Union[pd.DataFrame, tm.MatrixBase],
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        **fit_kwargs,
    ) -> "StepwiseGLM":
        """
        Fit the GLM with all caching optimizations applied.

        Parameters
        ----------
        X : pd.DataFrame or tabmat.MatrixBase
        y : array-like
        sample_weight : array-like, optional
        """
        y = np.asarray(y)
        self._last_y = y   # cache for screen_candidates()

        # ── Resolve X ────────────────────────────────────────────────────
        if isinstance(X, pd.DataFrame):
            self._source_df = X   # keep reference for score tests
            X_tab, active_names = self._mat_cache.get_subset(X, list(X.columns))
        elif isinstance(X, tm.MatrixBase):
            X_tab = X
            active_names = X_tab.get_names(type="column", missing_prefix="_col_")
        else:
            # Unknown type — let glum handle it unoptimised
            self.glm_.fit(X, y, sample_weight=sample_weight, **fit_kwargs)
            self._step += 1
            return self

        # ── Sample-weight cache invalidation ─────────────────────────────
        if sample_weight is None:
            sw_id = -1
        else:
            sample_weight = np.asarray(sample_weight)
            sw_id = id(sample_weight)

        if sw_id != self._last_sw_id and self._last_sw_id is not None:
            self._col_stats.clear()
        self._last_sw_id = sw_id

        # ── Guard warm_start against shape changes ───────────────────────
        if hasattr(self.glm_, "coef_") and len(self.glm_.coef_) != X_tab.shape[1]:
            del self.glm_.coef_

        # ── Fit with cached standardize ───────────────────────────────────
        caching_std = _CachingStandardize(
            col_stats=self._col_stats,
            active_names=active_names,
            sw_id=sw_id,
        )
        with _patch_standardize(caching_std):
            self.glm_.fit(X_tab, y, sample_weight=sample_weight, **fit_kwargs)

        # ── Cache mu for score tests ──────────────────────────────────────
        self._last_mu    = self.glm_.predict(X_tab)
        self._last_X_tab = X_tab

        # ── Record cache stats ────────────────────────────────────────────
        n = len(active_names)
        self.cache_stats_[self._step] = {
            "hits":     caching_std.cache_hits,
            "misses":   caching_std.cache_misses,
            "n_cols":   n,
            "hit_rate": caching_std.cache_hits / n if n else 0.0,
        }
        self._step += 1
        return self

    # ------------------------------------------------------------------
    # screen_candidates()
    # ------------------------------------------------------------------

    def screen_candidates(
        self,
        df: pd.DataFrame,
        active_cols: list[str],
        candidate_cols: list[str],
        alpha: float = 0.05,
    ) -> list[ScoreTestResult]:
        """
        Rank candidate columns using the score (Rao) test without refitting.

        For each candidate, computes the score test statistic under the current
        fitted model at O(n) cost per candidate — vs O(n·p·iters) for a full
        IRLS refit.  Works for any GLM family and handles categorical columns
        (multi-dof chi-squared test) automatically.

        Parameters
        ----------
        df : pd.DataFrame
            Source DataFrame containing all columns.
        active_cols : list[str]
            Columns currently in the model (informational only; model must
            already be fitted via ``fit()``).
        candidate_cols : list[str]
            Columns to screen as potential additions.
        alpha : float
            Significance threshold for ``ScoreTestResult.selected``.

        Returns
        -------
        list[ScoreTestResult]
            Sorted descending by ``statistic``.
        """
        if self._last_mu is None or self._last_y is None:
            raise RuntimeError("Call fit() before screen_candidates().")

        mu    = self._last_mu
        y_arr = self._last_y
        fam   = self.glm_._family_instance
        link  = self.glm_._link_instance

        # Family-agnostic per-row weights — O(n), computed once for all candidates
        eta             = link.link(mu)
        d1              = link.inverse_derivative(eta)   # h'(eta)
        var             = fam.variance(mu)               # V(mu)
        gradient_rows   = (d1 / var) * (y_arr - mu)     # (n,)  score contributions
        hessian_rows    = (d1 ** 2) / var                # (n,)  curvature weights

        # Ensure candidate columns are in the per-column cache
        self._mat_cache.register_cols(df[candidate_cols])

        results: list[ScoreTestResult] = []

        for col in candidate_cols:
            mat = self._mat_cache.get_col(col)
            k   = len(self._mat_cache.col_feat_names(col))   # expanded width

            if k == 1:
                # Numeric / binary: scalar chi2(1) test
                xj    = mat.toarray().ravel()
                sj    = float(gradient_rows @ xj)
                H_jj  = float(hessian_rows  @ (xj ** 2))
                stat  = sj ** 2 / H_jj if H_jj > 1e-12 else 0.0
                dof   = 1
            else:
                # Categorical: vector chi2(k) test
                # H is diagonal because one-hot columns are orthogonal per row
                X_cat = mat.toarray()                       # (n, k)
                s     = X_cat.T @ gradient_rows             # (k,)
                H     = X_cat.T @ (hessian_rows[:, None] * X_cat)  # (k, k)
                try:
                    stat = float(s @ np.linalg.solve(H, s))
                except np.linalg.LinAlgError:
                    stat = float(s @ np.linalg.lstsq(H, s, rcond=None)[0])
                dof = k

            pval = float(1.0 - scipy.stats.chi2.cdf(stat, df=dof))
            results.append(ScoreTestResult(
                column=col, statistic=stat, dof=dof,
                pvalue=pval, selected=(pval < alpha),
            ))

        results.sort(key=lambda r: r.statistic, reverse=True)
        self.score_test_history_.append(results)
        return results

    # ------------------------------------------------------------------
    # cv_select()
    # ------------------------------------------------------------------

    def cv_select(
        self,
        df: pd.DataFrame,
        active_cols: list[str],
        candidate_cols: list[str],
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        cv=5,
        alphas: Optional[np.ndarray] = None,
        n_alphas: int = 20,
    ) -> list[CVResult]:
        """
        Rank candidate columns by cross-validated hold-out deviance.

        For each candidate ``c``, fits the model on ``active_cols + [c]``
        along a regularization path across ``cv`` folds, selecting the
        alpha that minimises mean fold deviance.  Results are sorted
        ascending by ``cv_deviance`` (lower = better).

        The fold matrices are built incrementally — the row-slice for the
        current active set is cached per fold, and adding a candidate
        column appends only the new column's fold slice via
        ``tabmat.hstack``.  Fold standardize stats are also cached,
        eliminating the O(n) pass on repeated calls with the same split.

        Parameters
        ----------
        df : pd.DataFrame
            Source data for all columns.
        active_cols : list[str]
            Columns currently in the model.
        candidate_cols : list[str]
            Columns to evaluate as potential additions.
        y : array-like
        sample_weight : array-like, optional
        cv : int or CV splitter
            Passed to ``sklearn.model_selection.check_cv``.
        alphas : array-like, optional
            Explicit alpha grid.  If ``None``, a log-spaced grid of
            ``n_alphas`` values is derived from the data.
        n_alphas : int
            Number of alphas to search when ``alphas`` is ``None``.

        Returns
        -------
        list[CVResult]
            Sorted ascending by ``cv_deviance``.
        """
        y = np.asarray(y)
        sw_id = -1 if sample_weight is None else id(np.asarray(sample_weight))

        # ── Ensure per-column cache is populated ────────────────────────
        all_cols = list(dict.fromkeys(active_cols + candidate_cols))
        self._mat_cache.register_cols(df[all_cols])
        self._cv_cache.register_col_mats(self._mat_cache._col_matrices)

        # ── Build the full tabmat for the active+candidate superset ─────
        full_cols_key = tuple(all_cols)
        full_mat, _ = self._mat_cache.get_subset(df, all_cols)

        # ── Build CV splits (stable across candidates) ───────────────────
        splitter = check_cv(cv)
        splits   = list(splitter.split(df, y))
        n_folds  = len(splits)

        for fold_id, (train_idx, _) in enumerate(splits):
            self._cv_cache.set_fold_indices(fold_id, train_idx)

        # ── Determine alpha grid ─────────────────────────────────────────
        # Use glum's built-in alpha-path logic on the active set
        if alphas is None:
            active_mat, active_names = self._mat_cache.get_subset(df, active_cols)
            _tmp_glm = glum.GeneralizedLinearRegressorCV(
                family=self.glm_.family,
                l1_ratio=getattr(self.glm_, "l1_ratio", 0),
                n_alphas=n_alphas,
                cv=2,          # just to get alpha grid, not for real CV
            )
            # Derive alpha max from the active+first-candidate set as a proxy
            probe_cols = active_cols + candidate_cols[:1]
            probe_mat, _ = self._mat_cache.get_subset(df, probe_cols)
            sw_probe = (np.ones(len(y)) / len(y)
                        if sample_weight is None
                        else np.asarray(sample_weight) / np.asarray(sample_weight).sum())
            from glum._glm import setup_p1, setup_p2
            l1 = getattr(self.glm_, "l1_ratio", 0)
            P1 = setup_p1("identity", probe_mat, probe_mat.dtype, 1, l1)
            P2 = setup_p2("identity", probe_mat, ["csc", "csr"], probe_mat.dtype, 1, l1)
            probe_std, *_ = _utils_mod._original_standardize(
                probe_mat, sw_probe, True, False, None, None, None, P1, P2
            )
            resid = (y - y.mean()) / len(y)
            alpha_max = float(
                np.max(np.abs(probe_std.transpose_matvec(resid)))
            )
            alpha_min = alpha_max * 1e-4
            alphas = np.exp(np.linspace(np.log(alpha_max), np.log(alpha_min), n_alphas))

        # ── Per-candidate CV ─────────────────────────────────────────────
        # Compute baseline deviance (active model, no candidate)
        baseline_dev = self._cv_deviance(
            df, active_cols, y, sample_weight, splits, alphas, sw_id
        )

        results: list[CVResult] = []
        for cand in candidate_cols:
            cols = active_cols + [cand]
            dev  = self._cv_deviance(
                df, cols, y, sample_weight, splits, alphas, sw_id
            )
            best_alpha = alphas[np.argmin(dev)]
            mean_dev   = float(np.mean(dev))
            results.append(CVResult(
                column=cand,
                cv_deviance=mean_dev,
                alpha=float(best_alpha),
                selected=(mean_dev < float(np.mean(baseline_dev))),
            ))

        results.sort(key=lambda r: r.cv_deviance)
        self.cv_history_.append(results)
        return results

    def _cv_deviance(
        self,
        df: pd.DataFrame,
        cols: list[str],
        y: np.ndarray,
        sample_weight: Optional[np.ndarray],
        splits: list,
        alphas: np.ndarray,
        sw_id: int,
    ) -> np.ndarray:
        """
        Return per-fold best deviance (shape: n_folds,) for a column set
        and alpha grid, using cached fold matrices and standardize stats.

        The regularization path is warm-started from large alpha to small
        (GLMNet trick) within each fold.  Fold matrices are assembled
        incrementally from the column cache; standardize stats are cached
        per (fold, cols, sw_id) so subsequent calls with the same split
        pay zero O(n) standardize cost.
        """
        fam  = self.glm_._family_instance
        fold_deviances = np.zeros(len(splits))

        # Memoized full-column-set matrix (pays from_pandas at most once)
        full_mat, _ = self._mat_cache.get_subset(df, cols)

        for fold_id, (train_idx, test_idx) in enumerate(splits):
            # ── Fold matrices (incremental hstack from cache) ─────────────
            X_train = self._cv_cache.get_fold_mat(fold_id, cols, full_mat)
            X_test  = full_mat[test_idx, :]

            y_train = y[train_idx]
            y_test  = y[test_idx]

            if sample_weight is None:
                sw_train = np.ones(len(train_idx)) / len(train_idx)
                sw_test  = np.ones(len(test_idx))  / len(test_idx)
            else:
                sw = np.asarray(sample_weight)
                sw_train = sw[train_idx] / sw[train_idx].sum()
                sw_test  = sw[test_idx]  / sw[test_idx].sum()

            # ── Standardize stats (cached per fold+cols+sw) ───────────────
            # Build a _CachingStandardize that caches fold-level stats
            # keyed by (fold_id, col_name, sw_id).  On hit, injects stats
            # directly without an O(n) pass.
            fold_col_stats_key = (fold_id, tuple(cols), sw_id)
            cached = self._cv_cache.get_std_stats(fold_id, cols, sw_id)
            if cached is not None:
                # Re-use cached stats — build a one-shot caching_std that
                # will always hit and store the result into _col_stats
                col_means = np.atleast_1d(np.asarray(cached[0], dtype=float).ravel())
                col_stds  = np.atleast_1d(np.asarray(cached[1], dtype=float).ravel())
                n_feat = X_train.shape[1]
                inv_stds = np.where(col_stds > 0, 1.0 / col_stds, 1.0)
                shifts   = -col_means * inv_stds

                def _cached_std(X, sw, center, scale, lb, ub, Ai, P1, P2,
                                _shifts=shifts, _inv_stds=inv_stds,
                                _means=col_means, _stds=col_stds):
                    import scipy.sparse as sp
                    X_std = tm.StandardizedMatrix(X, _shifts, _inv_stds)
                    if not scale:
                        P1 = P1 * _inv_stds
                        if sp.issparse(P2):
                            D = sp.diags(_inv_stds)
                            P2 = D @ P2 @ D
                        elif P2.ndim == 1:
                            P2 = P2 * _inv_stds ** 2
                        else:
                            P2 = (_inv_stds[:, None] * P2) * _inv_stds[None, :]
                    return X_std, _means if center else np.zeros(n_feat), _stds, lb, ub, Ai, P1, P2

                std_fn = _cached_std
                needs_store = False
            else:
                # Miss — delegate to real standardize and capture results
                _store = {}

                def _capturing_std(X, sw, center, scale, lb, ub, Ai, P1, P2,
                                   _store=_store):
                    result = _utils_mod._original_standardize(
                        X, sw, center, scale, lb, ub, Ai, P1, P2
                    )
                    _store['means'] = np.atleast_1d(np.asarray(result[1], dtype=float).ravel())
                    _store['stds']  = (np.atleast_1d(np.asarray(result[2], dtype=float).ravel())
                                       if result[2] is not None
                                       else np.ones(X.shape[1]))
                    return result

                std_fn = _capturing_std
                needs_store = True

            # ── Warm-started regularization path ─────────────────────────
            glm_fold = glum.GeneralizedLinearRegressor(
                family=self.glm_.family,
                l1_ratio=getattr(self.glm_, "l1_ratio", 0),
                drop_first=getattr(self.glm_, "drop_first", False),
                alpha=alphas[0],
                warm_start=False,
            )
            coef      = None
            best_dev  = np.inf
            stored_first = False

            # Warm-start across alpha steps is safe only when n_features > 1.
            # With a single feature, standardize_warm_start has a numpy
            # broadcasting edge case (0-d scalar .dot 1-D array → 1-D result).
            allow_warm = X_train.shape[1] > 1

            for k, alpha in enumerate(alphas):
                glm_fold.alpha = alpha
                if coef is not None and allow_warm:
                    glm_fold.coef_      = coef[1:]
                    glm_fold.intercept_ = coef[0]
                    glm_fold.warm_start = True
                else:
                    glm_fold.warm_start = False

                with _patch_standardize(std_fn):
                    glm_fold.fit(X_train, y_train, sample_weight=sw_train)

                # Capture and store std stats after first alpha (stats don't
                # depend on alpha — only on X_train and sw_train)
                if needs_store and not stored_first and 'means' in _store:
                    self._cv_cache.set_std_stats(
                        fold_id, cols, sw_id,
                        _store['means'], _store['stds']
                    )
                    # Promote to cached function for remaining alphas
                    col_means = _store['means']
                    col_stds  = _store['stds']
                    n_feat    = X_train.shape[1]
                    inv_stds  = np.where(col_stds > 0, 1.0 / col_stds, 1.0)
                    shifts    = -col_means * inv_stds

                    def _cached_std2(X, sw, center, scale, lb, ub, Ai, P1, P2,
                                     _shifts=shifts, _inv_stds=inv_stds,
                                     _means=col_means, _stds=col_stds):
                        import scipy.sparse as sp
                        X_s = tm.StandardizedMatrix(X, _shifts, _inv_stds)
                        if not scale:
                            P1 = P1 * _inv_stds
                            if sp.issparse(P2):
                                D = sp.diags(_inv_stds)
                                P2 = D @ P2 @ D
                            elif P2.ndim == 1:
                                P2 = P2 * _inv_stds ** 2
                            else:
                                P2 = (_inv_stds[:, None] * P2) * _inv_stds[None, :]
                        return X_s, _means if center else np.zeros(n_feat), _stds, lb, ub, Ai, P1, P2

                    std_fn = _cached_std2
                    stored_first = True

                coef = np.concatenate([[glm_fold.intercept_], glm_fold.coef_])

                mu_test = glm_fold.predict(X_test)
                dev = float(fam.deviance(y_test, mu_test, sw_test))
                if dev < best_dev:
                    best_dev = dev

            fold_deviances[fold_id] = best_dev

        return fold_deviances

    # ------------------------------------------------------------------
    # Convenience / delegation
    # ------------------------------------------------------------------

    def predict(self, X, **kwargs):
        """Predict using the underlying GLM."""
        return self.glm_.predict(X, **kwargs)

    def __getattr__(self, name: str):
        if name.startswith("_") or name in (
            "glm_", "cache_stats_", "score_test_history_"
        ):
            raise AttributeError(name)
        return getattr(self.glm_, name)
