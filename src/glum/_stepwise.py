"""
StepwiseGLM
===========

A wrapper around :class:`glum.GeneralizedLinearRegressor` that accelerates
iterative / stepwise model fitting via four complementary optimizations:

1. **Tabmat matrix cache** — built on :class:`glum.TabmatCache`.  Each
   DataFrame column is converted to a tabmat sub-matrix exactly once.
   Column subsets and fold row-slices are memoized.  See
   :mod:`glum._tabmat_cache`.

2. **Warm-start coefficients** — ``warm_start=True`` is always enabled so
   the IRLS solver begins from the previous solution, reducing outer
   iterations on incremental changes.

3. **Standardization cache** — column means and standard deviations are
   computed once per unique column and cached.  Subsequent fits inject the
   cached stats directly, replacing the full O(n) pass each step.  The
   cache is invalidated automatically when ``sample_weight`` changes.

4. **Score-test candidate screener** — :meth:`StepwiseGLM.screen_candidates`
   computes the score (Rao) test statistic for each candidate column
   against the current fitted model.  This costs one dot-product per
   candidate instead of one full IRLS solve, enabling forward-stepwise
   search to rank many candidates and only refit the winner.

5. **Cross-validated selection** — :meth:`StepwiseGLM.cv_select` evaluates
   candidates by hold-out deviance using cached per-fold matrices and
   standardize stats.

No global monkeypatching occurs.  The standardization cache is applied by
temporarily replacing ``glum._glm.standardize`` for the duration of each
``fit()`` call only, then restoring it unconditionally in a ``finally``
block.  The patch is also serialized across threads via a module-level
:class:`threading.Lock`.

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
import threading
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
from glum._tabmat_cache import TabmatCache

# ---------------------------------------------------------------------------
# Stash the real standardize at import time (idempotent)
# ---------------------------------------------------------------------------

if not hasattr(_utils_mod, "_original_standardize"):
    _utils_mod._original_standardize = _utils_mod.standardize

# ---------------------------------------------------------------------------
# Thread-safety lock for the module-level standardize patch
# ---------------------------------------------------------------------------
# `_patch_standardize` mutates `glum._glm.standardize` (a module global).
# Without serialization, concurrent fit() calls would clobber each other's
# patch.  This lock holds for the duration of one IRLS solve — concurrent
# StepwiseGLM users will fit sequentially, but each fit retains its full
# speedup vs. the unpatched code path.
_standardize_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Caching standardize replacement
# ---------------------------------------------------------------------------

class _CachingStandardize:
    """
    Drop-in replacement for ``glum._utils.standardize``.

    Serves cached column stats where available; falls back to the real
    ``standardize`` for new columns and populates the cache from the result.

    Returns ``col_means`` and ``col_stds`` as **1-D numpy arrays** in both
    the cache-hit and cache-miss paths.  This 1-D contract is required by
    ``glum._utils.standardize_warm_start`` — see its implementation for the
    edge case where 0-D scalars dotted with 1-D arrays produce shapes that
    cannot be assigned back to ``coef[0]``.
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
        means = np.empty(n, dtype=float)
        stds  = np.ones(n,  dtype=float)
        all_hit = True

        for j, name in enumerate(self._names):
            key = (name, self._sw_id)
            if key in self._col_stats:
                means[j], stds[j] = self._col_stats[key]
                self.cache_hits += 1
            else:
                all_hit = False
                self.cache_misses += 1

        # Enforce 1-D contract for downstream standardize_warm_start.
        means = np.atleast_1d(means.ravel())
        stds  = np.atleast_1d(stds.ravel())

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
        X_std, col_means, col_stds, *rest = result
        # Enforce 1-D contract on the returned values too.
        col_means = np.atleast_1d(np.asarray(col_means, dtype=float).ravel())
        if col_stds is not None:
            col_stds = np.atleast_1d(np.asarray(col_stds, dtype=float).ravel())
            for j, name in enumerate(self._names):
                key = (name, self._sw_id)
                if key not in self._col_stats:
                    self._col_stats[key] = (float(col_means[j]), float(col_stds[j]))
        return (X_std, col_means, col_stds, *rest)


@contextlib.contextmanager
def _patch_standardize(caching_std):
    """
    Swap ``glum._glm.standardize`` for one fit() call, restore on exit.

    Acquires :data:`_standardize_lock` for the duration of the patch so
    concurrent ``StepwiseGLM.fit()`` calls from different threads serialize
    cleanly instead of clobbering each other's patched function.
    """
    with _standardize_lock:
        orig = _glm_mod.standardize
        try:
            _glm_mod.standardize = caching_std
            yield
        finally:
            _glm_mod.standardize = orig


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ScoreTestResult:
    """Score (Rao) test result for one candidate column."""
    column:    str
    statistic: float   # chi-squared statistic
    dof:       int     # 1 for numeric; (levels - 1) for categorical
    pvalue:    float
    selected:  bool    # pvalue < alpha threshold


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
    cache : TabmatCache, optional
        Pre-existing matrix cache to share across multiple ``StepwiseGLM``
        instances or pre-warm via ``cache.register_cols(df)``.  If ``None``,
        a fresh ``TabmatCache`` is created on construction.
    cv_cache_maxsize : int, default 256
        Maximum number of cached fold row-slices when a fresh cache is
        created.  Ignored when ``cache`` is supplied.
    **glm_kwargs
        Forwarded to :class:`glum.GeneralizedLinearRegressor`.
        ``warm_start`` is always forced to ``True``.

    Attributes
    ----------
    glm_ : GeneralizedLinearRegressor
        The underlying fitted estimator.
    cache : TabmatCache
        Shared matrix cache; safe to use with vanilla GLMs and across
        multiple ``StepwiseGLM`` instances.
    cache_stats_ : dict
        Per-step standardization cache hit/miss counts, keyed by step index.
    score_test_history_ : list[list[ScoreTestResult]]
        One entry per ``screen_candidates()`` call.
    cv_history_ : list[list[CVResult]]
        One entry per ``cv_select()`` call.
    """

    def __init__(
        self,
        cache: Optional[TabmatCache] = None,
        cv_cache_maxsize: int = 256,
        **glm_kwargs,
    ):
        glm_kwargs["warm_start"] = True
        self.glm_ = glum.GeneralizedLinearRegressor(**glm_kwargs)

        self.cache = cache if cache is not None else TabmatCache(
            fold_mat_maxsize=cv_cache_maxsize,
        )

        self._col_stats: dict = {}
        self._last_sw_id: Optional[int] = None
        self._step = 0
        self._source_df: Optional[pd.DataFrame] = None

        self.cache_stats_:        dict                       = {}
        self.score_test_history_: list[list[ScoreTestResult]] = []
        self.cv_history_:         list[list[CVResult]]       = []

        # State carried between fit() and screen_candidates()
        self._last_y:      Optional[np.ndarray]    = None
        self._last_mu:     Optional[np.ndarray]    = None
        self._last_offset: Optional[np.ndarray]    = None
        self._last_X_tab:  Optional[tm.MatrixBase] = None

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
        **fit_kwargs
            Forwarded to ``GeneralizedLinearRegressor.fit``.  An ``offset``
            kwarg is captured and reused by ``screen_candidates()`` and
            ``cv_select()`` so the cached fitted mu stays consistent.
        """
        y = np.asarray(y)
        self._last_y = y

        # Capture and normalize offset (re-used by score test + CV)
        offset = fit_kwargs.get("offset")
        self._last_offset = np.asarray(offset) if offset is not None else None

        # ── Resolve X ────────────────────────────────────────────────────
        if isinstance(X, pd.DataFrame):
            self._source_df = X
            X_tab, active_names = self.cache.get_subset(X, list(X.columns))
        elif isinstance(X, tm.MatrixBase):
            X_tab = X
            active_names = X_tab.get_names(type="column", missing_prefix="_col_")
        else:
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

        # ── Cache mu (with offset!) for score tests ──────────────────────
        if self._last_offset is not None:
            self._last_mu = self.glm_.predict(X_tab, offset=self._last_offset)
        else:
            self._last_mu = self.glm_.predict(X_tab)
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

        For each candidate, computes the score test statistic under the
        current fitted model at O(n) cost per candidate — vs O(n·p·iters)
        for a full IRLS refit.  Works for any GLM family and handles
        categorical columns (multi-dof chi-squared test) automatically.
        Uses the cached ``mu`` from the last :meth:`fit` call, which
        correctly includes any offset that was passed to ``fit()``.

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
        self.cache.register_cols(df[candidate_cols])

        results: list[ScoreTestResult] = []

        for col in candidate_cols:
            mat = self.cache.get_col(col)
            k   = len(self.cache.col_feat_names(col))

            if k == 1:
                # Numeric / binary: scalar chi2(1) test
                xj    = mat.toarray().ravel()
                sj    = float(gradient_rows @ xj)
                H_jj  = float(hessian_rows  @ (xj ** 2))
                stat  = sj ** 2 / H_jj if H_jj > 1e-12 else 0.0
                dof   = 1
            else:
                # Categorical: vector chi2(k) test
                X_cat = mat.toarray()
                s     = X_cat.T @ gradient_rows
                H     = X_cat.T @ (hessian_rows[:, None] * X_cat)
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
        offset: Optional[np.ndarray] = None,
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

        Parameters
        ----------
        df, active_cols, candidate_cols, y, sample_weight, cv,
        alphas, n_alphas
            See class docstring.
        offset : array-like, optional
            Per-row offset (added to the linear predictor).  If ``None``,
            falls back to whatever offset was used in the most recent
            ``fit()`` call.

        Returns
        -------
        list[CVResult]
            Sorted ascending by ``cv_deviance``.
        """
        y = np.asarray(y)
        sw_id = -1 if sample_weight is None else id(np.asarray(sample_weight))

        # Default offset to whatever was used in fit()
        if offset is None:
            offset = self._last_offset
        if offset is not None:
            offset = np.asarray(offset)

        # ── Ensure per-column cache is populated ────────────────────────
        all_cols = list(dict.fromkeys(active_cols + candidate_cols))
        self.cache.register_cols(df[all_cols])
        full_mat, _ = self.cache.get_subset(df, all_cols)

        # ── Build CV splits (stable across candidates) ───────────────────
        splitter = check_cv(cv)
        splits   = list(splitter.split(df, y))

        for fold_id, (train_idx, _) in enumerate(splits):
            self.cache.set_fold_indices(fold_id, train_idx)

        # ── Determine alpha grid ─────────────────────────────────────────
        if alphas is None:
            probe_cols = active_cols + candidate_cols[:1]
            probe_mat, _ = self.cache.get_subset(df, probe_cols)
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
        baseline_dev = self._cv_deviance(
            df, active_cols, y, sample_weight, offset, splits, alphas, sw_id
        )

        results: list[CVResult] = []
        for cand in candidate_cols:
            cols = active_cols + [cand]
            dev  = self._cv_deviance(
                df, cols, y, sample_weight, offset, splits, alphas, sw_id
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
        offset: Optional[np.ndarray],
        splits: list,
        alphas: np.ndarray,
        sw_id: int,
    ) -> np.ndarray:
        """
        Return per-fold best deviance (shape: n_folds,) for a column set
        and alpha grid, using cached fold matrices and standardize stats.

        Offset (if provided) is sliced per-fold and forwarded to both the
        fold IRLS fit and the test-set predict, so deviances reflect the
        proper offset-bearing model.
        """
        fam  = self.glm_._family_instance
        fold_deviances = np.zeros(len(splits))
        full_mat, _ = self.cache.get_subset(df, cols)

        for fold_id, (train_idx, test_idx) in enumerate(splits):
            X_train = self.cache.get_fold_slice(fold_id, cols, full_mat)
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

            # Slice offset per-fold (None passes through transparently)
            offset_train = offset[train_idx] if offset is not None else None
            offset_test  = offset[test_idx]  if offset is not None else None

            # ── Standardize stats (cached per fold+cols+sw) ───────────────
            cached = self.cache.get_std_stats(fold_id, cols, sw_id)
            if cached is not None:
                col_means, col_stds = cached
                assert col_means.ndim == 1 and col_stds.ndim == 1
                n_feat = X_train.shape[1]
                inv_stds = np.where(col_stds > 0, 1.0 / col_stds, 1.0)
                shifts   = -col_means * inv_stds

                def _cached_std(X, sw, center, scale, lb, ub, Ai, P1, P2,
                                _shifts=shifts, _inv_stds=inv_stds,
                                _means=col_means, _stds=col_stds,
                                _n=n_feat):
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
                    return X_std, _means if center else np.zeros(_n), _stds, lb, ub, Ai, P1, P2

                std_fn = _cached_std
                needs_store = False
            else:
                _store: dict = {}

                def _capturing_std(X, sw, center, scale, lb, ub, Ai, P1, P2,
                                   _store=_store):
                    result = _utils_mod._original_standardize(
                        X, sw, center, scale, lb, ub, Ai, P1, P2
                    )
                    means_r = np.atleast_1d(np.asarray(result[1], dtype=float).ravel())
                    stds_r = (np.atleast_1d(np.asarray(result[2], dtype=float).ravel())
                              if result[2] is not None
                              else np.ones(X.shape[1]))
                    _store['means'] = means_r
                    _store['stds']  = stds_r
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
            coef         = None
            best_dev     = np.inf
            stored_first = False

            # Warm-start across alpha steps is safe only when n_features > 1.
            # With a single feature, glum's standardize_warm_start has a
            # numpy broadcasting edge case: np.squeeze on a (1,) array
            # yields a 0-d scalar; scalar.dot((1,)) returns a (1,) array,
            # which cannot be assigned via ``coef[0] += ...``.  This is an
            # upstream bug; the workaround keeps cv_select working in the
            # rare 1-feature-baseline case at the cost of cold IRLS starts
            # along the alpha path (still warm-started from previous CV
            # candidate evaluations within the same call, just not across
            # alpha within this single fold).
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
                    fit_kwargs = {"sample_weight": sw_train}
                    if offset_train is not None:
                        fit_kwargs["offset"] = offset_train
                    glm_fold.fit(X_train, y_train, **fit_kwargs)

                # Capture and store std stats after first alpha
                if needs_store and not stored_first and 'means' in _store:
                    self.cache.set_std_stats(
                        fold_id, cols, sw_id,
                        _store['means'], _store['stds']
                    )
                    col_means = _store['means']
                    col_stds  = _store['stds']
                    n_feat    = X_train.shape[1]
                    inv_stds  = np.where(col_stds > 0, 1.0 / col_stds, 1.0)
                    shifts    = -col_means * inv_stds

                    def _cached_std2(X, sw, center, scale, lb, ub, Ai, P1, P2,
                                     _shifts=shifts, _inv_stds=inv_stds,
                                     _means=col_means, _stds=col_stds,
                                     _n=n_feat):
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
                        return X_s, _means if center else np.zeros(_n), _stds, lb, ub, Ai, P1, P2

                    std_fn = _cached_std2
                    stored_first = True

                coef = np.concatenate([[glm_fold.intercept_], glm_fold.coef_])

                if offset_test is not None:
                    mu_test = glm_fold.predict(X_test, offset=offset_test)
                else:
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
            "glm_", "cache", "cache_stats_", "score_test_history_",
            "cv_history_",
        ):
            raise AttributeError(name)
        return getattr(self.glm_, name)
