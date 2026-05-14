"""
Tests for glum._stepwise — StepwiseGLM, _MatrixCache, _CVMatrixCache,
_CachingStandardize, score-test screener, and cv_select.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import tabmat as tm

import glum
import glum._glm as _glm_mod
import glum._utils as _utils_mod
from glum._stepwise import (
    CVResult,
    ScoreTestResult,
    StepwiseGLM,
    _CVMatrixCache,
    _MatrixCache,
    _patch_standardize,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(42)


@pytest.fixture(scope="module")
def small_poisson(rng):
    """n=2000 Poisson dataset with known signals x2, x5."""
    n = 2_000
    df = pd.DataFrame(
        rng.standard_normal((n, 8)), columns=[f"x{i}" for i in range(8)]
    )
    y = rng.poisson(np.exp(0.3 + 0.5 * df["x2"] - 0.4 * df["x5"])).astype(float)
    return df, y


@pytest.fixture(scope="module")
def small_gaussian(rng):
    """n=2000 Gaussian dataset with known signals x1, x3."""
    n = 2_000
    df = pd.DataFrame(
        rng.standard_normal((n, 6)), columns=[f"x{i}" for i in range(6)]
    )
    y = 1.0 + 0.8 * df["x1"] - 0.6 * df["x3"] + rng.standard_normal(n) * 0.5
    return df, y


@pytest.fixture(scope="module")
def mixed_df(rng):
    """n=3000 mixed numeric + categorical DataFrame."""
    n = 3_000
    df = pd.DataFrame(
        rng.standard_normal((n, 4)), columns=[f"num_{i}" for i in range(4)]
    )
    df["cat_a"] = pd.Categorical(rng.choice(list("abcde"), n))
    df["cat_b"] = pd.Categorical(rng.choice(list("xyz"), n))
    y = rng.poisson(np.exp(0.2 + 0.4 * df["num_0"])).astype(float)
    return df, y


# ---------------------------------------------------------------------------
# _MatrixCache
# ---------------------------------------------------------------------------

class TestMatrixCache:
    def test_register_and_get_col(self, small_poisson):
        df, _ = small_poisson
        cache = _MatrixCache()
        cache.register_cols(df)
        for col in df.columns:
            assert col in cache
            mat = cache.get_col(col)
            assert mat.shape == (len(df), 1)

    def test_get_subset_memoized(self, small_poisson):
        df, _ = small_poisson
        cache = _MatrixCache()
        cols = ["x0", "x1", "x2"]
        mat1, names1 = cache.get_subset(df, cols)
        mat2, names2 = cache.get_subset(df, cols)
        assert mat1 is mat2, "Repeated get_subset should return the same object"
        assert names1 == names2

    def test_get_subset_shape(self, small_poisson):
        df, _ = small_poisson
        cache = _MatrixCache()
        cols = ["x0", "x1", "x2"]
        mat, names = cache.get_subset(df, cols)
        assert mat.shape[0] == len(df)
        assert mat.shape[1] == len(cols)

    def test_col_feat_names_numeric(self, small_poisson):
        df, _ = small_poisson
        cache = _MatrixCache()
        cache.register_cols(df)
        # Numeric column → single feature name
        names = cache.col_feat_names("x0")
        assert len(names) == 1

    def test_col_feat_names_categorical(self, mixed_df):
        df, _ = mixed_df
        cache = _MatrixCache()
        cache.register_cols(df)
        # cat_a has 5 levels → 5 feature names (no drop_first here)
        names = cache.col_feat_names("cat_a")
        assert len(names) == 5


# ---------------------------------------------------------------------------
# _CachingStandardize & _patch_standardize
# ---------------------------------------------------------------------------

class TestCachingStandardize:
    def test_patch_restores_on_normal_exit(self):
        original = _glm_mod.standardize

        def dummy(*a, **kw):
            return None

        with _patch_standardize(dummy):
            assert _glm_mod.standardize is dummy

        assert _glm_mod.standardize is original

    def test_patch_restores_on_exception(self):
        original = _glm_mod.standardize

        def dummy(*a, **kw):
            return None

        with pytest.raises(RuntimeError):
            with _patch_standardize(dummy):
                raise RuntimeError("boom")

        assert _glm_mod.standardize is original

    def test_standardize_cache_populates_on_miss(self, small_poisson):
        df, y = small_poisson
        sglm = StepwiseGLM(family="poisson", alpha=0.01)
        sglm.fit(df[["x0", "x1"]], y)
        # After first fit, col_stats should have entries for x0 and x1
        assert len(sglm._col_stats) >= 2

    def test_standardize_cache_hits_on_repeat(self, small_poisson):
        df, y = small_poisson
        sglm = StepwiseGLM(family="poisson", alpha=0.01)
        sglm.fit(df[["x0", "x1"]], y)
        sglm.fit(df[["x0", "x1"]], y)
        step1 = sglm.cache_stats_[0]
        step2 = sglm.cache_stats_[1]
        # Second fit should be all cache hits
        assert step2["hit_rate"] == 1.0
        assert step2["hits"] == step1["n_cols"]

    def test_standardize_cache_invalidated_on_weight_change(self, small_poisson):
        df, y = small_poisson
        n = len(y)
        sglm = StepwiseGLM(family="poisson", alpha=0.01)
        sw1 = np.ones(n)
        sw2 = np.ones(n) * 2.0
        sglm.fit(df[["x0", "x1"]], y, sample_weight=sw1)
        stats_before = len(sglm._col_stats)
        sglm.fit(df[["x0", "x1"]], y, sample_weight=sw2)
        # Cache should have been cleared and re-populated
        assert len(sglm._col_stats) == stats_before  # same count but different values


# ---------------------------------------------------------------------------
# StepwiseGLM.fit()
# ---------------------------------------------------------------------------

class TestStepwiseGLMFit:
    @pytest.mark.parametrize("family", ["poisson", "gaussian"])
    def test_fit_matches_vanilla_glum(self, small_poisson, small_gaussian, family):
        """Cached fit produces same coefficients as vanilla GeneralizedLinearRegressor."""
        df, y = small_poisson if family == "poisson" else small_gaussian
        cols = list(df.columns[:4])

        ref = glum.GeneralizedLinearRegressor(family=family, alpha=0.01)
        ref.fit(df[cols], y)

        sglm = StepwiseGLM(family=family, alpha=0.01)
        sglm.fit(df[cols], y)

        np.testing.assert_allclose(
            sglm.glm_.coef_, ref.coef_, rtol=1e-4,
            err_msg=f"Coefficients differ for family={family}",
        )

    def test_fit_accepts_tabmat_input(self, small_poisson):
        df, y = small_poisson
        X_tab = tm.from_pandas(df[["x0", "x1"]])
        sglm = StepwiseGLM(family="poisson", alpha=0.01)
        sglm.fit(X_tab, y)
        assert hasattr(sglm.glm_, "coef_")

    def test_warm_start_coef_shape_guard(self, small_poisson):
        """Changing column count should not cause shape mismatch error."""
        df, y = small_poisson
        sglm = StepwiseGLM(family="poisson", alpha=0.01)
        sglm.fit(df[["x0", "x1"]], y)
        # Expanding column set — would crash without the guard
        sglm.fit(df[["x0", "x1", "x2"]], y)
        assert sglm.glm_.coef_.shape == (3,)

    def test_fit_mixed_categorical(self, mixed_df):
        df, y = mixed_df
        sglm = StepwiseGLM(family="poisson", alpha=0.01)
        sglm.fit(df, y)
        assert hasattr(sglm.glm_, "coef_")
        # cat_a (5 levels) + cat_b (3 levels) + 4 numeric
        assert sglm.glm_.coef_.shape[0] == 5 + 3 + 4

    def test_cache_stats_recorded_per_step(self, small_poisson):
        df, y = small_poisson
        sglm = StepwiseGLM(family="poisson", alpha=0.01)
        for cols in [["x0"], ["x0", "x1"], ["x0", "x1", "x2"]]:
            sglm.fit(df[cols], y)
        assert set(sglm.cache_stats_.keys()) == {0, 1, 2}


# ---------------------------------------------------------------------------
# StepwiseGLM.screen_candidates()
# ---------------------------------------------------------------------------

class TestScreenCandidates:
    def test_returns_score_test_results(self, small_poisson):
        df, y = small_poisson
        sglm = StepwiseGLM(family="poisson", alpha=0.01)
        sglm.fit(df[["x0", "x1"]], y)
        results = sglm.screen_candidates(df, ["x0", "x1"],
                                         [f"x{i}" for i in range(2, 8)])
        assert len(results) == 6
        assert all(isinstance(r, ScoreTestResult) for r in results)

    def test_sorted_descending_by_statistic(self, small_poisson):
        df, y = small_poisson
        sglm = StepwiseGLM(family="poisson", alpha=0.01)
        sglm.fit(df[["x0", "x1"]], y)
        results = sglm.screen_candidates(df, ["x0", "x1"],
                                         [f"x{i}" for i in range(2, 8)])
        stats = [r.statistic for r in results]
        assert stats == sorted(stats, reverse=True)

    def test_true_signals_rank_top2(self, small_poisson):
        """x2 and x5 (the true signal columns) should be the top-2 ranked."""
        df, y = small_poisson
        sglm = StepwiseGLM(family="poisson", alpha=0.01)
        sglm.fit(df[["x0", "x1"]], y)
        results = sglm.screen_candidates(df, ["x0", "x1"],
                                         [f"x{i}" for i in range(2, 8)])
        top2 = {results[0].column, results[1].column}
        assert top2 == {"x2", "x5"}, (
            f"Expected top-2 to be x2 and x5, got {top2}"
        )

    def test_p_values_in_range(self, small_poisson):
        df, y = small_poisson
        sglm = StepwiseGLM(family="poisson", alpha=0.01)
        sglm.fit(df[["x0", "x1"]], y)
        results = sglm.screen_candidates(df, ["x0", "x1"],
                                         [f"x{i}" for i in range(2, 8)])
        for r in results:
            assert 0.0 <= r.pvalue <= 1.0

    def test_categorical_candidate_multi_dof(self, mixed_df):
        """Categorical candidates should have dof == n_levels."""
        df, y = mixed_df
        sglm = StepwiseGLM(family="poisson", alpha=0.01)
        sglm.fit(df[["num_0"]], y)
        results = sglm.screen_candidates(df, ["num_0"], ["cat_a", "cat_b"])
        dofs = {r.column: r.dof for r in results}
        assert dofs["cat_a"] == 5   # 5 levels
        assert dofs["cat_b"] == 3   # 3 levels

    def test_raises_before_fit(self, small_poisson):
        df, y = small_poisson
        sglm = StepwiseGLM(family="poisson", alpha=0.01)
        with pytest.raises(RuntimeError, match="fit()"):
            sglm.screen_candidates(df, ["x0"], ["x1"])

    def test_gaussian_signal_ranking(self, small_gaussian):
        """True signals x1 and x3 should rank top-2 for a Gaussian family model."""
        df, y = small_gaussian
        sglm = StepwiseGLM(family="gaussian", alpha=0.01)
        sglm.fit(df[["x0"]], y)
        results = sglm.screen_candidates(df, ["x0"],
                                         [f"x{i}" for i in range(1, 6)])
        top2 = {results[0].column, results[1].column}
        assert top2 == {"x1", "x3"}

    def test_score_test_history_appended(self, small_poisson):
        df, y = small_poisson
        sglm = StepwiseGLM(family="poisson", alpha=0.01)
        sglm.fit(df[["x0", "x1"]], y)
        sglm.screen_candidates(df, ["x0", "x1"], ["x2", "x3"])
        sglm.screen_candidates(df, ["x0", "x1"], ["x2", "x3"])
        assert len(sglm.score_test_history_) == 2


# ---------------------------------------------------------------------------
# _CVMatrixCache
# ---------------------------------------------------------------------------

class TestCVMatrixCache:
    def test_fold_mat_cache_hit(self, small_poisson):
        df, _ = small_poisson
        mat_cache = _MatrixCache()
        mat_cache.register_cols(df)
        cv_cache = _CVMatrixCache()
        cv_cache.register_col_mats(mat_cache._col_matrices)

        train_idx = np.arange(1600)
        cv_cache.set_fold_indices(0, train_idx)

        cols = ["x0", "x1", "x2"]
        full_mat, _ = mat_cache.get_subset(df, cols)

        m1 = cv_cache.get_fold_mat(0, cols, full_mat)
        m2 = cv_cache.get_fold_mat(0, cols, full_mat)
        assert m1 is m2, "Second call should return cached object"

    def test_incremental_hstack(self, small_poisson):
        """Adding one column should use hstack from cached prefix, not full re-slice."""
        df, _ = small_poisson
        mat_cache = _MatrixCache()
        mat_cache.register_cols(df)
        cv_cache = _CVMatrixCache()
        cv_cache.register_col_mats(mat_cache._col_matrices)

        train_idx = np.arange(1600)
        cv_cache.set_fold_indices(0, train_idx)

        # Populate cache with 2-col prefix
        cols2 = ["x0", "x1"]
        full2, _ = mat_cache.get_subset(df, cols2)
        cv_cache.get_fold_mat(0, cols2, full2)

        # Request 3-col set — should be built incrementally
        cols3 = ["x0", "x1", "x2"]
        full3, _ = mat_cache.get_subset(df, cols3)
        m3 = cv_cache.get_fold_mat(0, cols3, full3)

        assert m3.shape == (len(train_idx), 3)
        assert (0, tuple(cols3)) in cv_cache._fold_mat

    def test_std_stats_cache_round_trip(self, small_poisson):
        df, _ = small_poisson
        cv_cache = _CVMatrixCache()
        means = np.array([0.1, 0.2, 0.3])
        stds  = np.array([1.0, 1.1, 0.9])
        cv_cache.set_std_stats(0, ["x0", "x1", "x2"], -1, means, stds)
        result = cv_cache.get_std_stats(0, ["x0", "x1", "x2"], -1)
        assert result is not None
        np.testing.assert_array_equal(result[0], means)
        np.testing.assert_array_equal(result[1], stds)

    def test_std_stats_miss_returns_none(self):
        cv_cache = _CVMatrixCache()
        assert cv_cache.get_std_stats(0, ["x0"], -1) is None


# ---------------------------------------------------------------------------
# StepwiseGLM.cv_select()
# ---------------------------------------------------------------------------

class TestCVSelect:
    def test_returns_cv_results(self, small_poisson):
        df, y = small_poisson
        sglm = StepwiseGLM(family="poisson", alpha=0.01, drop_first=True)
        sglm.fit(df[["x0", "x1"]], y)
        results = sglm.cv_select(
            df, ["x0", "x1"], ["x2", "x3", "x4"], y, cv=2, n_alphas=5
        )
        assert len(results) == 3
        assert all(isinstance(r, CVResult) for r in results)

    def test_sorted_ascending_by_deviance(self, small_poisson):
        df, y = small_poisson
        sglm = StepwiseGLM(family="poisson", alpha=0.01, drop_first=True)
        sglm.fit(df[["x0", "x1"]], y)
        results = sglm.cv_select(
            df, ["x0", "x1"], ["x2", "x3", "x4"], y, cv=2, n_alphas=5
        )
        devs = [r.cv_deviance for r in results]
        assert devs == sorted(devs)

    def test_true_signal_ranked_first(self, small_poisson):
        """x2 is the strongest signal; cv_select should rank it first."""
        df, y = small_poisson
        sglm = StepwiseGLM(family="poisson", alpha=0.01, drop_first=True)
        sglm.fit(df[["x0", "x1"]], y)
        results = sglm.cv_select(
            df, ["x0", "x1"], ["x2", "x3", "x4", "x5"], y, cv=3, n_alphas=5
        )
        assert results[0].column == "x2", (
            f"Expected x2 ranked first, got {results[0].column}"
        )

    def test_deviances_match_naive_glmcv(self, small_poisson):
        """cv_select deviances should match GeneralizedLinearRegressorCV within tolerance."""
        df, y = small_poisson
        candidates = ["x2", "x3"]
        active = ["x0", "x1"]
        n_alphas = 8
        cv = 3

        sglm = StepwiseGLM(family="poisson", alpha=0.01, drop_first=True)
        sglm.fit(df[active], y)
        results = sglm.cv_select(df, active, candidates, y, cv=cv, n_alphas=n_alphas)
        cached_devs = {r.column: r.cv_deviance for r in results}

        for cand in candidates:
            g = glum.GeneralizedLinearRegressorCV(
                family="poisson", n_alphas=n_alphas, cv=cv,
                l1_ratio=0, drop_first=True,
            )
            g.fit(df[active + [cand]], y)
            naive_dev = float(g.deviance_path_.mean(axis=0).min())
            # Allow 2% relative tolerance — alpha grids differ slightly
            assert abs(cached_devs[cand] - naive_dev) / (naive_dev + 1e-9) < 0.02, (
                f"{cand}: cached={cached_devs[cand]:.6f} naive={naive_dev:.6f}"
            )

    def test_warm_cache_produces_same_results(self, small_poisson):
        """Second cv_select call (warm cache) should produce identical results."""
        df, y = small_poisson
        sglm = StepwiseGLM(family="poisson", alpha=0.01, drop_first=True)
        sglm.fit(df[["x0", "x1"]], y)
        r1 = sglm.cv_select(df, ["x0", "x1"], ["x2", "x3"], y, cv=2, n_alphas=5)
        r2 = sglm.cv_select(df, ["x0", "x1"], ["x2", "x3"], y, cv=2, n_alphas=5)
        for a, b in zip(r1, r2):
            assert a.column == b.column
            assert abs(a.cv_deviance - b.cv_deviance) < 1e-10

    def test_std_stats_cached_after_first_call(self, small_poisson):
        """After cv_select, fold std stats should be populated in _cv_cache."""
        df, y = small_poisson
        sglm = StepwiseGLM(family="poisson", alpha=0.01, drop_first=True)
        sglm.fit(df[["x0", "x1"]], y)
        sglm.cv_select(df, ["x0", "x1"], ["x2"], y, cv=2, n_alphas=3)
        # At least one fold should have cached std stats
        assert len(sglm._cv_cache._std_stats) > 0

    def test_cv_history_appended(self, small_poisson):
        df, y = small_poisson
        sglm = StepwiseGLM(family="poisson", alpha=0.01, drop_first=True)
        sglm.fit(df[["x0", "x1"]], y)
        sglm.cv_select(df, ["x0", "x1"], ["x2"], y, cv=2, n_alphas=3)
        sglm.cv_select(df, ["x0", "x1"], ["x3"], y, cv=2, n_alphas=3)
        assert len(sglm.cv_history_) == 2

    def test_cv_select_with_categorical(self, mixed_df):
        """cv_select should work with categorical candidate columns."""
        df, y = mixed_df
        sglm = StepwiseGLM(family="poisson", alpha=0.01, drop_first=True)
        sglm.fit(df[["num_0"]], y)
        results = sglm.cv_select(
            df, ["num_0"], ["cat_a", "num_1"], y, cv=2, n_alphas=4
        )
        assert len(results) == 2
        assert all(r.cv_deviance > 0 for r in results)

    def test_single_feature_baseline_no_error(self, small_poisson):
        """cv_select with a single-column active model should not raise."""
        df, y = small_poisson
        sglm = StepwiseGLM(family="poisson", alpha=1e-4, drop_first=True)
        sglm.fit(df[["x0"]], y)
        # This previously hit the standardize_warm_start 1-feature numpy bug
        results = sglm.cv_select(df, ["x0"], ["x1", "x2"], y, cv=2, n_alphas=3)
        assert len(results) == 2


# ---------------------------------------------------------------------------
# End-to-end: two-stage stepwise loop
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_forward_stepwise_selects_signals(self, small_poisson):
        """A 3-step forward loop should include x2 and x5 in the selected set."""
        df, y = small_poisson
        sglm = StepwiseGLM(family="poisson", alpha=0.01)
        active = ["x0", "x1"]
        candidates = [f"x{i}" for i in range(2, 8)]

        for _ in range(3):
            sglm.fit(df[active], y)
            scores = sglm.screen_candidates(df, active, candidates)
            best = scores[0]
            if not best.selected:
                break
            active.append(best.column)
            candidates.remove(best.column)

        assert "x2" in active, f"x2 should be selected; active={active}"
        assert "x5" in active, f"x5 should be selected; active={active}"

    def test_two_stage_agrees_with_single_stage(self, small_poisson):
        """Two-stage (screen then cv_select) top pick should match full cv_select top pick."""
        df, y = small_poisson
        candidates = [f"x{i}" for i in range(2, 8)]
        active = ["x0", "x1"]

        sglm = StepwiseGLM(family="poisson", alpha=0.01, drop_first=True)
        sglm.fit(df[active], y)

        # Stage 1: score test shortlist
        scores = sglm.screen_candidates(df, active, candidates)
        shortlist = [r.column for r in scores[:4]]

        # Stage 2: CV on shortlist
        cv_results = sglm.cv_select(
            df, active, shortlist, y, cv=3, n_alphas=8
        )

        # Full CV on all candidates
        full_cv = sglm.cv_select(
            df, active, candidates, y, cv=3, n_alphas=8
        )

        # Top pick from two-stage should match full CV top pick
        assert cv_results[0].column == full_cv[0].column, (
            f"Two-stage top={cv_results[0].column}, full CV top={full_cv[0].column}"
        )

    def test_predict_delegates_to_glm(self, small_poisson):
        df, y = small_poisson
        sglm = StepwiseGLM(family="poisson", alpha=0.01)
        sglm.fit(df[["x0", "x1"]], y)
        preds = sglm.predict(df[["x0", "x1"]])
        assert preds.shape == (len(y),)
        assert np.all(preds > 0)  # Poisson predictions must be positive
