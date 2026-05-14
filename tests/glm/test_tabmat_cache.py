"""
Tests for glum._tabmat_cache.TabmatCache — column / subset / fold caches
and on-disk persistence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import tabmat as tm

import glum
from glum import CacheVersionError, TabmatCache
from glum._tabmat_cache import _CACHE_VERSION


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(0)


@pytest.fixture(scope="module")
def numeric_df(rng):
    return pd.DataFrame(
        rng.standard_normal((2_000, 6)),
        columns=[f"x{i}" for i in range(6)],
    )


@pytest.fixture(scope="module")
def mixed_df(rng):
    df = pd.DataFrame(
        rng.standard_normal((2_000, 4)),
        columns=[f"num_{i}" for i in range(4)],
    )
    df["cat_a"] = pd.Categorical(rng.choice(list("abcde"), 2_000))
    df["cat_b"] = pd.Categorical(rng.choice(list("xyz"), 2_000))
    return df


@pytest.fixture
def cache():
    return TabmatCache()


# ---------------------------------------------------------------------------
# Column / subset layer
# ---------------------------------------------------------------------------

class TestColumnSubsetCache:
    def test_register_cols_populates(self, cache, numeric_df):
        cache.register_cols(numeric_df)
        for col in numeric_df.columns:
            assert col in cache
            assert cache.get_col(col).shape == (len(numeric_df), 1)

    def test_register_is_idempotent(self, cache, numeric_df):
        cache.register_cols(numeric_df)
        before = cache.get_col("x0")
        cache.register_cols(numeric_df)
        after = cache.get_col("x0")
        assert before is after, "Repeated register should not rebuild"

    def test_get_subset_memoized(self, cache, numeric_df):
        cols = ["x0", "x1", "x2"]
        mat1, names1 = cache.get_subset(numeric_df, cols)
        mat2, names2 = cache.get_subset(numeric_df, cols)
        assert mat1 is mat2
        assert names1 == names2

    def test_get_subset_shape(self, cache, numeric_df):
        cols = ["x0", "x1", "x2"]
        mat, _ = cache.get_subset(numeric_df, cols)
        assert mat.shape == (len(numeric_df), 3)

    def test_get_subset_distinct_keys(self, cache, numeric_df):
        m1, _ = cache.get_subset(numeric_df, ["x0", "x1"])
        m2, _ = cache.get_subset(numeric_df, ["x0", "x1", "x2"])
        assert m1.shape != m2.shape
        assert m1 is not m2

    def test_get_subset_fast_path_after_register(self, cache, numeric_df):
        """After register_cols, subset miss is built via hstack of cached cols."""
        cache.register_cols(numeric_df)
        # All cols registered → subset miss takes the hstack fast path.
        # We can't directly observe which path was taken, but we can verify
        # the result is equivalent to from_pandas (correctness) and that
        # the per-column store was NOT re-populated on the miss.
        n_cols_before = cache.stats()["n_cols"]
        m, _ = cache.get_subset(numeric_df, ["x0", "x1", "x2", "x3"])
        n_cols_after = cache.stats()["n_cols"]
        # No new per-column entries created — hstack reused the cached ones
        assert n_cols_before == n_cols_after
        # Correctness: assembled matrix matches a fresh from_pandas
        import tabmat as tm
        ref = tm.from_pandas(numeric_df[["x0", "x1", "x2", "x3"]])
        np.testing.assert_allclose(m.toarray(), ref.toarray())

    def test_get_subset_cold_path_backfills_per_col(self, cache, numeric_df):
        """Cold subset miss (without register_cols) should back-fill per-col."""
        # Don't register any cols up front
        assert cache.stats()["n_cols"] == 0
        m, _ = cache.get_subset(numeric_df, ["x0", "x1", "x2"])
        # Per-column store should now contain those 3 columns
        assert cache.stats()["n_cols"] == 3
        for c in ["x0", "x1", "x2"]:
            assert c in cache

    def test_col_feat_names_numeric(self, cache, numeric_df):
        cache.register_cols(numeric_df)
        names = cache.col_feat_names("x0")
        assert len(names) == 1

    def test_col_feat_names_categorical(self, cache, mixed_df):
        cache.register_cols(mixed_df)
        # cat_a has 5 levels
        assert len(cache.col_feat_names("cat_a")) == 5
        # cat_b has 3 levels
        assert len(cache.col_feat_names("cat_b")) == 3

    def test_contains_operator(self, cache, numeric_df):
        cache.register_cols(numeric_df[["x0", "x1"]])
        assert "x0" in cache
        assert "x_does_not_exist" not in cache


# ---------------------------------------------------------------------------
# Fold layer
# ---------------------------------------------------------------------------

class TestFoldCache:
    def _seed(self, cache, df, train_idx=None):
        cache.register_cols(df)
        idx = train_idx if train_idx is not None else np.arange(int(0.8 * len(df)))
        cache.set_fold_indices(0, idx)
        return idx

    def test_fold_slice_hit(self, cache, numeric_df):
        self._seed(cache, numeric_df)
        full, _ = cache.get_subset(numeric_df, ["x0", "x1"])
        m1 = cache.get_fold_slice(0, ["x0", "x1"], full)
        m2 = cache.get_fold_slice(0, ["x0", "x1"], full)
        assert m1 is m2

    def test_fold_slice_shape(self, cache, numeric_df):
        train_idx = self._seed(cache, numeric_df)
        full, _ = cache.get_subset(numeric_df, ["x0", "x1", "x2"])
        m = cache.get_fold_slice(0, ["x0", "x1", "x2"], full)
        assert m.shape == (len(train_idx), 3)

    def test_incremental_hstack_uses_prefix(self, cache, numeric_df):
        """When prefix is cached, new cols should be assembled via hstack."""
        self._seed(cache, numeric_df)
        full_pref, _ = cache.get_subset(numeric_df, ["x0", "x1"])
        cache.get_fold_slice(0, ["x0", "x1"], full_pref)   # warm prefix

        full_ext, _ = cache.get_subset(numeric_df, ["x0", "x1", "x2", "x3"])
        m = cache.get_fold_slice(0, ["x0", "x1", "x2", "x3"], full_ext)

        # Should produce an equivalent matrix to a fresh full slice
        train_idx = cache._fold_idx[0]
        fresh = full_ext[train_idx, :]
        np.testing.assert_allclose(m.toarray(), fresh.toarray())

    def test_full_miss_falls_back_to_full_slice(self, cache, numeric_df):
        self._seed(cache, numeric_df)
        # No prefix cached → fall through to full row-slice
        full, _ = cache.get_subset(numeric_df, ["x0", "x1", "x2"])
        m = cache.get_fold_slice(0, ["x0", "x1", "x2"], full)
        train_idx = cache._fold_idx[0]
        np.testing.assert_allclose(m.toarray(), full[train_idx, :].toarray())

    def test_lru_evicts_oldest_at_maxsize(self, numeric_df):
        cache = TabmatCache(fold_mat_maxsize=3)
        cache.register_cols(numeric_df)
        cache.set_fold_indices(0, np.arange(1000))

        for i in range(4):
            cols = [f"x{i}"]
            full, _ = cache.get_subset(numeric_df, cols)
            cache.get_fold_slice(0, cols, full)

        keys = [k[1] for k in cache._fold_mat.keys()]
        assert len(cache._fold_mat) == 3
        assert ("x0",) not in keys      # oldest evicted
        assert ("x3",) in keys          # newest present

    def test_lru_disabled_when_maxsize_zero(self, numeric_df):
        cache = TabmatCache(fold_mat_maxsize=0)
        cache.register_cols(numeric_df)
        cache.set_fold_indices(0, np.arange(1000))
        for i in range(10):
            full, _ = cache.get_subset(numeric_df, [f"x{i % 6}"])
            cache.get_fold_slice(0, [f"x{i % 6}_{i}"], full)
        # No bound → grows unbounded
        assert cache.stats()["n_fold_slices"] > 3

    def test_evicted_key_recomputes_correctly(self, numeric_df):
        cache = TabmatCache(fold_mat_maxsize=2)
        cache.register_cols(numeric_df)
        cache.set_fold_indices(0, np.arange(1000))

        full0, _ = cache.get_subset(numeric_df, ["x0"])
        full1, _ = cache.get_subset(numeric_df, ["x1"])
        full2, _ = cache.get_subset(numeric_df, ["x2"])

        m0 = cache.get_fold_slice(0, ["x0"], full0)
        cache.get_fold_slice(0, ["x1"], full1)
        cache.get_fold_slice(0, ["x2"], full2)   # evicts x0

        # x0 should be rebuilt correctly on re-request
        m0_again = cache.get_fold_slice(0, ["x0"], full0)
        np.testing.assert_allclose(m0.toarray(), m0_again.toarray())

    def test_std_stats_round_trip(self, cache):
        means = np.array([0.1, 0.2, 0.3])
        stds  = np.array([1.0, 1.1, 0.9])
        cache.set_std_stats(0, ["x0", "x1", "x2"], -1, means, stds)
        m, s = cache.get_std_stats(0, ["x0", "x1", "x2"], -1)
        np.testing.assert_array_equal(m, means)
        np.testing.assert_array_equal(s, stds)

    def test_std_stats_miss_returns_none(self, cache):
        assert cache.get_std_stats(0, ["x0"], -1) is None

    def test_std_stats_normalizes_shape(self, cache):
        """0-D and 2-D inputs should be flattened to 1-D on store."""
        cache.set_std_stats(0, ["x0"], -1, np.array(0.5), np.array(1.5))
        m, s = cache.get_std_stats(0, ["x0"], -1)
        assert m.ndim == 1 and s.ndim == 1
        assert m[0] == 0.5 and s[0] == 1.5


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_load_round_trip(self, tmp_path, numeric_df, mixed_df):
        cache = TabmatCache(fold_mat_maxsize=64)
        cache.register_cols(mixed_df)

        full, _ = cache.get_subset(mixed_df, ["num_0", "num_1", "cat_a"])
        train_idx = np.arange(int(0.8 * len(mixed_df)))
        cache.set_fold_indices(0, train_idx)
        cache.get_fold_slice(0, ["num_0", "num_1", "cat_a"], full)
        cache.set_std_stats(
            0, ["num_0", "num_1", "cat_a"], -1,
            np.array([0.1, 0.2, 0.3]),
            np.array([1.0, 1.1, 0.9]),
        )

        path = tmp_path / "cache.pkl"
        cache.save(path)
        assert path.exists()

        restored = TabmatCache.load(path)
        rs = restored.stats()
        os = cache.stats()
        assert rs == os

        # Same matrix shapes after reload
        rm, _ = restored.get_subset(mixed_df, ["num_0", "num_1", "cat_a"])
        om, _ = cache.get_subset(mixed_df, ["num_0", "num_1", "cat_a"])
        assert rm.shape == om.shape

        # Std stats survived
        m, s = restored.get_std_stats(0, ["num_0", "num_1", "cat_a"], -1)
        np.testing.assert_array_equal(m, [0.1, 0.2, 0.3])
        np.testing.assert_array_equal(s, [1.0, 1.1, 0.9])

    def test_load_version_mismatch_raises(self, tmp_path, numeric_df):
        import joblib
        cache = TabmatCache()
        cache.register_cols(numeric_df[["x0"]])

        path = tmp_path / "bad.pkl"
        cache.save(path)

        state = joblib.load(str(path))
        state["__cache_version__"] = _CACHE_VERSION + 99
        joblib.dump(state, str(path))

        with pytest.raises(CacheVersionError, match="cache version"):
            TabmatCache.load(path)

    def test_load_preserves_lru_bound(self, tmp_path, numeric_df):
        """After load, the fold-mat cache must still respect maxsize."""
        cache = TabmatCache(fold_mat_maxsize=3)
        cache.register_cols(numeric_df)
        cache.set_fold_indices(0, np.arange(1000))

        # Pre-warm with 2 entries (under the cap)
        for col in ["x0", "x1"]:
            full, _ = cache.get_subset(numeric_df, [col])
            cache.get_fold_slice(0, [col], full)

        path = tmp_path / "cache.pkl"
        cache.save(path)
        restored = TabmatCache.load(path)
        assert restored.fold_mat_maxsize == 3

        # Add more entries until the cap is exceeded — eviction must fire
        for col in ["x2", "x3", "x4"]:
            full, _ = restored.get_subset(numeric_df, [col])
            restored.get_fold_slice(0, [col], full)

        assert len(restored._fold_mat) == restored.fold_mat_maxsize
        # Earliest entry should have been evicted
        keys = [k[1] for k in restored._fold_mat.keys()]
        assert ("x0",) not in keys


# ---------------------------------------------------------------------------
# Stand-alone use with vanilla GLM
# ---------------------------------------------------------------------------

class TestStandaloneUsage:
    def test_cache_usable_with_vanilla_glm(self, numeric_df):
        """Demonstrate the primary use case: cache → vanilla GLM."""
        rng = np.random.default_rng(42)
        y = (
            1.0
            + 0.8 * numeric_df["x1"]
            - 0.6 * numeric_df["x3"]
            + rng.standard_normal(len(numeric_df)) * 0.5
        ).to_numpy()

        cache = TabmatCache()
        X_tab, _ = cache.get_subset(numeric_df, ["x0", "x1", "x2", "x3"])

        glm = glum.GeneralizedLinearRegressor(family="gaussian", alpha=0.01)
        glm.fit(X_tab, y)

        # Coefficients should match fitting the DataFrame directly
        ref = glum.GeneralizedLinearRegressor(family="gaussian", alpha=0.01)
        ref.fit(numeric_df[["x0", "x1", "x2", "x3"]], y)

        np.testing.assert_allclose(glm.coef_, ref.coef_, rtol=1e-4)

    def test_cache_reused_across_multiple_glms(self, numeric_df):
        """Same cache, multiple GLM fits with different column subsets."""
        rng = np.random.default_rng(42)
        y = rng.standard_normal(len(numeric_df))

        cache = TabmatCache()
        cache.register_cols(numeric_df)

        # First fit: 3 columns
        X1, _ = cache.get_subset(numeric_df, ["x0", "x1", "x2"])
        glum.GeneralizedLinearRegressor(family="gaussian", alpha=0.01).fit(X1, y)

        # Second fit: 4 columns — cache hit on shared 3-col prefix? No — different key
        # but from_pandas is paid only once per unique tuple
        X2, _ = cache.get_subset(numeric_df, ["x0", "x1", "x2", "x3"])
        glum.GeneralizedLinearRegressor(family="gaussian", alpha=0.01).fit(X2, y)

        assert cache.stats()["n_subsets"] == 2


# ---------------------------------------------------------------------------
# Introspection / clear
# ---------------------------------------------------------------------------

class TestIntrospection:
    def test_stats_keys(self, cache):
        s = cache.stats()
        for k in ("n_cols", "n_subsets", "n_fold_slices",
                  "n_std_stats", "n_folds", "fold_mat_maxsize"):
            assert k in s

    def test_clear_drops_everything(self, cache, numeric_df):
        cache.register_cols(numeric_df)
        cache.get_subset(numeric_df, ["x0", "x1"])
        cache.set_fold_indices(0, np.arange(1000))
        cache.set_std_stats(0, ["x0"], -1, np.array([0.0]), np.array([1.0]))
        full, _ = cache.get_subset(numeric_df, ["x0"])
        cache.get_fold_slice(0, ["x0"], full)

        cache.clear()
        s = cache.stats()
        assert s["n_cols"] == 0
        assert s["n_subsets"] == 0
        assert s["n_fold_slices"] == 0
        assert s["n_std_stats"] == 0
        assert s["n_folds"] == 0

    def test_clear_fold_slices_preserves_cols(self, cache, numeric_df):
        cache.register_cols(numeric_df)
        cache.get_subset(numeric_df, ["x0", "x1"])
        cache.set_fold_indices(0, np.arange(1000))
        cache.set_std_stats(0, ["x0"], -1, np.array([0.0]), np.array([1.0]))
        full, _ = cache.get_subset(numeric_df, ["x0"])
        cache.get_fold_slice(0, ["x0"], full)

        cache.clear_fold_slices()
        s = cache.stats()
        assert s["n_cols"] > 0
        assert s["n_subsets"] > 0
        assert s["n_fold_slices"] == 0
        assert s["n_std_stats"] == 0

    def test_invalid_maxsize_raises(self):
        with pytest.raises(ValueError):
            TabmatCache(fold_mat_maxsize=-1)
