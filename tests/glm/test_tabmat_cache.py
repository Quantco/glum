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
from glum import (
    CacheVersionError,
    SourceFingerprintError,
    TabmatCache,
    fingerprint_file,
)
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


# ---------------------------------------------------------------------------
# DataFrame mutation detection
# ---------------------------------------------------------------------------

class TestMutationDetection:
    """
    The cache fingerprints the source DataFrame on first contact and
    invalidates all entries when the fingerprint changes.  Detects
    reassignment, row count changes, and column rename / reorder.
    """

    def test_reassigned_df_invalidates_cache(self, cache, numeric_df):
        cache.register_cols(numeric_df)
        cache.get_subset(numeric_df, ["x0", "x1"])
        assert cache.stats()["n_cols"] > 0

        # Reassigning to a copy changes id(df).
        df2 = numeric_df.copy()
        cache.get_subset(df2, ["x0", "x1"])

        # Old per-column entries cleared and rebuilt against df2.
        # n_cols should be 2 (just x0 and x1, populated via cold-path back-fill).
        assert cache.stats()["n_cols"] == 2

    def test_added_row_invalidates_cache(self, cache, numeric_df):
        cache.register_cols(numeric_df[["x0", "x1"]])
        n_before_cols = cache.stats()["n_cols"]
        assert n_before_cols == 2

        # Append a row → df.shape changes.
        df_bigger = pd.concat([numeric_df, numeric_df.iloc[[0]]], ignore_index=True)
        cache.register_cols(df_bigger[["x0", "x1"]])

        # Cache should have been cleared and re-populated.
        # New per-column matrices reflect the longer DataFrame.
        for c in ["x0", "x1"]:
            assert cache.get_col(c).shape[0] == len(df_bigger)

    def test_renamed_column_invalidates_cache(self, cache, numeric_df):
        cache.register_cols(numeric_df)
        # Rename one column → column-tuple hash changes.
        df_renamed = numeric_df.rename(columns={"x0": "x0_renamed"})
        cache.register_cols(df_renamed[["x0_renamed", "x1"]])

        # Old x0 should be gone; new x0_renamed should be present.
        assert "x0" not in cache
        assert "x0_renamed" in cache

    def test_first_call_records_fingerprint(self, cache, numeric_df):
        assert cache._df_fingerprint is None
        cache.register_cols(numeric_df[["x0"]])
        assert cache._df_fingerprint is not None

    def test_value_mutation_via_iloc_may_or_may_not_be_detected(
        self, cache, numeric_df,
    ):
        """
        Documented limitation: in-place value mutation through pandas
        APIs is not guaranteed to invalidate the cache.

        Modern pandas Copy-on-Write *can* trigger an ``id(df)`` change
        on assignment (which our fingerprint catches incidentally), but
        this is an implementation detail of pandas — we don't rely on
        it.  The fingerprint only formally tracks ``(id, shape, cols)``.

        This test pins the contract: after a value mutation, the cache
        may still report success, and the burden is on the user to call
        :meth:`TabmatCache.clear`.  We verify the fingerprint behavior
        is at least *consistent* (either changed or unchanged), without
        asserting which.
        """
        cache.register_cols(numeric_df[["x0", "x1"]])
        fp_before = cache._df_fingerprint

        # Make a working copy and mutate it.  We intentionally don't
        # mutate `numeric_df` itself so we don't pollute the module-
        # scoped fixture for other tests.
        df_mut = numeric_df.copy()
        df_mut.iloc[0, 0] = 9999.0

        # Whatever pandas does internally, our cache is not contractually
        # required to detect this — just to behave correctly afterwards.
        cache.register_cols(df_mut[["x0", "x1"]])
        # Cache is still in a valid state (no exception, can serve
        # subsequent get_col calls).
        assert cache.get_col("x0").shape[0] == len(df_mut)

    def test_clear_resets_fingerprint(self, cache, numeric_df):
        cache.register_cols(numeric_df[["x0"]])
        assert cache._df_fingerprint is not None
        cache.clear()
        assert cache._df_fingerprint is None

    def test_load_does_not_carry_fingerprint(self, tmp_path, numeric_df):
        """
        After save/load, the fingerprint is None so the next call can
        bind to whatever DataFrame the user supplies.
        """
        cache = TabmatCache()
        cache.register_cols(numeric_df[["x0", "x1"]])
        path = tmp_path / "cache.pkl"
        cache.save(path)

        restored = TabmatCache.load(path)
        assert restored._df_fingerprint is None
        # Subsequent register_cols on the same DataFrame should not
        # clear, since the fingerprint is being established for the
        # first time post-load.
        n_before = restored.stats()["n_cols"]
        restored.register_cols(numeric_df[["x0", "x1"]])
        assert restored.stats()["n_cols"] == n_before


# ---------------------------------------------------------------------------
# fingerprint_file helper
# ---------------------------------------------------------------------------

class TestFingerprintFile:
    def test_basic_fingerprint_shape(self, tmp_path):
        p = tmp_path / "hello.bin"
        p.write_bytes(b"hello")
        fp = fingerprint_file(p)
        assert fp[0] == "file"
        assert fp[1] == str(p.resolve())
        assert fp[2] == 5
        assert isinstance(fp[3], int)   # mtime_ns

    def test_size_change_changes_fingerprint(self, tmp_path):
        p = tmp_path / "f.bin"
        p.write_bytes(b"hello")
        fp1 = fingerprint_file(p)
        p.write_bytes(b"hello world")   # different size
        fp2 = fingerprint_file(p)
        assert fp1 != fp2
        assert fp1[2] != fp2[2]   # size differs

    def test_mtime_change_changes_fingerprint(self, tmp_path):
        import os, time
        p = tmp_path / "f.bin"
        p.write_bytes(b"hello")
        fp1 = fingerprint_file(p)
        # Forcibly bump mtime
        old_size = p.stat().st_size
        future = time.time() + 100
        os.utime(p, (future, future))
        fp2 = fingerprint_file(p)
        assert fp1 != fp2
        assert fp1[3] != fp2[3]   # mtime_ns differs
        assert fp1[2] == fp2[2] == old_size

    def test_resolves_absolute_path(self, tmp_path):
        import os
        p = tmp_path / "f.bin"
        p.write_bytes(b"x")
        # Pass a relative path — fingerprint should still be absolute
        cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            fp = fingerprint_file("f.bin")
            assert fp[1] == str(p.resolve())
        finally:
            os.chdir(cwd)


# ---------------------------------------------------------------------------
# Source fingerprint binding & verify_source
# ---------------------------------------------------------------------------

class TestSourceFingerprint:
    def test_default_is_none(self):
        c = TabmatCache()
        assert c.source_fingerprint is None

    def test_set_and_get(self):
        c = TabmatCache()
        c.set_source_fingerprint(("custom", "v1", 42))
        assert c.source_fingerprint == ("custom", "v1", 42)

    def test_verify_matching_succeeds(self):
        c = TabmatCache()
        fp = ("file", "/tmp/x", 100, 1234567890)
        c.set_source_fingerprint(fp)
        assert c.verify_source(fp) is True

    def test_verify_mismatch_raises_in_strict_mode(self):
        c = TabmatCache()
        c.set_source_fingerprint(("a",))
        with pytest.raises(SourceFingerprintError, match="mismatch"):
            c.verify_source(("b",))

    def test_verify_mismatch_returns_false_in_lax_mode(self):
        c = TabmatCache()
        c.set_source_fingerprint(("a",))
        assert c.verify_source(("b",), strict=False) is False

    def test_verify_unset_raises_in_strict_mode(self):
        c = TabmatCache()
        with pytest.raises(SourceFingerprintError, match="No source fingerprint"):
            c.verify_source(("a",))

    def test_verify_unset_returns_false_in_lax_mode(self):
        c = TabmatCache()
        assert c.verify_source(("a",), strict=False) is False

    def test_clear_resets_source_fingerprint(self):
        c = TabmatCache()
        c.set_source_fingerprint(("a",))
        c.clear()
        assert c.source_fingerprint is None

    def test_source_fingerprint_round_trips_save_load(self, tmp_path):
        c = TabmatCache()
        c.set_source_fingerprint(("file", "/x", 7, 9))
        path = tmp_path / "c.pkl"
        c.save(path)
        restored = TabmatCache.load(path)
        assert restored.source_fingerprint == ("file", "/x", 7, 9)


# ---------------------------------------------------------------------------
# from_parquet factory
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_parquet(tmp_path_factory):
    """Write a small parquet file with mixed numeric + string columns."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    tmp = tmp_path_factory.mktemp("pq")
    p = tmp / "tiny.parquet"

    table = pa.table({
        "num1":  np.arange(1000, dtype=np.float64),
        "num2":  np.random.default_rng(0).standard_normal(1000),
        "cat_a": np.array(["a", "b", "c"] * 333 + ["a"]),
        "cat_b": np.array(["x", "y"] * 500),
    })
    pq.write_table(table, p)
    return p


class TestFromParquet:
    def test_returns_cache_with_fingerprint_bound(self, tiny_parquet):
        cache = TabmatCache.from_parquet(tiny_parquet)
        assert cache.source_fingerprint is not None
        assert cache.source_fingerprint[0] == "file"
        assert cache.source_fingerprint[1] == str(tiny_parquet.resolve())

    def test_register_cols_default_true_skips_plain_strings(self, tiny_parquet):
        """
        Without ``cat_cols``, plain string columns are skipped during
        registration — the user opted out of treating them as categorical.
        Only numeric / bool / Categorical dtypes are registered.
        """
        cache = TabmatCache.from_parquet(tiny_parquet)
        assert cache.stats()["n_cols"] == 2   # num1, num2 only
        assert "num1" in cache and "num2" in cache
        assert "cat_a" not in cache and "cat_b" not in cache

    def test_register_cols_default_true_with_cat_cols(self, tiny_parquet):
        """When cat_cols is provided, those columns get registered too."""
        cache = TabmatCache.from_parquet(
            tiny_parquet, cat_cols=["cat_a", "cat_b"],
        )
        assert cache.stats()["n_cols"] == 4

    def test_register_cols_false_defers(self, tiny_parquet):
        cache = TabmatCache.from_parquet(tiny_parquet, register_cols=False)
        assert cache.stats()["n_cols"] == 0
        # source_df is still attached so user can register selectively
        assert hasattr(cache, "source_df")

    def test_columns_subset(self, tiny_parquet):
        cache = TabmatCache.from_parquet(tiny_parquet, columns=["num1", "num2"])
        assert cache.stats()["n_cols"] == 2
        assert "num1" in cache
        assert "cat_a" not in cache

    def test_cat_cols_become_categorical(self, tiny_parquet):
        cache = TabmatCache.from_parquet(
            tiny_parquet,
            cat_cols=["cat_a", "cat_b"],
        )
        # cat_a has 3 levels → 3 feature names (no drop_first)
        assert len(cache.col_feat_names("cat_a")) == 3
        assert len(cache.col_feat_names("cat_b")) == 2

    def test_source_df_attached(self, tiny_parquet):
        cache = TabmatCache.from_parquet(tiny_parquet)
        assert isinstance(cache.source_df, pd.DataFrame)
        assert len(cache.source_df) == 1000

    def test_can_fit_glm_on_subset(self, tiny_parquet):
        import glum
        cache = TabmatCache.from_parquet(
            tiny_parquet, cat_cols=["cat_a"],
        )
        df = cache.source_df
        rng = np.random.default_rng(1)
        # num1 is np.arange(1000); use num2 (~standard normal) as the signal.
        y = rng.poisson(np.exp(0.3 + 0.5 * df["num2"])).astype(float)
        X, _ = cache.get_subset(df, ["num2", "cat_a"])
        glm = glum.GeneralizedLinearRegressor(
            family="poisson", alpha=0.01, drop_first=True,
        )
        glm.fit(X, y)
        # X has 1 numeric col + (3 - 1) dropped-first dummies = 3 cols total
        assert glm.coef_.shape[0] == X.shape[1]

    def test_verify_source_against_original_file(self, tiny_parquet):
        cache = TabmatCache.from_parquet(tiny_parquet)
        fp = fingerprint_file(tiny_parquet)
        assert cache.verify_source(fp) is True

    def test_verify_source_detects_modified_file(self, tiny_parquet, tmp_path):
        """If the parquet is replaced with new content, verify_source raises."""
        import pyarrow as pa, pyarrow.parquet as pq, shutil, time

        # Copy the original to a new location, build cache, then overwrite the copy.
        target = tmp_path / "data.parquet"
        shutil.copy(tiny_parquet, target)

        cache = TabmatCache.from_parquet(target)
        # Sleep a hair to ensure mtime_ns changes (1ns resolution may collide
        # on fast filesystems).
        time.sleep(0.05)

        new_table = pa.table({"num1": np.arange(500, dtype=np.float64)})
        pq.write_table(new_table, target)   # different size & mtime

        fp_new = fingerprint_file(target)
        with pytest.raises(SourceFingerprintError):
            cache.verify_source(fp_new)

    def test_save_load_preserves_source_fingerprint(self, tiny_parquet, tmp_path):
        cache = TabmatCache.from_parquet(tiny_parquet)
        path = tmp_path / "cache.pkl"
        cache.save(path)
        restored = TabmatCache.load(path)
        # Fingerprint round-trips cleanly
        assert restored.source_fingerprint == cache.source_fingerprint
        # And verify_source against the original file still succeeds
        assert restored.verify_source(fingerprint_file(tiny_parquet)) is True

    def test_source_df_not_persisted(self, tiny_parquet, tmp_path):
        """source_df is a convenience attribute, NOT pickled."""
        cache = TabmatCache.from_parquet(tiny_parquet)
        path = tmp_path / "cache.pkl"
        cache.save(path)
        restored = TabmatCache.load(path)
        assert not hasattr(restored, "source_df"), (
            "source_df should not survive save/load; users re-bind manually"
        )
