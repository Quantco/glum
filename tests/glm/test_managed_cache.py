"""
Tests for glum._managed_cache.managed_cache — the context manager that
wraps load-or-build-then-save lifecycle for TabmatCache.
"""

from __future__ import annotations

import shutil
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from glum import (
    CacheBackend,
    LocalFileBackend,
    SourceFingerprintError,
    TabmatCache,
    fingerprint_file,
    managed_cache,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def parquet_path(tmp_path):
    """A small parquet file for testing managed_cache lifecycle."""
    p = tmp_path / "data.parquet"
    table = pa.table({
        "x":     np.arange(500, dtype=np.float64),
        "y":     np.random.default_rng(0).standard_normal(500),
        "label": np.array(["a", "b"] * 250),
    })
    pq.write_table(table, p)
    return p


@pytest.fixture
def backend(tmp_path):
    return LocalFileBackend(tmp_path / "cache_root")


class InMemoryBackend:
    def __init__(self):
        self._store: dict[str, bytes] = {}

    def exists(self, key): return key in self._store
    def read(self, key):   return self._store[key]
    def write(self, key, data): self._store[key] = data
    def delete(self, key): self._store.pop(key, None)


# ---------------------------------------------------------------------------
# Entry: build / load
# ---------------------------------------------------------------------------

class TestEntryBehavior:
    def test_first_call_builds_and_saves(self, parquet_path, backend):
        key = parquet_path.stem + ".pkl"
        assert not backend.exists(key)
        with managed_cache(parquet_path, backend=backend) as cache:
            assert isinstance(cache, TabmatCache)
            assert cache.source_fingerprint is not None
        assert backend.exists(key)

    def test_warm_call_loads_from_backend(self, parquet_path, backend):
        # Cold call to populate
        with managed_cache(parquet_path, backend=backend) as cache:
            _, _ = cache.get_subset(cache.source_df, ["x", "y"])
        # Second call: should load the populated subset
        with managed_cache(parquet_path, backend=backend) as cache:
            assert ("x", "y") in cache._subset_cache, (
                "Warm cache should contain the subset persisted by the first call"
            )

    def test_default_key_derived_from_source(self, parquet_path, backend):
        with managed_cache(parquet_path, backend=backend):
            pass
        assert backend.exists(parquet_path.stem + ".pkl")

    def test_explicit_key_used(self, parquet_path, backend):
        with managed_cache(parquet_path, backend=backend, key="custom/path.pkl"):
            pass
        assert backend.exists("custom/path.pkl")

    def test_columns_and_cat_cols_passed_through(self, parquet_path, backend):
        with managed_cache(
            parquet_path,
            backend=backend,
            columns=["x", "label"],
            cat_cols=["label"],
        ) as cache:
            assert cache._source_columns == ["x", "label"]
            assert cache._source_cat_cols == ["label"]
            assert isinstance(cache.source_df["label"].dtype, pd.CategoricalDtype)


# ---------------------------------------------------------------------------
# Mismatched source
# ---------------------------------------------------------------------------

class TestSourceMismatch:
    def test_modified_source_silently_rebuilds_by_default(
        self, parquet_path, backend,
    ):
        with managed_cache(parquet_path, backend=backend) as cache:
            original_fp = cache.source_fingerprint

        # Overwrite the parquet with different content
        time.sleep(0.05)
        new_table = pa.table({
            "x":     np.arange(100, dtype=np.float64),
            "y":     np.zeros(100),
            "label": np.array(["c"] * 100),
        })
        pq.write_table(new_table, parquet_path)

        with managed_cache(parquet_path, backend=backend) as cache:
            # Cache rebuilt against the new file
            assert cache.source_fingerprint != original_fp
            assert cache.source_df.shape == (100, 3)

    def test_modified_source_raises_when_rebuild_disabled(
        self, parquet_path, backend,
    ):
        with managed_cache(parquet_path, backend=backend):
            pass

        time.sleep(0.05)
        new_table = pa.table({"x": np.arange(10, dtype=np.float64)})
        pq.write_table(new_table, parquet_path)

        with pytest.raises(SourceFingerprintError):
            with managed_cache(
                parquet_path, backend=backend, rebuild_on_mismatch=False,
            ):
                pass


# ---------------------------------------------------------------------------
# save_on_exit semantics
# ---------------------------------------------------------------------------

class TestSaveOnExit:
    def test_success_default_persists_on_clean_exit(self, parquet_path, backend):
        with managed_cache(parquet_path, backend=backend):
            pass
        assert backend.exists(parquet_path.stem + ".pkl")

    def test_success_skips_persistence_on_exception(self, parquet_path, backend):
        with pytest.raises(RuntimeError, match="boom"):
            with managed_cache(parquet_path, backend=backend):
                raise RuntimeError("boom")
        assert not backend.exists(parquet_path.stem + ".pkl"), (
            "save_on_exit='success' must NOT persist when block raised"
        )

    def test_always_persists_on_clean_exit(self, parquet_path, backend):
        with managed_cache(parquet_path, backend=backend, save_on_exit="always"):
            pass
        assert backend.exists(parquet_path.stem + ".pkl")

    def test_always_persists_on_exception(self, parquet_path, backend):
        with pytest.raises(RuntimeError):
            with managed_cache(parquet_path, backend=backend, save_on_exit="always"):
                raise RuntimeError("boom")
        assert backend.exists(parquet_path.stem + ".pkl"), (
            "save_on_exit='always' must persist even when block raised"
        )

    def test_never_skips_persistence_on_clean_exit(self, parquet_path, backend):
        with managed_cache(parquet_path, backend=backend, save_on_exit="never"):
            pass
        assert not backend.exists(parquet_path.stem + ".pkl")

    def test_never_skips_persistence_on_exception(self, parquet_path, backend):
        with pytest.raises(RuntimeError):
            with managed_cache(parquet_path, backend=backend, save_on_exit="never"):
                raise RuntimeError("boom")
        assert not backend.exists(parquet_path.stem + ".pkl")

    def test_invalid_save_on_exit_raises(self, parquet_path, backend):
        with pytest.raises(ValueError, match="save_on_exit"):
            with managed_cache(parquet_path, backend=backend, save_on_exit="bogus"):
                pass


# ---------------------------------------------------------------------------
# Backend pluggability
# ---------------------------------------------------------------------------

class TestBackendPluggability:
    def test_custom_in_memory_backend(self, parquet_path):
        """Proof the seam works: a non-file backend works without changes."""
        backend = InMemoryBackend()
        with managed_cache(parquet_path, backend=backend, key="x.pkl") as cache:
            _ = cache.get_subset(cache.source_df, ["x", "y"])

        # First run saved to the in-memory backend
        assert backend.exists("x.pkl")
        # Second run loads from it
        with managed_cache(parquet_path, backend=backend, key="x.pkl") as cache:
            # Subset persisted
            assert ("x", "y") in cache._subset_cache

    def test_default_backend_created_when_none(self, parquet_path, tmp_path, monkeypatch):
        """When ``backend`` is None, a LocalFileBackend(./.tabmat_cache) is used."""
        monkeypatch.chdir(tmp_path)
        with managed_cache(parquet_path):
            pass
        cache_root = tmp_path / ".tabmat_cache"
        assert cache_root.exists()
        assert any(cache_root.iterdir())   # at least one file


# ---------------------------------------------------------------------------
# End-to-end with a real GLM fit
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_full_workflow(self, parquet_path, backend):
        import glum

        rng = np.random.default_rng(7)
        # Build target inline so the test stands alone
        y_table = pa.table({
            "x":     np.arange(500, dtype=np.float64),
            "y":     rng.standard_normal(500),
            "label": np.array(["a", "b"] * 250),
            "target": rng.poisson(np.exp(0.5 + 0.01 * np.arange(500))).astype(float),
        })
        path = parquet_path.parent / "withtarget.parquet"
        pq.write_table(y_table, path)

        with managed_cache(
            path,
            backend=backend,
            cat_cols=["label"],
        ) as cache:
            y = cache.read_target("target")
            X, _ = cache.get_subset(cache.source_df, ["x", "y", "label"])
            glm = glum.GeneralizedLinearRegressor(
                family="poisson", alpha=0.01, drop_first=True,
            )
            glm.fit(X, y)

        # Re-enter: warm path, same GLM fit produces the same coefs
        with managed_cache(
            path,
            backend=backend,
            cat_cols=["label"],
        ) as cache:
            y2 = cache.read_target("target")
            X2, _ = cache.get_subset(cache.source_df, ["x", "y", "label"])
            glm2 = glum.GeneralizedLinearRegressor(
                family="poisson", alpha=0.01, drop_first=True,
            )
            glm2.fit(X2, y2)

        np.testing.assert_allclose(glm.coef_, glm2.coef_, rtol=1e-10)
