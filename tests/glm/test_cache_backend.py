"""
Tests for glum._cache_backend — CacheBackend Protocol and LocalFileBackend.

Plus integration tests showing TabmatCache.save_to/load_from work with
any object satisfying the protocol (verified via an in-memory backend).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from glum import (
    CacheBackend,
    LocalFileBackend,
    TabmatCache,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def backend(tmp_path):
    return LocalFileBackend(tmp_path / "cache_root")


@pytest.fixture(scope="module")
def tiny_parquet(tmp_path_factory):
    """A small parquet file for save_to/load_from integration tests."""
    tmp = tmp_path_factory.mktemp("pq")
    p = tmp / "t.parquet"
    table = pa.table({
        "x":     np.arange(500, dtype=np.float64),
        "y":     np.random.default_rng(0).standard_normal(500),
        "label": np.array(["a", "b"] * 250),
    })
    pq.write_table(table, p)
    return p


# ---------------------------------------------------------------------------
# A minimal in-memory backend to prove the protocol works for non-file backends
# ---------------------------------------------------------------------------

class InMemoryBackend:
    """A non-file backend implementation purely for testing the seam."""

    def __init__(self):
        self._store: dict[str, bytes] = {}

    def exists(self, key: str) -> bool:
        return key in self._store

    def read(self, key: str) -> bytes:
        return self._store[key]

    def write(self, key: str, data: bytes) -> None:
        self._store[key] = data

    def delete(self, key: str) -> None:
        self._store.pop(key, None)


# ---------------------------------------------------------------------------
# LocalFileBackend
# ---------------------------------------------------------------------------

class TestLocalFileBackend:
    def test_roundtrip(self, backend):
        backend.write("a.pkl", b"hello world")
        assert backend.read("a.pkl") == b"hello world"

    def test_exists_before_and_after_write(self, backend):
        assert backend.exists("z.pkl") is False
        backend.write("z.pkl", b"x")
        assert backend.exists("z.pkl") is True

    def test_delete_makes_it_gone(self, backend):
        backend.write("k.pkl", b"v")
        assert backend.exists("k.pkl")
        backend.delete("k.pkl")
        assert backend.exists("k.pkl") is False

    def test_delete_missing_is_noop(self, backend):
        # Should not raise
        backend.delete("never_existed.pkl")

    def test_nested_key_creates_dirs(self, backend):
        backend.write("sub/dir/file.pkl", b"deep")
        assert backend.exists("sub/dir/file.pkl")
        assert backend.read("sub/dir/file.pkl") == b"deep"

    def test_overwrite(self, backend):
        backend.write("o.pkl", b"first")
        backend.write("o.pkl", b"second")
        assert backend.read("o.pkl") == b"second"

    def test_root_is_created(self, tmp_path):
        # Pass a non-existent path; constructor should mkdir it.
        deep = tmp_path / "a" / "b" / "c"
        assert not deep.exists()
        backend = LocalFileBackend(deep)
        assert deep.exists() and deep.is_dir()
        backend.write("ok.pkl", b"yes")
        assert (deep / "ok.pkl").exists()

    def test_repr(self, backend):
        r = repr(backend)
        assert "LocalFileBackend" in r
        assert str(backend.root) in r


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------

class TestProtocolConformance:
    def test_local_file_backend_satisfies_protocol(self, backend):
        assert isinstance(backend, CacheBackend)

    def test_in_memory_backend_satisfies_protocol(self):
        assert isinstance(InMemoryBackend(), CacheBackend)

    def test_non_backend_object_fails_protocol_check(self):
        # An object missing one of the four methods.
        class Broken:
            def exists(self, key): return False
            def read(self, key): return b""
            # missing write/delete
        assert not isinstance(Broken(), CacheBackend)


# ---------------------------------------------------------------------------
# TabmatCache.save_to / load_from
# ---------------------------------------------------------------------------

class TestSaveToLoadFrom:
    def test_local_backend_roundtrip(self, backend, tiny_parquet):
        cache = TabmatCache.from_parquet(tiny_parquet)
        cache.save_to(backend, "c.pkl")
        assert backend.exists("c.pkl")

        restored = TabmatCache.load_from(backend, "c.pkl")
        assert restored.stats() == cache.stats()
        assert restored.source_fingerprint == cache.source_fingerprint

    def test_in_memory_backend_roundtrip(self, tiny_parquet):
        """The protocol seam: a non-file backend works without changes."""
        backend = InMemoryBackend()
        cache = TabmatCache.from_parquet(tiny_parquet)
        cache.save_to(backend, "c.pkl")
        # No filesystem touched — proof the bytes-only protocol is enough.
        assert backend.exists("c.pkl")
        assert isinstance(backend._store["c.pkl"], bytes)

        restored = TabmatCache.load_from(backend, "c.pkl")
        assert restored.source_fingerprint == cache.source_fingerprint
        # Lazy source_df after load — proves rehydration metadata survived
        assert restored._source_columns == cache._source_columns

    def test_load_from_missing_key_raises(self, backend):
        # The backend's read() should propagate whatever error it raises
        # (FileNotFoundError for LocalFileBackend).
        with pytest.raises(FileNotFoundError):
            TabmatCache.load_from(backend, "does_not_exist.pkl")

    def test_save_to_then_save_to_overwrites(self, backend, tiny_parquet):
        cache1 = TabmatCache.from_parquet(tiny_parquet, columns=["x"])
        cache1.save_to(backend, "c.pkl")
        bytes1 = backend.read("c.pkl")

        cache2 = TabmatCache.from_parquet(tiny_parquet, columns=["x", "y"])
        cache2.save_to(backend, "c.pkl")
        bytes2 = backend.read("c.pkl")

        assert bytes1 != bytes2
        restored = TabmatCache.load_from(backend, "c.pkl")
        assert restored._source_columns == ["x", "y"]
