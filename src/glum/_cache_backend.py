"""
CacheBackend
============

Storage backends for :class:`glum.TabmatCache`.

A :class:`CacheBackend` is a small protocol describing where a cache
lives: on the local filesystem, in Redis, in S3/GCS/Azure Blob, etc.
The protocol intentionally trades in raw ``bytes`` rather than Python
objects so distributed backends can implement it without depending on
:mod:`joblib` or pickle semantics — serialization is the cache's
responsibility, not the backend's.

This module ships :class:`LocalFileBackend` as the default implementation.
Other backends (Redis, blob storage) are out of scope for this module but
will plug into the same protocol; users with custom storage needs can
implement their own by satisfying the four-method protocol.

Examples
--------
Default local-file usage::

    from glum import TabmatCache, LocalFileBackend

    backend = LocalFileBackend(".tabmat_cache")
    cache   = TabmatCache.from_parquet("data/insurance.parquet",
                                       cat_cols=["Region"])
    cache.save_to(backend, "insurance.pkl")

    # Next session:
    cache = TabmatCache.load_from(backend, "insurance.pkl")

A trivial in-memory backend (e.g. for testing) is just four methods::

    class InMemoryBackend:
        def __init__(self): self._d: dict[str, bytes] = {}
        def exists(self, key): return key in self._d
        def read(self, key):   return self._d[key]
        def write(self, key, data): self._d[key] = data
        def delete(self, key): self._d.pop(key, None)
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, Union, runtime_checkable

__all__ = ["CacheBackend", "LocalFileBackend"]


@runtime_checkable
class CacheBackend(Protocol):
    """
    Storage backend protocol for :class:`TabmatCache` persistence.

    All implementations must provide these four methods.  The
    ``@runtime_checkable`` decoration means ``isinstance(obj,
    CacheBackend)`` works at runtime for duck-typed implementations.
    """

    def exists(self, key: str) -> bool:
        """Return whether ``key`` is present in the backend."""
        ...

    def read(self, key: str) -> bytes:
        """Read the bytes stored under ``key``.  Raises on miss."""
        ...

    def write(self, key: str, data: bytes) -> None:
        """Write ``data`` under ``key``, overwriting any prior value."""
        ...

    def delete(self, key: str) -> None:
        """Remove ``key`` from the backend.  No-op if absent."""
        ...


class LocalFileBackend:
    """
    Default backend: stores each key as a file under ``root``.

    Keys are interpreted as filesystem-relative paths, so a key like
    ``"models/insurance.pkl"`` becomes ``root/models/insurance.pkl``.
    Intermediate directories are created on demand by :meth:`write`.

    Parameters
    ----------
    root : str or Path
        Root directory for stored objects.  Created (with parents) if
        it doesn't exist.

    Examples
    --------
    >>> backend = LocalFileBackend("/tmp/glum_cache")
    >>> backend.write("a.pkl", b"hello")
    >>> backend.exists("a.pkl")
    True
    >>> backend.read("a.pkl")
    b'hello'
    """

    def __init__(self, root: Union[str, Path]):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # CacheBackend protocol
    # ------------------------------------------------------------------

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def read(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def write(self, key: str, data: bytes) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    def delete(self, key: str) -> None:
        p = self._path(key)
        if p.exists():
            p.unlink()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _path(self, key: str) -> Path:
        return self.root / Path(key)

    def __repr__(self) -> str:
        return f"LocalFileBackend(root={str(self.root)!r})"
