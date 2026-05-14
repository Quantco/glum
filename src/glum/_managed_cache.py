"""
managed_cache
=============

Context manager that wraps the load-or-build-then-save lifecycle for
:class:`~glum.TabmatCache`.  Collapses the most common cross-session
workflow — "load a warm cache if the source file hasn't changed,
otherwise rebuild; persist on clean exit" — into a single ``with``
block.

Examples
--------
Default local-file persistence::

    from glum import managed_cache, GeneralizedLinearRegressor

    with managed_cache("data/insurance.parquet",
                       cat_cols=["Region", "Area"]) as cache:
        y = cache.read_target("ClaimNb")
        X, _ = cache.get_subset(cache.source_df,
                                ["VehAge", "BonusMalus", "Region"])
        GeneralizedLinearRegressor(family="poisson", alpha=0.01).fit(X, y)

Custom backend (e.g. shared NFS, Redis, S3 — anything implementing
:class:`~glum.CacheBackend`)::

    from glum import managed_cache, LocalFileBackend

    backend = LocalFileBackend("/mnt/shared/glum_caches")
    with managed_cache("data/insurance.parquet",
                       backend=backend, key="prod/insurance.pkl",
                       cat_cols=["Region"]) as cache:
        ...
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional, Sequence, Union

from ._cache_backend import CacheBackend, LocalFileBackend
from ._tabmat_cache import (
    SourceFingerprintError,
    TabmatCache,
    fingerprint_file,
)

__all__ = ["managed_cache"]

_logger = logging.getLogger(__name__)

# Default save_on_exit values
_VALID_SAVE_ON_EXIT = ("success", "always", "never")


@contextmanager
def managed_cache(
    source: Union[str, Path],
    backend: Optional[CacheBackend] = None,
    key: Optional[str] = None,
    columns: Optional[Sequence[str]] = None,
    cat_cols: Optional[Sequence[str]] = None,
    fold_mat_maxsize: int = 256,
    save_on_exit: str = "success",
    rebuild_on_mismatch: bool = True,
) -> Iterator[TabmatCache]:
    """
    Yield a warm :class:`~glum.TabmatCache` bound to ``source``.

    On entry, if the backend already has an entry under ``key``, load it
    and verify against ``source``'s current fingerprint:

    - **Match**: yield the loaded cache (fully warm — column subsets,
      fold slices, and standardize stats from earlier sessions are
      available immediately).
    - **Mismatch + rebuild_on_mismatch=True**: silently rebuild from
      ``source`` and yield the fresh cache.  The stale cached entry
      will be overwritten on exit if ``save_on_exit`` permits.
    - **Mismatch + rebuild_on_mismatch=False**: re-raise
      :class:`~glum.SourceFingerprintError`.

    If the backend has no entry under ``key``, build fresh from
    ``source``.

    On exit, persist the cache according to ``save_on_exit``:

    - ``"success"`` (default): persist iff the with-block exited
      without exception.
    - ``"always"``: persist regardless.
    - ``"never"``: discard.

    Parameters
    ----------
    source : str or Path
        Parquet file backing the cache.
    backend : CacheBackend, optional
        Storage backend.  Defaults to ``LocalFileBackend(".tabmat_cache")``
        in the current working directory.
    key : str, optional
        Key under which the cache is stored.  Defaults to
        ``Path(source).stem + ".pkl"``.
    columns, cat_cols, fold_mat_maxsize
        Passed through to :meth:`TabmatCache.from_parquet` on the
        rebuild path.
    save_on_exit : {"success", "always", "never"}, default "success"
        Persistence policy on context exit.
    rebuild_on_mismatch : bool, default True
        Whether to silently rebuild when the cached fingerprint
        disagrees with the current source file.

    Yields
    ------
    TabmatCache
        A cache bound to ``source``, ready for ``get_subset`` /
        ``read_target`` / ``source_df``.

    Examples
    --------
    >>> with managed_cache("data/insurance.parquet",
    ...                    cat_cols=["Region"]) as cache:
    ...     y = cache.read_target("ClaimNb")
    """
    if save_on_exit not in _VALID_SAVE_ON_EXIT:
        raise ValueError(
            f"save_on_exit must be one of {_VALID_SAVE_ON_EXIT}; "
            f"got {save_on_exit!r}"
        )

    source = Path(source)
    if backend is None:
        backend = LocalFileBackend(".tabmat_cache")
    if key is None:
        key = source.stem + ".pkl"

    cache = _enter_managed_cache(
        source=source,
        backend=backend,
        key=key,
        columns=columns,
        cat_cols=cat_cols,
        fold_mat_maxsize=fold_mat_maxsize,
        rebuild_on_mismatch=rebuild_on_mismatch,
    )

    exception_raised = False
    try:
        yield cache
    except Exception:
        exception_raised = True
        raise
    finally:
        should_save = (
            save_on_exit == "always"
            or (save_on_exit == "success" and not exception_raised)
        )
        if should_save:
            cache.save_to(backend, key)


def _enter_managed_cache(
    source: Path,
    backend: CacheBackend,
    key: str,
    columns: Optional[Sequence[str]],
    cat_cols: Optional[Sequence[str]],
    fold_mat_maxsize: int,
    rebuild_on_mismatch: bool,
) -> TabmatCache:
    """Resolve the cache on entry.  Factored out for testability."""
    if backend.exists(key):
        # Try the warm path
        try:
            cache = TabmatCache.load_from(backend, key)
        except Exception:
            _logger.warning(
                "managed_cache: failed to load cache from backend at %r; "
                "rebuilding from source.", key,
            )
            return _build_fresh_cache(
                source, columns, cat_cols, fold_mat_maxsize,
            )

        # Verify the source file hasn't moved.
        expected_fp = fingerprint_file(source)
        try:
            cache.verify_source(expected_fp)
            return cache
        except SourceFingerprintError:
            if not rebuild_on_mismatch:
                raise
            _logger.info(
                "managed_cache: cached fingerprint disagrees with source; "
                "rebuilding from %r.", str(source),
            )

    # Cold path: build fresh from parquet.
    return _build_fresh_cache(source, columns, cat_cols, fold_mat_maxsize)


def _build_fresh_cache(
    source: Path,
    columns: Optional[Sequence[str]],
    cat_cols: Optional[Sequence[str]],
    fold_mat_maxsize: int,
) -> TabmatCache:
    return TabmatCache.from_parquet(
        source,
        columns=columns,
        cat_cols=cat_cols,
        fold_mat_maxsize=fold_mat_maxsize,
    )
