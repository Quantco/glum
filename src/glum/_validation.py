import copy
import typing

import numpy as np
import packaging.version
import pandas as pd
import sklearn as skl
import tabmat as tm
from scipy import sparse

from ._typing import ArrayLike, VectorLike

if packaging.version.parse(skl.__version__).release < (1, 6):
    keyword_finiteness = "force_all_finite"
    validate_data = skl.base.BaseEstimator._validate_data
else:
    keyword_finiteness = "ensure_all_finite"


def check_array_tabmat_compliant(mat: ArrayLike, drop_first: bool = False, **kwargs):
    to_copy = kwargs.get("copy", False)

    if isinstance(mat, pd.DataFrame):
        raise RuntimeError("DataFrames should have been converted by this point.")

    if isinstance(mat, tm.SplitMatrix):
        kwargs.update({"ensure_min_features": 0})
        new_matrices = [
            check_array_tabmat_compliant(m, drop_first=drop_first, **kwargs)
            for m in mat.matrices
        ]
        new_indices = [elt.copy() for elt in mat.indices] if to_copy else mat.indices
        return tm.SplitMatrix(new_matrices, new_indices)

    if isinstance(mat, tm.CategoricalMatrix):
        if to_copy:
            return copy.copy(mat)
        return mat

    if isinstance(mat, tm.StandardizedMatrix):
        return tm.StandardizedMatrix(
            check_array_tabmat_compliant(mat.mat, drop_first=drop_first, **kwargs),
            skl.utils.check_array(mat.shift, **kwargs),
        )

    original_type = type(mat)
    if isinstance(mat, (tm.DenseMatrix, tm.SparseMatrix)):
        res = skl.utils.check_array(mat.unpack(), **kwargs)
    else:
        res = skl.utils.check_array(mat, **kwargs)

    if res is not mat and original_type in (tm.DenseMatrix, tm.SparseMatrix):
        res = original_type(
            res,
            column_names=mat.column_names,  # type: ignore
            term_names=mat.term_names,  # type: ignore
        )

    return res


def check_X_y_tabmat_compliant(
    X: ArrayLike, y: typing.Union[VectorLike, sparse.spmatrix], **kwargs
) -> tuple[typing.Union[tm.MatrixBase, sparse.spmatrix, np.ndarray], np.ndarray]:
    """
    See the documentation for :func:`sklearn.utils.check_X_y`. This function
    behaves identically for inputs that are not from the Matrix package and
    fixes some parameters, such as ``'force_all_finite'``, to match the needs of
    GLMs.

    Returns
    -------
    X_converted : array-like
        The converted and validated X.
    y_converted : numpy.ndarray
        The converted and validated y.
    """
    if y is None:
        raise ValueError("y cannot be None")

    y = skl.utils.column_or_1d(y, warn=True)

    skl.utils.assert_all_finite(y)
    skl.utils.check_consistent_length(X, y)

    if y.dtype.kind == "O":
        y = y.astype(np.float64)

    X = check_array_tabmat_compliant(X, **kwargs)

    return X, y


def check_bounds(
    bounds: typing.Optional[typing.Union[float, VectorLike]], n_features: int, dtype
) -> typing.Optional[np.ndarray]:
    """Check that the bounds have the right shape."""
    if bounds is None:
        return None
    if np.isscalar(bounds):
        return np.full(n_features, bounds, dtype=dtype)

    bounds = skl.utils.check_array(
        bounds,
        accept_sparse=False,
        ensure_2d=False,
        dtype=dtype,
        **{keyword_finiteness: False},
    )

    bounds = typing.cast(np.ndarray, bounds)

    if bounds.ndim > 1:  # type: ignore
        raise ValueError("Bounds must be 1D array or scalar.")
    if bounds.shape[0] != n_features:  # type: ignore
        raise ValueError("Bounds must be the same length as X.shape[1].")

    return bounds


def check_inequality_constraints(
    A_ineq: typing.Optional[np.ndarray],
    b_ineq: typing.Optional[np.ndarray],
    n_features: int,
    dtype,
) -> tuple[typing.Union[None, np.ndarray], typing.Union[None, np.ndarray]]:
    """Check that the inequality constraints are well-defined."""
    if A_ineq is None or b_ineq is None:
        return None, None
    else:
        A_ineq = skl.utils.check_array(
            A_ineq,
            accept_sparse=False,
            ensure_2d=True,
            dtype=dtype,
            copy=True,
            **{keyword_finiteness: False},
        )
        b_ineq = skl.utils.check_array(
            b_ineq,
            accept_sparse=False,
            ensure_2d=False,
            dtype=dtype,
            copy=True,
            **{keyword_finiteness: False},
        )
        if A_ineq.shape[1] != n_features:  # type: ignore
            raise ValueError("A_ineq must have same number of columns as X.")
        if A_ineq.shape[0] != b_ineq.shape[0]:  # type: ignore
            raise ValueError("A_ineq and b_ineq must have same number of rows.")
        if b_ineq.ndim > 1:  # type: ignore
            raise ValueError("b_ineq must be 1D array.")
    return A_ineq, b_ineq


def check_offset(
    offset: typing.Optional[typing.Union[VectorLike, float]], n_rows: int, dtype
) -> typing.Optional[np.ndarray]:
    """
    Unlike weights, if the offset is ``None``, it can stay ``None``, so we only
    need to validate it when it is not.
    """
    if offset is None:
        return None
    if np.isscalar(offset):
        return np.full(n_rows, offset)

    offset = skl.utils.check_array(
        offset,
        accept_sparse=False,
        ensure_2d=False,
        dtype=dtype,
        **{keyword_finiteness: True},
    )

    offset = typing.cast(np.ndarray, offset)

    if offset.ndim > 1:  # type: ignore
        raise ValueError("Offsets must be 1D array or scalar.")
    if offset.shape[0] != n_rows:  # type: ignore
        raise ValueError("Offsets must have the same length as y.")

    return offset


def check_weights(
    sample_weight: typing.Optional[typing.Union[float, VectorLike]],
    n_samples: int,
    dtype,
    force_all_finite: bool = True,
) -> np.ndarray:
    """Check that sample weights are non-negative and have the right shape."""
    if sample_weight is None:
        return np.ones(n_samples, dtype=dtype)
    if np.isscalar(sample_weight):
        if sample_weight <= 0:  # type: ignore
            raise ValueError("Sample weights must be non-negative.")
        return np.full(n_samples, sample_weight, dtype=dtype)

    sample_weight = skl.utils.check_array(
        sample_weight,
        accept_sparse=False,
        ensure_2d=False,
        dtype=[np.float64, np.float32],
        **{keyword_finiteness: force_all_finite},
    )

    if sample_weight.ndim > 1:  # type: ignore
        raise ValueError("Sample weights must be 1D array or scalar.")
    if sample_weight.shape[0] != n_samples:  # type: ignore
        raise ValueError("Sample weights must have the same length as y.")
    if np.any(sample_weight < 0):  # type: ignore
        raise ValueError("Sample weights must be non-negative.")
    if np.sum(sample_weight) == 0:  # type: ignore
        raise ValueError("Sample weights must have at least one positive element.")

    return sample_weight  # type: ignore
