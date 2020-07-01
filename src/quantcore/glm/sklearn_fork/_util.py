from typing import Union

import numpy as np
from scipy import sparse

from quantcore.glm.matrix import MatrixBase, StandardizedMat


def _safe_lin_pred(
    X: Union[MatrixBase, StandardizedMat], coef: np.ndarray, offset: np.ndarray = None
) -> np.ndarray:
    """Compute the linear predictor taking care if intercept is present."""
    idx_offset = 0 if X.shape[1] == coef.shape[0] else 1
    nonzero_coefs = np.where(coef[idx_offset:] != 0.0)[0].astype(np.int32)
    res = X.dot(
        coef[idx_offset:],
        rows=np.arange(X.shape[0], dtype=np.int32),
        cols=nonzero_coefs,
    )

    if idx_offset == 1:
        res += coef[0]
    if offset is not None:
        return res + offset
    return res


def _safe_sandwich_dot(
    X: Union[MatrixBase, StandardizedMat],
    d: np.ndarray,
    rows: np.ndarray = None,
    cols: np.ndarray = None,
    intercept=False,
) -> np.ndarray:
    """Compute sandwich product X.T @ diag(d) @ X.

    With ``intercept=True``, X is treated as if a column of 1 were appended as
    first column of X.
    X can be sparse, d must be an ndarray. Always returns a ndarray."""
    result = X.sandwich(d, rows, cols)
    if isinstance(result, sparse.dia_matrix):
        result = result.A

    if intercept:
        dim = result.shape[0] + 1
        res_including_intercept = np.empty((dim, dim), dtype=X.dtype)
        res_including_intercept[0, 0] = d.sum()
        res_including_intercept[1:, 0] = X.transpose_dot(d, rows, cols)
        res_including_intercept[0, 1:] = res_including_intercept[1:, 0]
        res_including_intercept[1:, 1:] = result
    else:
        res_including_intercept = result
    return res_including_intercept


def _safe_toarray(X) -> np.ndarray:
    """Returns a numpy array."""
    if sparse.issparse(X):
        return X.toarray()
    else:
        return np.asarray(X)
