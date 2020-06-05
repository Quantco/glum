from typing import Union

import numpy as np
from scipy import sparse

from quantcore.glm.matrix import ColScaledMat, MatrixBase


def _safe_lin_pred(
    X: Union[MatrixBase, ColScaledMat], coef: np.ndarray, offset: np.ndarray = None
) -> np.ndarray:
    """Compute the linear predictor taking care if intercept is present."""
    res = X.dot(coef[1:]) + coef[0] if coef.size == X.shape[1] + 1 else X.dot(coef)
    if offset is not None:
        return res + offset
    return res


def _safe_sandwich_dot(
    X: Union[MatrixBase, ColScaledMat], d: np.ndarray, intercept=False
) -> np.ndarray:
    """Compute sandwich product X.T @ diag(d) @ X.

    With ``intercept=True``, X is treated as if a column of 1 were appended as
    first column of X.
    X can be sparse, d must be an ndarray. Always returns a ndarray."""
    result = X.sandwich(d)

    if intercept:
        dim = X.shape[1] + 1
        res_including_intercept = np.empty((dim, dim), dtype=X.dtype)
        res_including_intercept[0, 0] = d.sum()
        res_including_intercept[1:, 0] = d @ X
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
