import numpy as np
from scipy import sparse

from glm_benchmarks.matrix import MatrixBase


def _safe_lin_pred(
    X: MatrixBase, coef: np.ndarray, offset: np.ndarray = None
) -> np.ndarray:
    """Compute the linear predictor taking care if intercept is present."""
    res = X.dot(coef[1:]) + coef[0] if coef.size == X.shape[1] + 1 else X.dot(coef)
    if offset is not None:
        return res + offset
    return res


def _safe_sandwich_dot(
    X: MatrixBase, d: np.ndarray, cols: np.ndarray, intercept=False
) -> np.ndarray:
    """Compute sandwich product X.T @ diag(d) @ X.

    With ``intercept=True``, X is treated as if a column of 1 were appended as
    first column of X.
    X can be sparse, d must be an ndarray. Always returns a ndarray."""
    import time

    start = time.time()
    result = X.sandwich(d, cols)
    print(len(cols), time.time() - start)

    if intercept:
        dim = cols.shape[0] + 1
        res_including_intercept = np.empty((dim, dim), dtype=X.dtype)
        res_including_intercept[0, 0] = d.sum()
        res_including_intercept[1:, 0] = X.limited_rmatvec(d, cols)
        res_including_intercept[0, 1:] = res_including_intercept[1:, 0]
        res_including_intercept[1:, 1:] = result
    else:
        res_including_intercept = result
    return res_including_intercept


def _safe_toarray(X):
    """Returns a numpy array."""
    if sparse.issparse(X):
        return X.toarray()
    else:
        return np.asarray(X)
