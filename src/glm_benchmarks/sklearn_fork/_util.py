import numpy as np
from scipy import sparse


def _min_norm_subgrad(
    coef: np.ndarray, grad: np.ndarray, P2: np.ndarray, P1: np.ndarray
) -> np.ndarray:
    """Compute the gradient of all subgradients with minimal L2-norm.

    subgrad = grad + P2 * coef + P1 * subgrad(|coef|_1)

    g_i = grad_i + (P2*coef)_i

    if coef_i > 0:   g_i + P1_i
    if coef_i < 0:   g_i - P1_i
    if coef_i = 0:   sign(g_i) * max(|g_i|-P1_i, 0)

    Parameters
    ----------
    coef : ndarray
        coef[0] may be intercept.

    grad : ndarray, shape=coef.shape

    P2 : {1d or 2d array, None}
        always without intercept, ``None`` means P2 = 0

    P1 : ndarray
        always without intercept
    """
    intercept = coef.size == P1.size + 1
    idx = 1 if intercept else 0  # offset if coef[0] is intercept
    # compute grad + coef @ P2 without intercept
    grad_wP2 = grad[idx:].copy()
    if P2 is None:
        pass
    elif P2.ndim == 1:
        grad_wP2 += coef[idx:] * P2
    else:
        grad_wP2 += coef[idx:] @ P2
    res = np.where(
        coef[idx:] == 0,
        np.sign(grad_wP2) * np.maximum(np.abs(grad_wP2) - P1, 0),
        grad_wP2 + np.sign(coef[idx:]) * P1,
    )
    if intercept:
        return np.concatenate(([grad[0]], res))
    else:
        return res


def _safe_lin_pred(X, coef: np.ndarray, offset: np.ndarray = None) -> np.ndarray:
    """Compute the linear predictor taking care if intercept is present."""
    res = X.dot(coef[1:]) + coef[0] if coef.size == X.shape[1] + 1 else X.dot(coef)
    if offset is not None:
        return res + offset
    return res


def _safe_sandwich_dot(X, d: np.ndarray, intercept=False) -> np.ndarray:
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


def _safe_toarray(X):
    """Returns a numpy array."""
    if sparse.issparse(X):
        return X.toarray()
    else:
        return np.asarray(X)
