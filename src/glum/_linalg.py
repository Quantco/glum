import typing
from typing import Optional, Union

import numpy as np
import tabmat as tm
from scipy import linalg, sparse

from ._utils import safe_toarray


def is_pos_semidef(p: Union[sparse.spmatrix, np.ndarray]) -> Union[bool, np.bool_]:
    """
    Checks for positive semidefiniteness of ``p`` if ``p`` is a matrix, or
    ``diag(p)`` if a vector.

    ``np.linalg.cholesky(P2)`` 'only' asserts positive definiteness; due to
    numerical precision, we allow eigenvalues to be a tiny bit negative.
    """
    # 1d case
    if p.ndim == 1 or p.shape[0] == 1:
        any_negative = (p < 0).max() if sparse.isspmatrix(p) else (p < 0).any()
        return not any_negative

    # 2d case
    # About -6e-7 for 32-bit, -1e-15 for 64-bit
    epsneg = -10 * np.finfo(np.result_type(float, p.dtype)).epsneg

    if sparse.issparse(p):
        # Computing eigenvalues for sparse matrices is inefficient. If the matrix is
        # not huge, convert to dense. Otherwise, calculate 10% of its eigenvalues.
        p = typing.cast(sparse.spmatrix, p)
        if p.shape[0] < 2000:
            eigenvalues = linalg.eigvalsh(p.toarray())
        else:
            n_evals_to_compuate = p.shape[0] // 10 + 1
            sigma = -1000 * epsneg  # start searching near this value
            which = "SA"  # find smallest algebraic eigenvalues first
            eigenvalues = linalg.eigsh(
                p,
                k=n_evals_to_compuate,
                sigma=sigma,
                which=which,
                return_eigenvectors=False,
            )
    else:  # dense
        eigenvalues = linalg.eigvalsh(p)

    return np.all(eigenvalues >= epsneg)


def _safe_lin_pred(
    X: Union[tm.MatrixBase, tm.StandardizedMatrix],
    coef: np.ndarray,
    offset: np.ndarray = None,
) -> np.ndarray:
    """Compute the linear predictor taking care if intercept is present."""
    idx_offset = 0 if X.shape[1] == coef.shape[0] else 1
    nonzero_coefs = np.where(coef[idx_offset:] != 0.0)[0].astype(np.int32)
    res = X.matvec(coef[idx_offset:], cols=nonzero_coefs)

    if idx_offset == 1:
        res += coef[0]
    if offset is not None:
        return res + offset
    return res


def _safe_sandwich_dot(
    X: Union[tm.MatrixBase, tm.StandardizedMatrix],
    d: np.ndarray,
    rows: np.ndarray = None,
    cols: np.ndarray = None,
    intercept=False,
) -> np.ndarray:
    """
    Compute sandwich product ``X.T @ diag(d) @ X``.

    With ``intercept=True``, ``X`` is treated as if a column of 1 were appended
    as first column of ``X``. ``X`` can be sparse; ``d`` must be an ndarray.
    Always returns an ndarray.
    """
    result = X.sandwich(d, rows, cols)
    if isinstance(result, sparse.dia_matrix):
        result = result.toarray()

    if intercept:
        dim = result.shape[0] + 1
        res_including_intercept = np.empty((dim, dim), dtype=X.dtype)
        res_including_intercept[0, 0] = d.sum()
        res_including_intercept[1:, 0] = X.transpose_matvec(d, rows, cols)
        res_including_intercept[0, 1:] = res_including_intercept[1:, 0]
        res_including_intercept[1:, 1:] = result
    else:
        res_including_intercept = result
    return res_including_intercept


def _solve_least_squares_tikhonov(
    X: Union[tm.MatrixBase, tm.StandardizedMatrix],
    y: np.ndarray,
    sample_weight: np.ndarray,
    P2,
    fit_intercept: bool,
    offset: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Solve weighted least squares with optional Tikhonov (L2) penalty.

    Finds ``beta`` minimising

        0.5 * (y - X @ beta).T @ W @ (y - X @ beta)
        + 0.5 * beta.T @ P2 @ beta

    with ``W = diag(sample_weight)``, via the closed-form solution

        beta_hat = inv(X.T @ W @ X + P2) @ X.T @ W @ y

    With ``fit_intercept=True`` the design matrix is implicitly augmented
    with a leading column of ones (unpenalised). When ``offset`` is not
    ``None``, ``y`` is replaced by ``y - offset``.

    See Hastie, Tibshirani & Friedman (2009), *The Elements of Statistical
    Learning*, 2nd ed., Springer, Section 3.4.1, Eq. 3.44.
    """
    y_minus_offset = y if offset is None else (y - offset)
    is_sparse_p2 = sparse.issparse(P2)
    if is_sparse_p2:
        has_l2_penalty = np.any(P2.data != 0)
    else:
        has_l2_penalty = np.any(P2 != 0)

    if not has_l2_penalty:
        # Use direct WLS via sqrt-weighted data (avoids squaring the condition
        # number of the normal equations) for maximum numerical stability.
        sw_sqrt = np.sqrt(sample_weight)
        X_arr = X.toarray()
        A = X_arr * sw_sqrt[:, None]
        b = sw_sqrt * y_minus_offset
        if fit_intercept:
            # Prepend a column of sqrt(w) for the unpenalised intercept.
            A = np.column_stack([sw_sqrt, A])
        return linalg.lstsq(A, b)[0]

    # ------------------------------------------------------------------
    # P2 != 0: the L2 penalty makes the normal equations well-conditioned
    # so the Hessian approach remains efficient and stable.
    # ------------------------------------------------------------------
    weighted_y = sample_weight * y_minus_offset
    is_diag_p2 = (not is_sparse_p2 and getattr(P2, "ndim", 0) == 1) or (
        is_sparse_p2
        and (
            sparse.isspmatrix_dia(P2)
            or (P2.nnz <= P2.shape[0] and np.all(P2.tocoo().row == P2.tocoo().col))
        )
    )

    if fit_intercept:
        hessian = _safe_sandwich_dot(X, sample_weight, intercept=True)
        rhs = np.empty(X.shape[1] + 1, dtype=X.dtype)
        rhs[0] = weighted_y.sum()
        rhs[1:] = X.transpose_matvec(weighted_y)
        if is_diag_p2:
            diag_idx = np.arange(1, hessian.shape[0])
            diag = P2 if not is_sparse_p2 else P2.diagonal()
            hessian[(diag_idx, diag_idx)] += diag
        else:
            hessian[1:, 1:] += safe_toarray(P2)
    else:
        hessian = _safe_sandwich_dot(X, sample_weight)
        rhs = X.transpose_matvec(weighted_y)
        if is_diag_p2:
            diag = P2 if not is_sparse_p2 else P2.diagonal()
            hessian[np.diag_indices_from(hessian)] += diag
        else:
            hessian += safe_toarray(P2)

    # With nonzero L2 penalty this system is often SPD, so we try the faster
    # Cholesky-oriented path (assume_a="pos"); fallback handles failures.
    try:
        coef = linalg.solve(hessian, rhs, assume_a="pos")
    except linalg.LinAlgError:
        # OLS can be singular / rank-deficient (e.g. collinearity). Use
        # minimum-norm least squares solution as a robust closed-form fallback.
        coef = linalg.lstsq(hessian, rhs)[0]
    return coef
