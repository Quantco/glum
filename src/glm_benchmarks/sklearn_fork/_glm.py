"""
Generalized Linear Models with Exponential Dispersion Family
"""

# Author: Christian Lorentzen <lorentzen.ch@googlemail.com>
# some parts and tricks stolen from other sklearn files.
# License: BSD 3 clause

# TODO: Add cross validation support, e.g. GCV?
# TODO: Should GeneralizedLinearRegressor inherit from LinearModel?
#       So far, it does not.
# TODO: Include further classes in class.rst? ExponentialDispersionModel?
#       TweedieDistribution?
# TODO: Negative values in P1 are not allowed so far. They could be used
#       for group lasso.

# Design Decisions:
# - Which name? GeneralizedLinearModel vs GeneralizedLinearRegressor.
#   Estimators in sklearn are either regressors or classifiers. A GLM can do
#   both depending on the distr (Normal => regressor, Binomial => classifier).
#   Solution: GeneralizedLinearRegressor since this is the focus.
# - Allow for finer control of penalty terms:
#   L1: ||P1*w||_1 with P1*w as element-wise product, this allows to exclude
#       factors from the L1 penalty.
#   L2: w*P2*w with P2 a positive (semi-) definite matrix, e.g. P2 could be
#   a 1st or 2nd order difference matrix (compare B-spline penalties and
#   Tikhonov regularization).
# - The link function (instance of class Link) is necessary for the evaluation
#   of deviance, score, Fisher and Hessian matrix as functions of the
#   coefficients, which is needed by optimizers.
#   Solution: link as argument in those functions
# - Which name/symbol for sample_weight in docu?
#   sklearn.linear_models uses w for coefficients, standard literature on
#   GLMs use beta for coefficients and w for (sample) weights.
#   So far, coefficients=w and sample weights=s.
# - The intercept term is the first index, i.e. coef[0]


from __future__ import division

import time
import warnings
from typing import Any, Iterable, List, Tuple, Union

import numpy as np
import scipy.sparse.linalg as splinalg
from scipy import linalg, sparse
from scipy.optimize import fmin_l_bfgs_b
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils import check_array, check_X_y
from sklearn.utils.validation import check_is_fitted, check_random_state

from glm_benchmarks.scaled_spmat.mkl_sparse_matrix import MKLSparseMatrix

from ._distribution import (
    BinomialDistribution,
    ExponentialDispersionModel,
    GammaDistribution,
    GeneralizedHyperbolicSecant,
    InverseGaussianDistribution,
    NormalDistribution,
    PoissonDistribution,
    TweedieDistribution,
    guess_intercept,
)
from ._link import IdentityLink, Link, LogitLink, LogLink
from ._util import _safe_lin_pred, _safe_sandwich_dot
from .dense_glm_matrix import DenseGLMDataMatrix

_float_itemsize_to_dtype = {8: np.float64, 4: np.float32, 2: np.float16}


def _to_precision(arr: np.ndarray, itemsize: int) -> np.ndarray:
    """
    >>> ones = np.ones(2)
    >>> ones.dtype
    dtype('float64')
    >>> _to_precision(ones, itemsize=8).dtype
    dtype('float64')
    >>> _to_precision(ones, itemsize=np.dtype(np.int32).itemsize).dtype
    dtype('float32')

    Useful for getting floats and ints to have the same precision.
    """
    size = arr.dtype.itemsize
    if size == itemsize:
        return arr
    if np.issubdtype(arr.dtype, np.signedinteger):
        target_dtype = {8: np.int64, 4: np.int32, 2: np.int16}
    if np.issubdtype(arr.dtype, np.floating):
        target_dtype = _float_itemsize_to_dtype
    return arr.astype(target_dtype[itemsize])


def get_float_dtype_of_size(itemsize: int):
    return _float_itemsize_to_dtype[itemsize]


def _check_weights(
    sample_weight: Union[float, np.ndarray, None], n_samples: int, dtype
):
    """Check that sample weights are non-negative and have the right shape."""
    if sample_weight is None:
        weights = np.ones(n_samples, dtype=dtype)
    elif np.isscalar(sample_weight):
        if sample_weight <= 0:
            raise ValueError("Sample weights must be non-negative.")
        weights = sample_weight * np.ones(n_samples, dtype=dtype)
    else:
        _dtype = [np.float64, np.float32]
        weights = check_array(
            sample_weight,
            accept_sparse=False,
            force_all_finite=True,
            ensure_2d=False,
            dtype=_dtype,
        )
        if weights.ndim > 1:
            raise ValueError("Sample weight must be 1D array or scalar")
        elif weights.shape[0] != n_samples:
            raise ValueError("Sample weights must have the same length as y")
        if not np.all(weights >= 0):
            raise ValueError("Sample weights must be non-negative.")
        elif not np.sum(weights) > 0:
            raise ValueError(
                "Sample weights must have at least one positive " "element."
            )

    return weights


def _check_offset(
    offset: Union[np.ndarray, float, None], n_rows: int, dtype
) -> np.ndarray:
    """
    Unlike weights, if the offset is given as None, it can stay None. So we only need
    to validate it when it is not none.
    """
    if offset is None:
        return None
    if not np.isscalar(offset):
        offset = check_array(
            offset,
            accept_sparse=False,
            force_all_finite=True,
            ensure_2d=False,
            dtype=dtype,
        )
        if offset.ndim > 1:
            raise ValueError("Offset must be 1D array or scalar.")
        elif offset.shape[0] != n_rows:
            raise ValueError("offset must have the same length as y.")
    return np.full(n_rows, offset)


def _safe_toarray(X):
    """Returns a numpy array."""
    if sparse.issparse(X):
        return X.toarray()
    else:
        return np.asarray(X)


def _min_norm_sugrad(
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


def _unstandardize(
    X, col_means: np.ndarray, col_stds: np.ndarray, intercept: float, coef
) -> Tuple[Any, float, np.ndarray]:
    X = X.unstandardize(col_means, col_stds)
    intercept -= float(np.squeeze(col_means / col_stds).dot(coef))
    coef /= col_stds
    return X, intercept, coef


def _standardize_warm_start(coef, col_means, col_stds):
    coef[1:] *= col_stds
    coef[0] += np.squeeze(col_means / col_stds).dot(coef[1:])


def _irls_step(X, W: np.ndarray, P2, z: np.ndarray, fit_intercept=True):
    """Compute one step in iteratively reweighted least squares.

    Solve A w = b for w with
    A = (X' W X + P2)
    b = X' W z
    z = eta + D^-1 (y-mu)

    See also fit method of :class:`GeneralizedLinearRegressor`.

    Parameters
    ----------
    X : {ndarray, sparse matrix}, shape (n_samples, n_features)
        Training data (with intercept included if present)

    W : ndarray, shape (n_samples,)

    P2 : {ndarray, sparse matrix}, shape (n_features, n_features)
        The L2-penalty matrix or vector (=diagonal matrix)

    z : ndarray, shape (n_samples,)
        Working observations

    fit_intercept : boolean, optional (default=True)

    Returns
    -------
    coef : ndarray, shape (c,)
        If fit_intercept=False, shape c=X.shape[1].
        If fit_intercept=True, then c=X.shapee[1] + 1.
    """
    # Note: solve vs least squares, what is more appropriate?
    #       scipy.linalg.solve seems faster, but scipy.linalg.lstsq
    #       is more robust.
    # Note: X.T @ W @ X is not sparse, even when X is sparse.
    #      Sparse solver would splinalg.spsolve(A, b) or splinalg.lsmr(A, b)
    assert np.all(np.isfinite(W))
    if fit_intercept:
        Wz = W * z
        if sparse.issparse(X):
            b = np.concatenate(([Wz.sum()], X.transpose() @ Wz))
        else:
            b = np.concatenate(([Wz.sum()], X.T @ Wz))
        A = _safe_sandwich_dot(X, W, intercept=fit_intercept)
        if P2.ndim == 1:
            idx = np.arange(start=1, stop=A.shape[0])
            A[(idx, idx)] += P2  # add to diag elements without intercept
        elif sparse.issparse(P2):
            A[1:, 1:] += P2.toarray()
        else:
            A[1:, 1:] += P2
    else:
        if sparse.issparse(X):
            XtW = X.transpose().multiply(W)
            # for older versions of numpy and scipy, A may be a np.matrix
            A = _safe_toarray(XtW @ X)
        else:
            XtW = X.T * W
            A = XtW @ X
        b = XtW @ z
        if P2.ndim == 1:
            A[np.diag_indices_from(A)] += P2
        elif sparse.issparse(P2):
            A += P2.toarray()
        else:
            A += P2

    coef, _, rank, sing_vals = linalg.lstsq(A, b, overwrite_a=True, overwrite_b=True)

    expected_rank = A.shape[1]
    if rank < expected_rank:  # rank deficient
        # may have been rank deficient due to lack of precision
        if X.dtype == np.float32:
            warning = f"""
                A matrix used for IRLS is poorly conditioned or rank deficient;
                it has measued rank {rank} when rank {expected_rank} is required, and
                condition_number {sing_vals[0] / sing_vals[1]}. Numerical failures are
                likely. To avoid this problem, try using double precision
                (X.dtype = np.float64) or increasing regularization.
            """
        else:  # unknown cause of rank deficiency
            warning = f"""
                A matrix used for IRLS is poorly conditioned or rank deficient;
                it has measued rank {rank} when rank {expected_rank} is required, and
                condition_number {sing_vals[0] / sing_vals[1]}. Numerical failures are
                likely. Try increasing regularization.
             """
        warnings.warn(warning)

    return coef


def _irls_solver(
    coef,
    X,
    y: np.ndarray,
    weights: np.ndarray,
    P2: Union[np.ndarray, sparse.spmatrix],
    fit_intercept: bool,
    family: ExponentialDispersionModel,
    link: Link,
    max_iter: int,
    tol: float,
    offset: np.ndarray = None,
):
    """Solve GLM with L2 penalty by IRLS algorithm.

    Note: If X is sparse, P2 must also be sparse.
    """
    # Solve Newton-Raphson (1): Obj'' (w - w_old) = -Obj'
    #   Obj = objective function = 1/2 Dev + l2/2 w P2 w
    #   Dev = deviance, s = normalized weights, variance V(mu) but phi=1
    #   D   = link.inverse_derivative(eta) = diag_matrix(h'(X w))
    #   D2  = link.inverse_derivative(eta)^2 = D^2
    #   W   = D2/V(mu)
    #   l2  = alpha * (1 - l1_ratio)
    #   Obj' = d(Obj)/d(w) = 1/2 Dev' + l2 P2 w
    #        = -X' D (y-mu)/V(mu) + l2 P2 w
    #   Obj''= d2(Obj)/d(w)d(w') = Hessian = -X'(...) X + l2 P2
    #   Use Fisher matrix instead of full info matrix -X'(...) X,
    #    i.e. E[Dev''] with E[y-mu]=0:
    #   Obj'' ~ X' W X + l2 P2
    # (1): w = (X' W X + l2 P2)^-1 X' W z,
    #      with z = eta + D^-1 (y-mu)
    # Note: P2 must be symmetrized
    # Note: ' denotes derivative, but also transpose for matrices

    eta_no_offset = _safe_lin_pred(X, coef)
    eta = eta_no_offset if offset is None else eta_no_offset + offset
    mu = link.inverse(eta)
    # D = h'(eta)
    hp = link.inverse_derivative(eta)
    V = family.variance(mu, phi=1, weights=weights)

    converged = False
    n_iter = 0
    while n_iter < max_iter:
        n_iter += 1
        # coef_old not used so far.
        # coef_old = coef
        # working weights W, in principle a diagonal matrix
        # therefore here just as 1d array
        W = hp ** 2 / V
        # working observations
        # float32 - int32 = float64, unless you specify dtype
        z = (
            eta_no_offset
            + np.subtract(y, mu, dtype=get_float_dtype_of_size(X.dtype.itemsize)) / hp
        )
        # solve A*coef = b
        # A = X' W X + P2, b = X' W z
        coef = _irls_step(X, W, P2, z, fit_intercept=fit_intercept)
        # updated linear predictor
        # do it here for updated values for tolerance
        eta_no_offset = _safe_lin_pred(X, coef)
        eta = eta_no_offset if offset is None else eta_no_offset + offset
        mu = link.inverse(eta)
        hp = link.inverse_derivative(eta)
        V = family.variance(mu, phi=1, weights=weights)

        # which tolerance? |coef - coef_old| or gradient?
        # use gradient for compliance with lbfgs
        # gradient = -X' D (y-mu)/V(mu) + l2 P2 w
        temp = hp * (y - mu) / V
        if sparse.issparse(X):
            gradient = -(X.transpose() @ temp)
        else:
            gradient = -(X.T @ temp)
        idx = 1 if fit_intercept else 0  # offset if coef[0] is intercept
        if P2.ndim == 1:
            gradient += P2 * coef[idx:]
        else:
            gradient += P2 @ coef[idx:]
        if fit_intercept:
            gradient = np.concatenate(([-temp.sum()], gradient))
        if np.max(np.abs(gradient)) <= tol:
            converged = True
            break

    if not converged:
        warnings.warn(
            "irls failed to converge. Increase the number "
            "of iterations (currently {})".format(max_iter),
            ConvergenceWarning,
        )

    return coef, n_iter


def _cd_cycle(
    d: np.ndarray,
    X,
    coef: np.ndarray,
    score,
    fisher,
    P1,
    P2,
    n_cycles: int,
    inner_tol: float,
    max_inner_iter=1000,
    selection="cyclic",
    random_state=None,
    diag_fisher=False,
):
    """Compute inner loop of coordinate descent, i.e. cycles through features.

    Minimization of 1-d subproblems::

        min_z q(d+z*e_j) - q(d)
        = min_z A_j z + 1/2 B_jj z^2 + ||P1_j (w_j+d_j+z)||_1

    A = f'(w) + d*H(w) + (w+d)*P2
    B = H+P2
    Note: f'=-score and H=fisher are updated at the end of outer iteration.
    """
    # TODO: split into diag_fisher and non diag_fisher cases to make optimization easier
    # TODO: Cython/C++?
    # TODO: use sparsity (coefficient already 0 due to L1 penalty)
    #       => active set of features for featurelist, see paper
    #          of Improved GLMNET or Gap Safe Screening Rules
    #          https://arxiv.org/abs/1611.05780
    n_samples, n_features = X.shape
    intercept = coef.size == X.shape[1] + 1
    idx = 1 if intercept else 0  # offset if coef[0] is intercept
    f_cont = fisher.flags["F_CONTIGUOUS"]

    if P2.ndim == 1:
        coef_P2 = coef[idx:] * P2
        if not diag_fisher:
            idiag = np.arange(start=idx, stop=fisher.shape[0])
            # B[np.diag_indices_from(B)] += P2
            fisher[(idiag, idiag)] += P2
    else:
        coef_P2 = coef[idx:] @ P2
        if not diag_fisher:
            if sparse.issparse(P2):
                fisher[idx:, idx:] += P2.toarray()
            else:
                fisher[idx:, idx:] += P2
    A = -score
    A[idx:] += coef_P2
    # A += d @ (H+P2) but so far d=0
    # inner loop
    for inner_iter in range(1, max_inner_iter + 1):
        inner_iter += 1
        n_cycles += 1
        # cycle through features, update intercept separately at the end
        # TODO: move a lot of this to outer loop
        if selection == "random":
            featurelist = random_state.permutation(n_features)
        else:
            featurelist = np.arange(n_features)

        # if selection is not random, only need to do this once
        jdx_vec = featurelist + idx
        if diag_fisher:
            b_vec = fisher[jdx_vec]
        else:
            b_vec = np.diag(fisher)[jdx_vec]
        b_less_than_zero_vec = b_vec <= 0
        p1_is_zero_vec = P1[featurelist] == 0

        for num, j in enumerate(featurelist):
            # minimize_z: a z + 1/2 b z^2 + c |d+z|
            # a = A_j
            # b = B_jj > 0
            # c = |P1_j| = P1_j > 0, see 1.3
            # d = w_j + d_j
            # cf. https://arxiv.org/abs/0708.1485 Eqs. (3) - (4)
            # with beta = z+d, beta_hat = d-a/b and gamma = c/b
            # z = 1/b * S(bd-a,c) - d
            # S(a,b) = sign(a) max(|a|-b, 0) soft thresholding
            # jdx = j + idx  # index for arrays containing entries for intercept
            jdx = jdx_vec[num]
            a = A[jdx]
            if diag_fisher:
                # Note: fisher is ndarray of shape (n_samples,) => no idx
                # Calculate Bj = B[j, :] = B[:, j] as it is needed later anyway
                Bj = np.zeros_like(A)
                if intercept:
                    Bj[0] = fisher.sum()

                x_j = np.squeeze(np.array(X.getcol(j).toarray()))
                Bj[idx:] = _safe_toarray((fisher * x_j) @ X).ravel()

                if P2.ndim == 1:
                    Bj[idx:] += P2[j]
                else:
                    if sparse.issparse(P2):
                        # slice columns as P2 is csc
                        Bj[idx:] += P2[:, j].toarray().ravel()
                    else:
                        Bj[idx:] += P2[:, j]
                b = Bj[jdx]
            else:
                b = b_vec[num]

            # those ten lines are what it is all about
            if b_less_than_zero_vec[num]:
                z = 0
            elif p1_is_zero_vec[num]:
                z = -a / b
            elif a + P1[j] < b * (coef[jdx] + d[jdx]):
                z = -(a + P1[j]) / b
            elif a - P1[j] > b * (coef[jdx] + d[jdx]):
                z = -(a - P1[j]) / b
            else:
                z = -(coef[jdx] + d[jdx])

            # update direction d
            d[jdx] += z
            # update A because d_j is now d_j+z
            # A = f'(w) + d*H(w) + (w+d)*P2
            # => A += (H+P2)*e_j z = B_j * z
            # Note: B is symmetric B = B.transpose
            if diag_fisher:
                # Bj = B[:, j] calculated above, still valid
                bj = Bj
            # otherwise, B is symmetric, C- or F-contiguous, but never sparse
            elif f_cont:
                # slice columns like for sparse csc
                bj = fisher[:, jdx]
            else:  # B.flags['C_CONTIGUOUS'] might be true
                # slice rows
                bj = fisher[jdx, :]
            A += bj * z

            # end of cycle over features
        # update intercept
        if intercept:
            if diag_fisher:
                Bj = np.zeros_like(A)
                Bj[0] = fisher.sum()
                Bj[1:] = fisher @ X
                b = Bj[0]
            else:
                b = fisher[0, 0]
            z = 0 if b <= 0 else -A[0] / b
            d[0] += z
            if diag_fisher:
                A += Bj * z
            else:
                if fisher.flags["F_CONTIGUOUS"]:
                    A += fisher[:, 0] * z
                else:
                    A += fisher[0, :] * z
        # end of complete cycle
        # stopping criterion for inner loop
        # sum_i(|minimum of norm of subgrad of q(d)_i|)
        # subgrad q(d) = A + subgrad ||P1*(w+d)||_1
        mn_subgrad = _min_norm_sugrad(coef=coef + d, grad=A, P2=None, P1=P1)
        mn_subgrad = linalg.norm(mn_subgrad, ord=1)
        if mn_subgrad <= inner_tol:
            if inner_iter == 1:
                inner_tol = inner_tol / 4.0
            break
        # end of inner loop
    return d, coef_P2, n_cycles, inner_tol


def _cd_solver(
    coef,
    X,
    y: np.ndarray,
    weights: np.ndarray,
    P1: Union[np.ndarray, sparse.spmatrix],
    P2: Union[np.ndarray, sparse.spmatrix],
    fit_intercept: bool,
    family: ExponentialDispersionModel,
    link: Link,
    max_iter: int = 100,
    max_inner_iter: int = 1000,
    tol: float = 1e-4,
    selection="cyclic ",
    random_state=None,
    diag_fisher=False,
    offset: np.ndarray = None,
) -> Tuple[np.ndarray, int, int, List[List]]:
    """Solve GLM with L1 and L2 penalty by coordinate descent algorithm.

    The objective being minimized in the coefficients w=coef is::

        F = f + g, f(w) = 1/2 deviance, g = 1/2 w*P2*w + ||P1*w||_1

    An Improved GLMNET for L1-regularized Logistic Regression:

    1. Find optimal descent direction d by minimizing
       min_d F(w+d) = min_d F(w+d) - F(w)
    2. Quadratic approximation of F(w+d)-F(w) = q(d):
       using f(w+d) = f(w) + f'(w)*d + 1/2 d*H(w)*d + O(d^3) gives:
       q(d) = (f'(w) + w*P2)*d + 1/2 d*(H(w)+P2)*d
       + ||P1*(w+d)||_1 - ||P1*w||_1
       Then minimize q(d): min_d q(d)
    3. Coordinate descent by updating coordinate j (d -> d+z*e_j):
       min_z q(d+z*e_j)
       = min_z q(d+z*e_j) - q(d)
       = min_z A_j z + 1/2 B_jj z^2
               + ||P1_j (w_j+d_j+z)||_1 - ||P1_j (w_j+d_j)||_1
       A = f'(w) + d*H(w) + (w+d)*P2
       B = H + P2

    Repeat steps 1-3 until convergence.
    Note: Use Fisher matrix instead of Hessian for H.
    Note: f' = -score, H = Fisher matrix

    Parameters
    ----------
    coef : ndarray, shape (c,)
        If fit_intercept=False, shape c=X.shape[1].
        If fit_intercept=True, then c=X.shape[1] + 1.

    X : {ndarray, csc sparse matrix}, shape (n_samples, n_features)
        Training data (with intercept included if present). If not sparse,
        pass directly as Fortran-contiguous data to avoid
        unnecessary memory duplication.

    y : ndarray, shape (n_samples,)
        Target values.

    weights: ndarray, shape (n_samples,)
        Sample weights with which the deviance is weighted. The weights must
        bee normalized and sum to 1.

    P1 : {ndarray}, shape (n_features,)
        The L1-penalty vector (=diagonal matrix)

    P2 : {ndarray, csc sparse matrix}, shape (n_features, n_features)
        The L2-penalty matrix or vector (=diagonal matrix). If a matrix is
        passed, it must be symmetric. If X is sparse, P2 must also be sparse.

    fit_intercept : boolean, optional (default=True)
        Specifies if a constant (a.k.a. bias or intercept) should be
        added to the linear predictor (X*coef+intercept).

    family : ExponentialDispersionModel

    link : Link

    max_iter : int, optional (default=100)
        Maximum numer of outer (Newton) iterations.

    max_inner_iter : int, optional (default=1000)
        Maximum number of iterations in each inner loop, i.e. max number of
        cycles over all features per inner loop.

    tol : float, optional (default=1e-4)
        Convergence criterion is
        sum_i(|minimum of norm of subgrad of objective_i|)<=tol.

    selection : str, optional (default='cyclic')
        If 'random', randomly chose features in inner loop.

    random_state : {int, RandomState instance, None}, optional (default=None)

    diag_fisher : boolean, optional (default=False)
        ``False`` calculates full fisher matrix, ``True`` only diagonal matrix
        s.t. fisher = X.T @ diag @ X. This saves storage but needs more
        matrix-vector multiplications.


    Returns
    -------
    coef : ndarray, shape (c,)
        If fit_intercept=False, shape c=X.shape[1].
        If fit_intercept=True, then c=X.shape[1] + 1.

    n_iter : number of outer iterations = newton iterations

    n_cycles : number of cycles over features

    References
    ----------
    Guo-Xun Yuan, Chia-Hua Ho, Chih-Jen Lin
    An Improved GLMNET for L1-regularized Logistic Regression,
    Journal of Machine Learning Research 13 (2012) 1999-2030
    https://www.csie.ntu.edu.tw/~cjlin/papers/l1_glmnet/long-glmnet.pdf
    """
    if P2.ndim == 2:
        P2 = check_array(P2, "csc", dtype=[np.float64, np.float32])

    if sparse.issparse(X):
        if not sparse.isspmatrix_csc(P2):
            raise ValueError(
                "If X is sparse, P2 must also be sparse csc"
                "format. Got P2 not sparse."
            )

    random_state = check_random_state(random_state)
    # Note: we already set P2 = l2*P2, P1 = l1*P1
    # Note: we already symmetrized P2 = 1/2 (P2 + P2')
    n_iter = 0  # number of outer iterations
    n_cycles = 0  # number of (complete) cycles over features
    converged = False
    idx = 1 if fit_intercept else 0  # offset if coef[0] is intercept
    # line search parameters
    (beta, sigma) = (0.5, 0.01)
    # some precalculations
    # Note: For diag_fisher=False, fisher = X.T @ fisher @ X and fisher is a
    #       1d array representing a diagonal matrix.
    iteration_start = time.time()
    eta, mu, score, fisher = family._eta_mu_score_fisher(
        coef=coef,
        phi=1,
        X=X,
        y=y,
        weights=weights,
        link=link,
        diag_fisher=diag_fisher,
        offset=offset,
    )
    # set up space for search direction d for inner loop
    d = np.zeros_like(coef)

    # minimum subgradient norm
    def calc_mn_subgrad_norm():
        return linalg.norm(
            _min_norm_sugrad(coef=coef, grad=-score, P2=P2, P1=P1), ord=1
        )

    # the ratio of inner _cd_cycle tolerance to the minimum subgradient norm
    # This wasn't explored in the newGLMNET paper linked above.
    # That paper essentially uses inner_tol_ratio = 1.0, but using a slightly
    # lower value is much faster.
    # By comparison, the original GLMNET paper uses inner_tol = tol.
    # So, inner_tol_ratio < 1 is sort of a compromise between the two papers.
    # The value should probably be between 0.01 and 0.5. 0.1 works well for many problems
    inner_tol_ratio = 0.1

    def calc_inner_tol(mn_subgrad_norm):
        # Another potential rule limits the inner tol to be no smaller than tol
        # return max(mn_subgrad_norm * inner_tol_ratio, tol)
        return mn_subgrad_norm * inner_tol_ratio

    # initial stopping tolerance of inner loop
    # use L1-norm of minimum of norm of subgradient of F
    inner_tol = calc_inner_tol(calc_mn_subgrad_norm())

    Fw = None

    diagnostics = []
    # outer loop
    while n_iter < max_iter:
        n_iter += 1
        # initialize search direction d (to be optimized) with zero
        d.fill(0)
        # inner loop = _cd_cycle
        d, coef_P2, n_cycles, inner_tol = _cd_cycle(
            d,
            X,
            coef,
            score,
            fisher,
            P1,
            P2,
            n_cycles,
            inner_tol,
            max_inner_iter=max_inner_iter,
            selection=selection,
            random_state=random_state,
            diag_fisher=diag_fisher,
        )

        # line search by sequence beta^k, k=0, 1, ..
        # F(w + lambda d) - F(w) <= lambda * bound
        # bound = sigma * (f'(w)*d + w*P2*d
        #                  +||P1 (w+d)||_1 - ||P1 w||_1)
        P1w_1 = linalg.norm(P1 * coef[idx:], ord=1)
        P1wd_1 = linalg.norm(P1 * (coef + d)[idx:], ord=1)
        # Note: coef_P2 already calculated and still valid
        bound = sigma * (-(score @ d) + coef_P2 @ d[idx:] + P1wd_1 - P1w_1)

        # In the first iteration, we must compute Fw explicitly.
        # In later iterations, we just use Fwd from the previous iteration
        # as set after the line search loop below.
        if Fw is None:
            Fw = (
                0.5 * family.deviance(y, mu, weights)
                + 0.5 * (coef_P2 @ coef[idx:])
                + P1w_1
            )

        la = 1.0 / beta

        # TODO: if we keep track of X_dot_coef, we can add this to avoid a
        # _safe_lin_pred in _eta_mu_score_fisher every loop
        X_dot_d = _safe_lin_pred(X, d)

        # Try progressively shorter line search steps.
        for k in range(20):
            la *= beta  # starts with la=1
            coef_wd = coef + la * d

            # The simple version of the next line is:
            # mu_wd = link.inverse(_safe_lin_pred(X, coef_wd))
            # but because coef_wd can be factored as
            # coef_wd = coef + la * d
            # we can rewrite to only perform one dot product with the data
            # matrix per loop which is substantially faster
            mu_wd = link.inverse(eta + la * X_dot_d)

            # TODO - optimize: for Tweedie that isn't one of the special cases
            # (gaussian, poisson, gamma), family.deviance is quite slow! Can we
            # fix that somehow?
            Fwd = 0.5 * family.deviance(y, mu_wd, weights) + linalg.norm(
                P1 * coef_wd[idx:], ord=1
            )
            if P2.ndim == 1:
                Fwd += 0.5 * ((coef_wd[idx:] * P2) @ coef_wd[idx:])
            else:
                Fwd += 0.5 * (coef_wd[idx:] @ (P2 @ coef_wd[idx:]))
            if Fwd - Fw <= sigma * la * bound:
                break

        # Fw in the next iteration will be equal to Fwd this iteration.
        Fw = Fwd

        # update coefficients
        coef += la * d

        iteration_runtime = time.time() - iteration_start
        diagnostics.append([inner_tol, n_iter, n_cycles, iteration_runtime, coef[0]])
        iteration_start = time.time()

        # calculate eta, mu, score, Fisher matrix for next iteration
        eta, mu, score, fisher = family._eta_mu_score_fisher(
            coef=coef,
            phi=1,
            X=X,
            y=y,
            weights=weights,
            link=link,
            diag_fisher=diag_fisher,
            offset=offset,
        )

        # stopping criterion for outer loop
        # sum_i(|minimum-norm of subgrad of F(w)_i|)
        # fp_wP2 = f'(w) + w*P2
        # Note: eta, mu and score are already updated
        # this also updates the inner tolerance for the next loop!
        mn_subgrad_norm = calc_mn_subgrad_norm()
        if mn_subgrad_norm <= tol:
            converged = True
            break

        inner_tol = calc_inner_tol(mn_subgrad_norm)
        # end of outer loop

    if not converged:
        warnings.warn(
            "Coordinate descent failed to converge. Increase"
            " the maximum number of iterations max_iter"
            " (currently {})".format(max_iter),
            ConvergenceWarning,
        )
    return coef, n_iter, n_cycles, diagnostics


def get_family(
    family: Union[str, ExponentialDispersionModel]
) -> ExponentialDispersionModel:
    if isinstance(family, ExponentialDispersionModel):
        return family
    name_to_dist = {
        "normal": NormalDistribution,
        "poisson": PoissonDistribution,
        "gamma": GammaDistribution,
        "inverse.gaussian": InverseGaussianDistribution,
        "binomial": BinomialDistribution,
    }
    try:
        return name_to_dist[family]()
    except KeyError:
        raise ValueError(
            "The family must be an instance of class"
            " ExponentialDispersionModel or an element of"
            " ['normal', 'poisson', 'gamma', 'inverse.gaussian', "
            "'binomial']; got (family={})".format(family)
        )


def get_link(link: Union[str, Link], family: ExponentialDispersionModel) -> Link:
    """
    For the Tweedie distribution, this code follows actuarial best practices regarding
    link functions. Note that these links are sometimes non-canonical:
        - Identity for normal (p=0)
        - No convention for p < 0, so let's leave it as identity
        - Log otherwise
    """
    if isinstance(link, Link):
        return link
    if link == "auto":
        if isinstance(family, TweedieDistribution):
            # This code
            if family.power <= 0:
                return IdentityLink()
            if family.power < 1:
                # TODO: move more detailed error here
                raise ValueError("No distribution")
            return LogLink()
        if isinstance(family, GeneralizedHyperbolicSecant):
            return IdentityLink()
        if isinstance(family, BinomialDistribution):
            return LogitLink()
        raise ValueError(
            """No default link known for the specified distribution family. Please
            set link manually, i.e. not to 'auto';
            got (link='auto', family={})""".format(
                family.__class__.__name__
            )
        )
    if link == "identity":
        return IdentityLink()
    if link == "log":
        return LogLink()
    if link == "logit":
        return LogitLink()
    raise ValueError(
        """The link must be an instance of class Link or an element of
        ['auto', 'identity', 'log', 'logit']; got (link={})""".format(
            link
        )
    )


def setup_p1(
    P1: Union[str, np.ndarray],
    X: Union[np.ndarray, sparse.spmatrix],
    _dtype,
    alpha: float,
    l1_ratio: float,
) -> np.ndarray:
    n_features = X.shape[1]
    if isinstance(P1, str) and P1 == "identity":
        P1 = np.ones(n_features, dtype=_dtype)
    else:
        P1 = np.atleast_1d(P1)
        try:
            P1 = P1.astype(_dtype, casting="safe", copy=False)
        except TypeError:
            raise TypeError(
                "The given P1 cannot be converted to a numeric"
                "array; got (P1.dtype={}).".format(P1.dtype)
            )
        if (P1.ndim != 1) or (P1.shape[0] != n_features):
            raise ValueError(
                "P1 must be either 'identity' or a 1d array "
                "with the length of X.shape[1]; "
                "got (P1.shape[0]={}), "
                "needed (X.shape[1]={}).".format(P1.shape[0], n_features)
            )

    # P1 and P2 are now for sure copies
    P1 = alpha * l1_ratio * P1
    return P1


def setup_p2(
    P2: Union[str, np.ndarray],
    X: Union[np.ndarray, sparse.spmatrix],
    _stype,
    _dtype,
    alpha: float,
    l1_ratio: float,
) -> Union[np.ndarray, sparse.spmatrix]:
    n_features = X.shape[1]

    # If X is sparse, make P2 sparse, too.
    if isinstance(P2, str) and P2 == "identity":
        if sparse.issparse(X):
            P2 = (
                sparse.dia_matrix(
                    (np.ones(n_features, dtype=_dtype), 0),
                    shape=(n_features, n_features),
                )
            ).tocsc()
        else:
            P2 = np.ones(n_features, dtype=_dtype)
    else:
        P2 = check_array(
            P2, copy=True, accept_sparse=_stype, dtype=_dtype, ensure_2d=False
        )
        if P2.ndim == 1:
            P2 = np.asarray(P2)
            if P2.shape[0] != n_features:
                raise ValueError(
                    "P2 should be a 1d array of shape "
                    "(n_features,) with "
                    "n_features=X.shape[1]; "
                    "got (P2.shape=({},)), needed ({},)".format(P2.shape[0], X.shape[1])
                )
            if sparse.issparse(X):
                P2 = (
                    sparse.dia_matrix((P2, 0), shape=(n_features, n_features))
                ).tocsc()
        elif P2.ndim == 2 and P2.shape[0] == P2.shape[1] and P2.shape[0] == X.shape[1]:
            if sparse.issparse(X):
                P2 = sparse.csc_matrix(P2)
        else:
            raise ValueError(
                "P2 must be either None or an array of shape "
                "(n_features, n_features) with "
                "n_features=X.shape[1]; "
                "got (P2.shape=({0}, {1})), needed ({2}, {2})".format(
                    P2.shape[0], P2.shape[1], X.shape[1]
                )
            )

    # P1 and P2 are now for sure copies
    P2 = alpha * (1 - l1_ratio) * P2
    # one only ever needs the symmetrized L2 penalty matrix 1/2 (P2 + P2')
    # reason: w' P2 w = (w' P2 w)', i.e. it is symmetric
    if P2.ndim == 2:
        if sparse.issparse(P2):
            if sparse.isspmatrix_csc(P2):
                P2 = 0.5 * (P2 + P2.transpose()).tocsc()
            else:
                P2 = 0.5 * (P2 + P2.transpose()).tocsr()
        else:
            P2 = 0.5 * (P2 + P2.T)
    return P2


def initialize_start_params(
    start_params: Union[str, np.ndarray], n_cols: int, fit_intercept: bool, _dtype
) -> np.ndarray:
    if isinstance(start_params, str):
        if start_params not in ["guess", "zero"]:
            raise ValueError(
                "The argument start_params must be 'guess', "
                "'zero' or an array of correct length; "
                "got(start_params={})".format(start_params)
            )
    else:
        start_params = check_array(
            start_params,
            accept_sparse=False,
            force_all_finite=True,
            ensure_2d=False,
            dtype=_dtype,
            copy=True,
        )
        if (start_params.shape[0] != n_cols + fit_intercept) or (
            start_params.ndim != 1
        ):
            raise ValueError(
                "Start values for parameters must have the"
                "right length and dimension; required (length"
                "={}, ndim=1); got (length={}, ndim={}).".format(
                    n_cols + fit_intercept, start_params.shape[0], start_params.ndim,
                )
            )
    return start_params


def is_pos_semidef(p: Union[np.ndarray, sparse.spmatrix]) -> bool:
    """
    Checks for positive semidefiniteness of p if p is a matrix, or diag(p) if p is a
    vector.

    np.linalg.cholesky(P2) 'only' asserts positive definite due to numerical precision,
    we allow eigenvalues to be a tiny bit negative
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
        if p.shape[0] < 2000:
            eigenvalues = linalg.eigvalsh(p.toarray())
        else:
            n_evals_to_compuate = p.shape[0] // 10 + 1
            sigma = -1000 * epsneg  # start searching near this value
            which = "SA"  # find smallest algebraic eigenvalues first
            eigenvalues = splinalg.eigsh(
                p,
                k=n_evals_to_compuate,
                sigma=sigma,
                which=which,
                return_eigenvectors=False,
            )
    else:
        # dense
        eigenvalues = linalg.eigvalsh(p)
    pos_semidef = np.all(eigenvalues >= epsneg)
    return pos_semidef


# TODO: abc
class GeneralizedLinearRegressorBase(BaseEstimator, RegressorMixin):
    """
    Base class for GeneralizedLinearRegressor and GeneralizedLinearRegressorCV.
    """

    def __init__(
        self,
        l1_ratio: Union[int, float] = 0,
        P1="identity",
        P2: Union[np.ndarray, Iterable, int, float] = "identity",
        fit_intercept=True,
        family: Union[str, ExponentialDispersionModel] = "normal",
        link: Union[str, Link] = "auto",
        fit_dispersion=None,
        solver="auto",
        max_iter=100,
        tol=1e-4,
        warm_start=False,
        start_params="guess",
        selection="cyclic",
        random_state=None,
        diag_fisher=False,
        copy_X=True,
        check_input=True,
        verbose=0,
        scale_predictors=False,
    ):
        self.l1_ratio = l1_ratio
        self.P1 = P1
        self.P2 = P2
        self.fit_intercept = fit_intercept
        self.family = family
        self.link = link
        self.fit_dispersion = fit_dispersion
        self.solver = solver
        self.max_iter = max_iter
        self.tol = tol
        self.warm_start = warm_start
        self.start_params = start_params
        self.selection = selection
        self.random_state = random_state
        self.diag_fisher = diag_fisher
        self.copy_X = copy_X
        self.check_input = check_input
        self.verbose = verbose
        self.scale_predictors = scale_predictors

    def _guess_start_params(
        self,
        y: np.ndarray,
        weights: np.ndarray,
        solver: str,
        X,
        P1,
        P2,
        random_state,
        offset: np.ndarray = None,
    ) -> np.ndarray:
        """
        Set mu=starting_mu of the family and do one Newton step
        If solver=cd use cd, else irls
        """
        n_features = X.shape[1]
        family = get_family(self.family)
        link = get_link(self.link, family)
        mu = family.starting_mu(y, weights=weights, offset=offset, link=link)
        eta = link.link(mu)  # linear predictor
        if solver in ["cd", "lbfgs"]:
            # see function _cd_solver
            sigma_inv = 1 / family.variance(mu, phi=1, weights=weights)
            d1 = link.inverse_derivative(eta)
            temp = sigma_inv * d1 * (y - mu)
            if self.fit_intercept:
                score = np.concatenate(([temp.sum()], temp @ X))
            else:
                score = temp @ X  # same as X.T @ temp

            d2_sigma_inv = d1 * d1 * sigma_inv
            diag_fisher = self.diag_fisher
            if diag_fisher:
                fisher = d2_sigma_inv
            else:
                fisher = _safe_sandwich_dot(X, d2_sigma_inv, self.fit_intercept)
            # set up space for search direction d for inner loop
            if self.fit_intercept:
                zero = np.zeros(n_features + 1, dtype=X.dtype)
            else:
                zero = np.zeros(n_features, dtype=X.dtype)
            # initial stopping tolerance of inner loop
            # use L1-norm of minimum of norm of subgradient of F
            # use less restrictive tolerance for initial guess
            inner_tol = 4 * linalg.norm(
                _min_norm_sugrad(coef=zero, grad=-score, P2=P2, P1=P1), ord=1
            )

            # just one outer loop = Newton step
            n_cycles = 0
            coef, coef_P2, n_cycles, inner_tol = _cd_cycle(
                zero,
                X,
                zero,
                score,
                fisher,
                P1,
                P2,
                n_cycles,
                inner_tol,
                max_inner_iter=1000,
                selection=self.selection,
                random_state=random_state,
                diag_fisher=self.diag_fisher,
            )
            # for simplicity no line search here
        else:
            # See _irls_solver
            # h'(eta)
            hp = link.inverse_derivative(eta)
            # working weights W, in principle a diagonal matrix
            # therefore here just as 1d array
            W = hp ** 2 / family.variance(mu, phi=1, weights=weights)
            # working observations
            z = (
                eta
                + np.subtract(y, mu, dtype=get_float_dtype_of_size(X.dtype.itemsize))
                / hp
            )
            # solve A*coef = b
            # A = X' W X + l2 P2, b = X' W z
            coef = _irls_step(X, W, P2, z, fit_intercept=self.fit_intercept)
        return coef

    def get_start_coef(
        self, start_params, X, y, weights, P1, P2, offset, col_means, col_stds
    ) -> np.ndarray:

        if self.warm_start and hasattr(self, "coef_"):
            coef = self.coef_
            intercept = self.intercept_
            if self.fit_intercept:
                coef = np.concatenate((np.array([intercept]), coef))
            if self._center_predictors:
                _standardize_warm_start(coef, col_means, col_stds)

        elif isinstance(start_params, str):
            if start_params == "guess":
                coef = self._guess_start_params(
                    y, weights, self._solver, X, P1, P2, self._random_state, offset
                )
            else:  # start_params == 'zero'
                if self.fit_intercept:
                    coef = np.zeros(
                        X.shape[1] + 1, dtype=_float_itemsize_to_dtype[X.dtype.itemsize]
                    )
                    coef[0] = guess_intercept(y, weights, self._link_instance, offset)
                else:
                    coef = np.zeros(
                        X.shape[1], dtype=_float_itemsize_to_dtype[X.dtype.itemsize]
                    )
        else:  # assign given array as start values
            coef = start_params
            if self._center_predictors:
                _standardize_warm_start(coef, col_means, col_stds)
        return coef

    def set_up_for_fit(self, X, y) -> None:
        #######################################################################
        # 1. input validation                                                 #
        #######################################################################
        # 1.1
        self._validate_hyperparameters()
        # self.family and self.link are user-provided inputs and may be strings or
        #  ExponentialDispersonModel/Link objects
        # self.family_instance_ and self.link_instance_ are cleaned by 'fit' to be
        # ExponentialDispersionModel and Link arguments
        self._family_instance: ExponentialDispersionModel = get_family(self.family)
        # Guarantee that self._link_instance is set to an instance of class Link
        self._link_instance: Link = get_link(self.link, self._family_instance)

        # when fit_intercept is False, we can't center because that would
        # substantially change estimates
        self._center_predictors: bool = self.fit_intercept

        if self.solver == "auto":
            if self.l1_ratio == 0:
                self._solver = "irls"
            else:
                self._solver = "cd"
        else:
            self._solver = self.solver

        self._random_state = check_random_state(self.random_state)

        # 1.4 additional validations ##########################################
        if self.check_input:
            if not np.all(self._family_instance.in_y_range(y)):
                raise ValueError(
                    "Some value(s) of y are out of the valid "
                    "range for family {}".format(
                        self._family_instance.__class__.__name__
                    )
                )

    def tear_down_from_fit(self, X, y, col_means, col_stds, weights, weights_sum):
        """
        Delete attributes that were only needed for the fit method.
        """
        #######################################################################
        # 5a. undo standardization
        #######################################################################
        if self._center_predictors:
            X, self.intercept_, self.coef_ = _unstandardize(
                X, col_means, col_stds, self.intercept_, self.coef_
            )

        if self.fit_dispersion in ["chisqr", "deviance"]:
            # attention because of rescaling of weights
            self.dispersion_ = self.estimate_phi(X, y, weights) * weights_sum

        del self._center_predictors
        del self._solver
        del self._random_state
        return X

    def solve(
        self,
        X: Union[DenseGLMDataMatrix, MKLSparseMatrix],
        y: np.ndarray,
        weights: np.ndarray,
        P2,
        P1: np.ndarray,
        coef: np.ndarray,
        offset: Union[np.ndarray, None],
    ) -> None:
        """
        Must be run after running set_up_for_fit and before running tear_down_from_fit.
        Sets self.coef_ and self.intercept_.
        """
        # 4.1 IRLS ############################################################
        # Note: we already set P2 = l2*P2, see above
        # Note: we already symmetrized P2 = 1/2 (P2 + P2')
        if self._solver == "irls":
            coef, self.n_iter_ = _irls_solver(
                coef=coef,
                X=X,
                y=y,
                weights=weights,
                P2=P2,
                fit_intercept=self.fit_intercept,
                family=self._family_instance,
                link=self._link_instance,
                max_iter=self.max_iter,
                tol=self.tol,
                offset=offset,
            )

        # 4.2 L-BFGS ##########################################################
        elif self._solver == "lbfgs":

            def get_obj_and_derivative(coef):
                mu, devp = self._family_instance._mu_deviance_derivative(
                    coef, X, y, weights, self._link_instance, offset
                )
                dev = self._family_instance.deviance(y, mu, weights)
                intercept = coef.size == X.shape[1] + 1
                idx = 1 if intercept else 0  # offset if coef[0] is intercept
                if P2.ndim == 1:
                    L2 = P2 * coef[idx:]
                else:
                    L2 = P2 @ coef[idx:]
                obj = 0.5 * dev + 0.5 * (coef[idx:] @ L2)
                objp = 0.5 * devp
                objp[idx:] += L2
                return obj, objp

            coef, loss, info = fmin_l_bfgs_b(
                get_obj_and_derivative,
                coef,
                fprime=None,
                iprint=(self.verbose > 0) - 1,
                pgtol=self.tol,
                maxiter=self.max_iter,
                factr=1e3,
            )
            if info["warnflag"] == 1:
                warnings.warn(
                    "lbfgs failed to converge." " Increase the number of iterations.",
                    ConvergenceWarning,
                )
            elif info["warnflag"] == 2:
                warnings.warn("lbfgs failed for the reason: {}".format(info["task"]))
            self.n_iter_ = info["nit"]

        # 4.3 coordinate descent ##############################################
        # Note: we already set P1 = l1*P1, see above
        # Note: we already set P2 = l2*P2, see above
        # Note: we already symmetrized P2 = 1/2 (P2 + P2')
        elif self._solver == "cd":
            coef, self.n_iter_, self._n_cycles, self.diagnostics = _cd_solver(
                coef=coef,
                X=X,
                y=y,
                weights=weights,
                P1=P1,
                P2=P2,
                fit_intercept=self.fit_intercept,
                family=self._family_instance,
                link=self._link_instance,
                max_iter=self.max_iter,
                tol=self.tol,
                selection=self.selection,
                random_state=self._random_state,
                diag_fisher=self.diag_fisher,
                offset=offset,
            )
        #######################################################################
        # 5a. handle intercept
        #######################################################################
        if self.fit_intercept:
            self.intercept_ = coef[0]
            self.coef_ = coef[1:]
        else:
            # set intercept to zero as the other linear models do
            self.intercept_ = 0.0
            self.coef_ = coef

        return coef

    def linear_predictor(self, X, offset: np.ndarray = None):
        """Compute the linear_predictor = X*coef_ + intercept_.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Samples.

        Returns
        -------
        C : array, shape (n_samples,)
            Returns predicted values of linear predictor.
        """
        check_is_fitted(self, "coef_")
        X = check_array(
            X,
            accept_sparse=["csr", "csc", "coo"],
            dtype="numeric",
            copy=True,
            ensure_2d=True,
            allow_nd=False,
        )
        xb = X @ self.coef_ + self.intercept_
        if offset is None:
            return xb
        return xb + offset

    def predict(self, X, sample_weight=None, offset: np.ndarray = None):
        """Predict using GLM with feature matrix X.

        If sample_weight is given, returns prediction*sample_weight.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Samples.

        sample_weight : {None, array-like}, shape (n_samples,), optional \
                (default=None)

        Returns
        -------
        C : array, shape (n_samples,)
            Returns predicted values times sample_weight.
        """
        # TODO: Is copy=True necessary?
        X = check_array(
            X,
            accept_sparse=["csr", "csc", "coo"],
            dtype="numeric",
            copy=True,
            ensure_2d=True,
            allow_nd=False,
        )
        eta = self.linear_predictor(X, offset=offset)
        mu = get_link(self.link, get_family(self.family)).inverse(eta)
        weights = _check_weights(sample_weight, X.shape[0], X.dtype)

        return mu * weights

    def estimate_phi(self, X, y, sample_weight=None):
        """Estimate/fit the dispersion parameter phi.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training data.

        y : array-like, shape (n_samples,)
            Target values.

        sample_weight : {None, array-like}, shape (n_samples,), optional \
                (default=None)
            Sample weights.

        Returns
        -------
        phi : float
            Dispersion parameter.
        """
        check_is_fitted(self, "coef_")
        _dtype = [np.float64, np.float32]
        X, y = check_X_y(
            X,
            y,
            accept_sparse=["csr", "csc", "coo"],
            dtype=_dtype,
            y_numeric=True,
            multi_output=False,
        )
        n_samples, n_features = X.shape
        weights = _check_weights(sample_weight, n_samples, X.dtype)
        eta = X @ self.coef_
        if self.fit_intercept is True:
            eta += self.intercept_
            n_features += 1
        if n_samples <= n_features:
            raise ValueError(
                "Estimation of dispersion parameter phi requires"
                " more samples than features, got"
                " samples=X.shape[0]={} and"
                " n_features=X.shape[1]+fit_intercept={}.".format(n_samples, n_features)
            )
        mu = self._link_instance.inverse(eta)
        if self.fit_dispersion == "chisqr":
            chisq = np.sum(
                weights * (y - mu) ** 2 / self._family_instance.unit_variance(mu)
            )
            return chisq / (n_samples - n_features)
        elif self.fit_dispersion == "deviance":
            dev = self._family_instance.deviance(y, mu, weights)
            return dev / (n_samples - n_features)

    # Note: check_estimator(GeneralizedLinearRegressor) might raise
    # "AssertionError: -0.28014056555724598 not greater than 0.5"
    # unless GeneralizedLinearRegressor has a score which passes the test.
    def score(self, X, y, sample_weight=None):
        """Compute D^2, the percentage of deviance explained.

        D^2 is a generalization of the coefficient of determination R^2.
        R^2 uses squared error and D^2 deviance. Note that those two are equal
        for family='normal'.

        D^2 is defined as
        :math:`D^2 = 1-\\frac{D(y_{true},y_{pred})}{D_{null}}`,
        :math:`D_{null}` is the null deviance, i.e. the deviance of a model
        with intercept alone, which corresponds to :math:`y_{pred} = \\bar{y}`.
        The mean :math:`\\bar{y}` is averaged by sample_weight.
        Best possible score is 1.0 and it can be negative (because the model
        can be arbitrarily worse).

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Test samples.

        y : array-like, shape (n_samples,)
            True values of target.

        sample_weight : {None, array-like}, shape (n_samples,), optional \
                (default=None)
            Sample weights.

        Returns
        -------
        score : float
            D^2 of self.predict(X) w.r.t. y.
        """
        # Note, default score defined in RegressorMixin is R^2 score.
        # TODO: make D^2 a score function in module metrics (and thereby get
        #       input validation and so on)
        weights = _check_weights(sample_weight, y.shape[0], X.dtype)
        mu = self.predict(X)
        family = get_family(self.family)
        dev = family.deviance(y, mu, weights=weights)
        y_mean = np.average(y, weights=weights)
        dev_null = family.deviance(y, y_mean, weights=weights)
        return 1.0 - dev / dev_null

    def _validate_hyperparameters(self) -> None:

        if not isinstance(self.fit_intercept, bool):
            raise ValueError(
                "The argument fit_intercept must be bool;"
                " got {}".format(self.fit_intercept)
            )

        if self.solver == "newton-cg":
            raise ValueError(
                """
                newton-cg solver is no longer supported because
                sklearn.utils.optimize.newton_cg has been deprecated. If you need this
                functionality, please use
                https://github.com/scikit-learn/scikit-learn/pull/9405.
                """
            )

        if self.solver not in ["auto", "irls", "lbfgs", "cd"]:
            raise ValueError(
                "GeneralizedLinearRegressor supports only solvers"
                " 'auto', 'irls', 'lbfgs', and 'cd';"
                " got {}".format(self.solver)
            )
        if not isinstance(self.max_iter, int) or self.max_iter <= 0:
            raise ValueError(
                "Maximum number of iteration must be a positive "
                "integer;"
                " got (max_iter={!r})".format(self.max_iter)
            )
        if not isinstance(self.tol, float) or self.tol <= 0:
            raise ValueError(
                "Tolerance for stopping criteria must be "
                "positive; got (tol={!r})".format(self.tol)
            )
        if not isinstance(self.warm_start, bool):
            raise ValueError(
                "The argument warm_start must be bool;"
                " got {}".format(self.warm_start)
            )
        if self.selection not in ["cyclic", "random"]:
            raise ValueError(
                "The argument selection must be 'cyclic' or "
                "'random'; got (selection={})".format(self.selection)
            )
        if not isinstance(self.diag_fisher, bool):
            raise ValueError(
                "The argument diag_fisher must be bool;"
                " got {}".format(self.diag_fisher)
            )
        if not isinstance(self.copy_X, bool):
            raise ValueError(
                "The argument copy_X must be bool;" " got {}".format(self.copy_X)
            )
        if not isinstance(self.check_input, bool):
            raise ValueError(
                "The argument check_input must be bool; got "
                "(check_input={})".format(self.check_input)
            )
        if self.scale_predictors and not self.fit_intercept:
            raise ValueError(
                "scale_predictors=True is not supported when fit_intercept=False"
            )
        if self.check_input:

            # check if P1 has only non-negative values, negative values might
            # indicate group lasso in the future.
            if not isinstance(self.P1, str):  # if self.P1 != 'identity':
                if not np.all(self.P1 >= 0):
                    raise ValueError("P1 must not have negative values.")


def set_up_and_check_fit_args(
    X,
    y: np.ndarray,
    sample_weight: Union[np.ndarray, None],
    offset: Union[np.ndarray, None],
    solver: str,
    copy_X: bool,
) -> Tuple[
    Union[MKLSparseMatrix, DenseGLMDataMatrix],
    np.ndarray,
    np.ndarray,
    Union[np.ndarray, None],
    float,
]:
    _dtype = [np.float64, np.float32]
    if solver == "cd":
        _stype = ["csc"]
    else:
        _stype = ["csc", "csr"]

    if hasattr(X, "dtype") and X.dtype == np.int64:
        # check_X_y will convert to float32 if we don't do this, which causes
        # precision issues with the new handling of single precision. The new
        # behavior is to give everything the precision of X, but we don't want to
        # do that if X was intially int64.
        X = X.astype(np.float64)

    X, y = check_X_y(
        X,
        y,
        accept_sparse=_stype,
        dtype=_dtype,
        y_numeric=True,
        multi_output=False,
        copy=copy_X,
    )

    # Without converting y to float, deviance might raise
    # ValueError: Integers to negative integer powers are not allowed.
    # Also, y must not be sparse.

    y = np.asarray(y)
    # Make sure everything has the same precision as X
    # This will prevent accidental upcasting later and slow operations on
    # mixed-precision numbers
    y = _to_precision(y, X.dtype.itemsize)
    weights = _check_weights(sample_weight, y.shape[0], X.dtype)
    offset = _check_offset(offset, y.shape[0], X.dtype)

    # IMPORTANT NOTE: Since we want to minimize
    # 1/(2*sum(sample_weight)) * deviance + L1 + L2,
    # deviance = sum(sample_weight * unit_deviance),
    # we rescale weights such that sum(weights) = 1 and this becomes
    # 1/2*deviance + L1 + L2 with deviance=sum(weights * unit_deviance)
    weights_sum: float = np.sum(weights)
    weights = weights / weights_sum
    weights = _to_precision(weights, X.dtype.itemsize)
    #######################################################################
    # 2b. convert to wrapper matrix types
    #######################################################################
    if sparse.issparse(X):
        X = MKLSparseMatrix(X)
    else:
        X = DenseGLMDataMatrix(X)

    return X, y, weights, offset, weights_sum


class GeneralizedLinearRegressor(GeneralizedLinearRegressorBase):
    """Regression via a Generalized Linear Model (GLM) with penalties.

    GLMs based on a reproductive Exponential Dispersion Model (EDM) aim at
    fitting and predicting the mean of the target y as mu=h(X*w). Therefore,
    the fit minimizes the following objective function with combined L1 and L2
    priors as regularizer::

            1/(2*sum(s)) * deviance(y, h(X*w); s)
            + alpha * l1_ratio * ||P1*w||_1
            + 1/2 * alpha * (1 - l1_ratio) * w*P2*w

    with inverse link function h and s=sample_weight. Note that for
    ``sample_weight=None``, one has s_i=1 and sum(s)=n_samples).
    For ``P1=P2='identity'``, the penalty is the elastic net::

            alpha * l1_ratio * ||w||_1
            + 1/2 * alpha * (1 - l1_ratio) * ||w||_2^2

    If you are interested in controlling the L1 and L2 penalties
    separately, keep in mind that this is equivalent to::

            a * L1 + b * L2

    where::

            alpha = a + b and l1_ratio = a / (a + b)

    The parameter ``l1_ratio`` corresponds to alpha in the R package glmnet,
    while ``alpha`` corresponds to the lambda parameter in glmnet.
    Specifically, l1_ratio = 1 is the lasso penalty.

    Read more in the :ref:`User Guide <Generalized_linear_regression>`.

    Parameters
    ----------
    alpha : float, optional (default=1)
        Constant that multiplies the penalty terms and thus determines the
        regularization strength.
        See the notes for the exact mathematical meaning of this
        parameter.``alpha = 0`` is equivalent to unpenalized GLMs. In this
        case, the design matrix X must have full column rank
        (no collinearities).

    l1_ratio : float, optional (default=0)
        The elastic net mixing parameter, with ``0 <= l1_ratio <= 1``. For
        ``l1_ratio = 0`` the penalty is an L2 penalty. ``For l1_ratio = 1`` it
        is an L1 penalty.  For ``0 < l1_ratio < 1``, the penalty is a
        combination of L1 and L2.

    P1 : {'identity', array-like}, shape (n_features,), optional \
            (default='identity')
        With this array, you can exclude coefficients from the L1 penalty.
        Set the corresponding value to 1 (include) or 0 (exclude). The
        default value ``'identity'`` is the same as a 1d array of ones.
        Note that n_features = X.shape[1].

    P2 : {'identity', array-like, sparse matrix}, shape \

            (n_features,) or (n_features, n_features), optional \
            (default='identity')
        With this option, you can set the P2 matrix in the L2 penalty `w*P2*w`.
        This gives a fine control over this penalty (Tikhonov regularization).
        A 2d array is directly used as the square matrix P2. A 1d array is
        interpreted as diagonal (square) matrix. The default 'identity' sets
        the identity matrix, which gives the usual squared L2-norm. If you just
        want to exclude certain coefficients, pass a 1d array filled with 1,
        and 0 for the coefficients to be excluded.
        Note that P2 must be positive semi-definite.

    fit_intercept : boolean, optional (default=True)
        Specifies if a constant (a.k.a. bias or intercept) should be
        added to the linear predictor (X*coef+intercept).

    family : {'normal', 'poisson', 'gamma', 'inverse.gaussian', 'binomial'} \
            or an instance of class ExponentialDispersionModel, \
            optional(default='normal')
        The distributional assumption of the GLM, i.e. which distribution from
        the EDM, specifies the loss function to be minimized.

    link : {'auto', 'identity', 'log', 'logit'} or an instance of class Link, \
            optional (default='auto')
        The link function of the GLM, i.e. mapping from linear predictor
        (X*coef) to expectation (mu). Option 'auto' sets the link depending on
        the chosen family as follows:

        - 'identity' for family 'normal'

        - 'log' for families 'poisson', 'gamma', 'inverse.gaussian'

        - 'logit' for family 'binomial'

    fit_dispersion : {None, 'chisqr', 'deviance'}, optional (default=None)
        Method for estimation of the dispersion parameter phi. Whether to use
        the chi squared statistic or the deviance statistic. If None, the
        dispersion is not estimated.

    solver : {'auto', 'cd', 'irls', 'lbfgs'}, \
            optional (default='auto')
        Algorithm to use in the optimization problem:

        'auto'
            Sets 'irls' if l1_ratio equals 0, else 'cd'.

        'cd'
            Coordinate descent algorithm. It can deal with L1 as well as L2
            penalties. Note that in order to avoid unnecessary memory
            duplication of X in the ``fit`` method, X should be directly passed
            as a Fortran-contiguous numpy array or sparse csc matrix.

        'irls'
            Iterated reweighted least squares.
            It is the standard algorithm for GLMs. It cannot deal with
            L1 penalties.

        'lbfgs'
            Calls scipy's L-BFGS-B optimizer. It cannot deal with L1 penalties.

        Note that all solvers except lbfgs use the fisher matrix, i.e. the
        expected Hessian instead of the Hessian matrix.

    max_iter : int, optional (default=100)
        The maximal number of iterations for solver algorithms.

    tol : float, optional (default=1e-4)
        Stopping criterion. For the irls and lbfgs solvers,
        the iteration will stop when ``max{|g_i|, i = 1, ..., n} <= tol``
        where ``g_i`` is the i-th component of the gradient (derivative) of
        the objective function. For the cd solver, convergence is reached
        when ``sum_i(|minimum-norm of g_i|)``, where ``g_i`` is the
        subgradient of the objective and minimum-norm of ``g_i`` is the element
        of the subgradient ``g_i`` with the smallest L2-norm.

    warm_start : boolean, optional (default=False)
        If set to ``True``, reuse the solution of the previous call to ``fit``
        as initialization for ``coef_`` and ``intercept_`` (supersedes option
        ``start_params``). If set to ``True`` or if the attribute ``coef_``
        does not exit (first call to ``fit``), option ``start_params`` sets the
        start values for ``coef_`` and ``intercept_``.

    start_params : {'guess', 'zero', array of shape (n_features*, )}, \
            optional (default='guess')
        Relevant only if ``warm_start=False`` or if fit is called
        the first time (``self.coef_`` does not yet exist).

        'guess'
            Start values of mu are calculated by family.starting_mu(..). Then,
            one Newton step obtains start values for ``coef_``. If
            ``solver='irls'``, it uses one irls step, else the Newton step is
            calculated by the cd solver.
            This gives usually good starting values.

        'zero'
        All coefficients are set to zero. If ``fit_intercept=True``, the
        start value for the intercept is obtained by the weighted average of y.

        array
        The array of size n_features* is directly used as start values
        for ``coef_``. If ``fit_intercept=True``, the first element
        is assumed to be the start value for the ``intercept_``.
        Note that n_features* = X.shape[1] + fit_intercept, i.e. it includes
        the intercept in counting.

    selection : str, optional (default='cyclic')
        For the solver 'cd' (coordinate descent), the coordinates (features)
        can be updated in either cyclic or random order.
        If set to 'random', a random coefficient is updated every iteration
        rather than looping over features sequentially in the same order. This
        (setting to 'random') often leads to significantly faster convergence
        especially when tol is higher than 1e-4.

    random_state : {int, RandomState instance, None}, optional (default=None)
        The seed of the pseudo random number generator that selects a random
        feature to be updated for solver 'cd' (coordinate descent).
        If int, random_state is the seed used by the random
        number generator; if RandomState instance, random_state is the random
        number generator; if None, the random number generator is the
        RandomState instance used by `np.random`. Used when ``selection`` ==
        'random'.

    diag_fisher : boolean, optional, (default=False)
        Only relevant for solver 'cd' (see also ``start_params='guess'``).
        If ``False``, the full Fisher matrix (expected Hessian) is computed in
        each outer iteration (Newton iteration). If ``True``, only a diagonal
        matrix (stored as 1d array) is computed, such that
        fisher = X.T @ diag @ X. This saves memory and matrix-matrix
        multiplications, but needs more matrix-vector multiplications. If you
        use large sparse X or if you have many features,
        i.e. n_features >> n_samples, you might set this option to ``True``.

    copy_X : boolean, optional, (default=True)
        If ``True``, X will be copied; else, it may be overwritten.

    check_input : boolean, optional (default=True)
        Allow to bypass several checks on input: y values in range of family,
        sample_weight non-negative, P2 positive semi-definite.
        Don't use this parameter unless you know what you do.

    center_predictors : boolean, optional (default=True)
        Subtract the means from each column. Centering predictors can improve
        performance of coordinate descent by a substantial amount. This
        defaults to True, but will be False if fit_intercept is False or if
        diag_fisher is True

    verbose : int, optional (default=0)
        For the lbfgs solver set verbose to any positive number for verbosity.

    Attributes
    ----------
    coef_ : array, shape (n_features,)
        Estimated coefficients for the linear predictor (X*coef_+intercept_) in
        the GLM.

    intercept_ : float
        Intercept (a.k.a. bias) added to linear predictor.

    dispersion_ : float
        The dispersion parameter :math:`\\phi` if ``fit_dispersion`` was set.

    n_iter_ : int
        Actual number of iterations used in solver.

    Notes
    -----
    The fit itself does not need Y to be from an EDM, but only assumes
    the first two moments to be :math:`E[Y_i]=\\mu_i=h((Xw)_i)` and
    :math:`Var[Y_i]=\\frac{\\phi}{s_i} v(\\mu_i)`. The unit variance function
    :math:`v(\\mu_i)` is a property of and given by the specific EDM, see
    :ref:`User Guide <Generalized_linear_regression>`.

    The parameters :math:`w` (`coef_` and `intercept_`) are estimated by
    minimizing the deviance plus penalty term, which is equivalent to
    (penalized) maximum likelihood estimation.

    For alpha > 0, the feature matrix X should be standardized in order to
    penalize features equally strong. Call
    :class:`sklearn.preprocessing.StandardScaler` before calling ``fit``.

    If the target y is a ratio, appropriate sample weights s should be
    provided.
    As an example, consider Poisson distributed counts z (integers) and
    weights s=exposure (time, money, persons years, ...). Then you fit
    y = z/s, i.e. ``GeneralizedLinearModel(family='poisson').fit(X, y,
    sample_weight=s)``. The weights are necessary for the right (finite
    sample) mean.
    Consider :math:`\\bar{y} = \\frac{\\sum_i s_i y_i}{\\sum_i s_i}`,
    in this case one might say that y has a 'scaled' Poisson distributions.
    The same holds for other distributions.

    References
    ----------
    For the coordinate descent implementation:
        * Guo-Xun Yuan, Chia-Hua Ho, Chih-Jen Lin
          An Improved GLMNET for L1-regularized Logistic Regression,
          Journal of Machine Learning Research 13 (2012) 1999-2030
          https://www.csie.ntu.edu.tw/~cjlin/papers/l1_glmnet/long-glmnet.pdf
    """

    def __init__(
        self,
        alpha=1.0,
        l1_ratio=0,
        P1="identity",
        P2="identity",
        fit_intercept=True,
        family: Union[str, ExponentialDispersionModel] = "normal",
        link: Union[str, Link] = "auto",
        fit_dispersion=None,
        solver="auto",
        max_iter=100,
        tol=1e-4,
        warm_start=False,
        start_params="guess",
        selection="cyclic",
        random_state=None,
        diag_fisher=False,
        copy_X=True,
        check_input=True,
        verbose=0,
        scale_predictors=False,
        fit_args_reformat="safe",
    ):
        self.alpha = alpha
        self.fit_args_reformat = fit_args_reformat
        super().__init__(
            l1_ratio,
            P1,
            P2,
            fit_intercept,
            family,
            link,
            fit_dispersion,
            solver,
            max_iter,
            tol,
            warm_start,
            start_params,
            selection,
            random_state,
            diag_fisher,
            copy_X,
            check_input,
            verbose,
            scale_predictors,
        )

    def _validate_hyperparameters(self) -> None:
        if (
            not (isinstance(self.alpha, float) or isinstance(self.alpha, int))
            or self.alpha < 0
        ):
            raise ValueError(
                "Penalty term must be a non-negative number;"
                " got (alpha={})".format(self.alpha)
            )

        if (
            not np.isscalar(self.l1_ratio)
            # check for numeric, i.e. not a string
            or not np.issubdtype(np.asarray(self.l1_ratio).dtype, np.number)
            or self.l1_ratio < 0
            or self.l1_ratio > 1
        ):
            raise ValueError(
                "l1_ratio must be a number in interval [0, 1];"
                " got (l1_ratio={})".format(self.l1_ratio)
            )
        super()._validate_hyperparameters()

    def fit(self, X, y, sample_weight=None, offset=None, weights_sum: float = None):
        """Fit a Generalized Linear Model.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training data.

        y : array-like, shape (n_samples,)
            Target values.

        sample_weight : {None, array-like}, shape (n_samples,),\
                optional (default=None)
            Individual weights w_i for each sample. Note that for an
            Exponential Dispersion Model (EDM), one has
            Var[Y_i]=phi/w_i * v(mu).
            If Y_i ~ EDM(mu, phi/w_i), then
            sum(w*Y)/sum(w) ~ EDM(mu, phi/sum(w)), i.e. the mean of y is a
            weighted average with weights=sample_weight.

        offset: {None, array-like}, shape (n_samples,), optional (default=None)
            Added to linear predictor "eta". An offset of 3 will increase expected
            y by 3 if the link is linear, and will multiply expected y by 3 if the
            link is log.

        Returns
        -------
        self : returns an instance of self.
        """

        if self.fit_args_reformat == "safe":
            X, y, weights, offset, weights_sum = set_up_and_check_fit_args(
                X, y, sample_weight, offset, solver=self.solver, copy_X=self.copy_X
            )
        else:
            weights = sample_weight

        #######################################################################
        # 2a. rescaling of weights (sample_weight)                             #
        #######################################################################

        self.set_up_for_fit(X, y)

        if self.alpha > 0 and self.l1_ratio > 0 and self._solver not in ["cd"]:
            raise ValueError(
                "The chosen solver (solver={}) can't deal "
                "with L1 penalties, which are included with "
                "(alpha={}) and (l1_ratio={}).".format(
                    self._solver, self.alpha, self.l1_ratio
                )
            )

        _dtype = [np.float64, np.float32]
        if self._solver == "cd":
            _stype = ["csc"]
        else:
            _stype = ["csc", "csr"]

        # 1.3 arguments to take special care ##################################
        # P1, P2, start_params
        P1 = setup_p1(self.P1, X, X.dtype, self.alpha, self.l1_ratio)
        P2 = setup_p2(self.P2, X, _stype, X.dtype, self.alpha, self.l1_ratio)

        start_params = initialize_start_params(
            self.start_params,
            n_cols=X.shape[1],
            fit_intercept=self.fit_intercept,
            _dtype=_dtype,
        )

        # 1.4 additional validations ##########################################
        if self.check_input:
            # check if P2 is positive semidefinite
            if not isinstance(self.P2, str):  # self.P2 != 'identity'

                if not is_pos_semidef(P2):
                    if P2.ndim == 1 or P2.shape[0] == 1:
                        error = "1d array P2 must not have negative values."
                    else:
                        error = "P2 must be positive semi-definite."
                    raise ValueError(error)

            # TODO: if alpha=0 check that X is not rank deficient
            # TODO: what else to check?

        #######################################################################
        # 2c. potentially rescale predictors
        #######################################################################
        if self._center_predictors:
            X, col_means, col_stds = X.standardize(weights, self.scale_predictors)
        else:
            col_means, col_stds = None, None

        #######################################################################
        # 3. initialization of coef = (intercept_, coef_)                     #
        #######################################################################
        # Note: Since phi=self.dispersion_ does not enter the estimation
        #       of mu_i=E[y_i], set it to 1.

        # set start values for coef
        coef = self.get_start_coef(
            start_params, X, y, weights, P1, P2, offset, col_means, col_stds
        )

        #######################################################################
        # 4. fit                                                              #
        #######################################################################
        self.solve(X, y, weights, P2, P1, coef, offset)

        self.tear_down_from_fit(X, y, col_means, col_stds, weights, weights_sum)

        return self

    def report_diagnostics(self):
        if hasattr(self, "diagnostics"):
            print("diagnostics:")
            import pandas as pd

            print(
                pd.DataFrame(
                    columns=["inner_tol", "n_iter", "n_cycles", "runtime", "intercept"],
                    data=self.diagnostics,
                ).set_index("n_iter", drop=True)
            )
        else:
            print("solver does not report diagnostics")


class PoissonRegressor(GeneralizedLinearRegressor):
    """Regression with the response variable y following a Poisson distribution

    GLMs based on a reproductive Exponential Dispersion Model (EDM) aim at
    fitting and predicting the mean of the target y as mu=h(X*w).
    The fit minimizes the following objective function with L2 regularization::

            1/(2*sum(s)) * deviance(y, h(X*w); s) + 1/2 * alpha * ||w||_2^2

    with inverse link function h and s=sample_weight. Note that for
    ``sample_weight=None``, one has s_i=1 and sum(s)=n_samples).

    Read more in the :ref:`User Guide <Generalized_linear_regression>`.

    Parameters
    ----------
    alpha : float, optional (default=1)
        Constant that multiplies the penalty terms and thus determines the
        regularization strength.
        See the notes for the exact mathematical meaning of this
        parameter.``alpha = 0`` is equivalent to unpenalized GLMs. In this
        case, the design matrix X must have full column rank
        (no collinearities).

    fit_intercept : boolean, optional (default=True)
        Specifies if a constant (a.k.a. bias or intercept) should be
        added to the linear predictor (X*coef+intercept).

    fit_dispersion : {None, 'chisqr', 'deviance'}, optional (default=None)
        Method for estimation of the dispersion parameter phi. Whether to use
        the chi squared statistic or the deviance statistic. If None, the
        dispersion is not estimated.

    solver : {'irls', 'lbfgs'}, optional (default='irls')
        Algorithm to use in the optimization problem:

        'irls'
            Iterated reweighted least squares. It is the standard algorithm
            for GLMs.

        'lbfgs'
            Calls scipy's L-BFGS-B optimizer.

        Note that all solvers except lbfgs use the fisher matrix, i.e. the
        expected Hessian instead of the Hessian matrix.

    max_iter : int, optional (default=100)
        The maximal number of iterations for solver algorithms.

    tol : float, optional (default=1e-4)
        Stopping criterion. For the irls and lbfgs solvers,
        the iteration will stop when ``max{|g_i|, i = 1, ..., n} <= tol``
        where ``g_i`` is the i-th component of the gradient (derivative) of
        the objective function.

    warm_start : boolean, optional (default=False)
        If set to ``True``, reuse the solution of the previous call to ``fit``
        as initialization for ``coef_`` and ``intercept_`` (supersedes option
        ``start_params``). If set to ``True`` or if the attribute ``coef_``
        does not exit (first call to ``fit``), option ``start_params`` sets the
        start values for ``coef_`` and ``intercept_``.

    start_params : {'guess', 'zero', array of shape (n_features*, )}, \
            optional (default='guess')
        Relevant only if ``warm_start=False`` or if fit is called
        the first time (``self.coef_`` does not yet exist).

        'guess'
            Start values of mu are calculated by family.starting_mu(..). Then,
            one Newton step obtains start values for ``coef_``. If
            ``solver='irls'``, it uses one irls step. This gives usually good
            starting values.

        'zero'
        All coefficients are set to zero. If ``fit_intercept=True``, the
        start value for the intercept is obtained by the weighted average of y.

        array
        The array of size n_features* is directly used as start values
        for ``coef_``. If ``fit_intercept=True``, the first element
        is assumed to be the start value for the ``intercept_``.
        Note that n_features* = X.shape[1] + fit_intercept, i.e. it includes
        the intercept in counting.

    random_state : {int, RandomState instance, None}, optional (default=None)
        If int, random_state is the seed used by the random
        number generator; if RandomState instance, random_state is the random
        number generator; if None, the random number generator is the
        RandomState instance used by `np.random`. Used when ``selection`` ==
        'random'.

    copy_X : boolean, optional, (default=True)
        If ``True``, X will be copied; else, it may be overwritten.

    verbose : int, optional (default=0)
        For the lbfgs solver set verbose to any positive number for verbosity.

    Attributes
    ----------
    coef_ : array, shape (n_features,)
        Estimated coefficients for the linear predictor (X*coef_+intercept_) in
        the GLM.

    intercept_ : float
        Intercept (a.k.a. bias) added to linear predictor.

    dispersion_ : float
        The dispersion parameter :math:`\\phi` if ``fit_dispersion`` was set.

    n_iter_ : int
        Actual number of iterations used in solver.

    Notes
    -----
    The fit itself does not need Y to be from an EDM, but only assumes
    the first two moments to be :math:`E[Y_i]=\\mu_i=h((Xw)_i)` and
    :math:`Var[Y_i]=\\frac{\\phi}{s_i} v(\\mu_i)`. The unit variance function
    :math:`v(\\mu_i)` is a property of and given by the specific EDM, see
    :ref:`User Guide <Generalized_linear_regression>`.

    The parameters :math:`w` (`coef_` and `intercept_`) are estimated by
    minimizing the deviance plus penalty term, which is equivalent to
    (penalized) maximum likelihood estimation.

    For alpha > 0, the feature matrix X should be standardized in order to
    penalize features equally strong.

    If the target y is a ratio, appropriate sample weights s should be
    provided.
    As an example, consider Poisson distributed counts z (integers) and
    weights s=exposure (time, money, persons years, ...). Then you fit
    y = z/s, i.e. ``PoissonRegressor().fit(X, y, sample_weight=s)``.
    The weights are necessary for the right (finite sample) mean.
    Consider :math:`\\bar{y} = \\frac{\\sum_i s_i y_i}{\\sum_i s_i}`,
    in this case one might say that y has a 'scaled' Poisson distributions.

    References
    ----------
    For the coordinate descent implementation:
        * Guo-Xun Yuan, Chia-Hua Ho, Chih-Jen Lin
          An Improved GLMNET for L1-regularized Logistic Regression,
          Journal of Machine Learning Research 13 (2012) 1999-2030
          https://www.csie.ntu.edu.tw/~cjlin/papers/l1_glmnet/long-glmnet.pdf
    """

    def __init__(
        self,
        alpha=1.0,
        fit_intercept=True,
        fit_dispersion=None,
        solver="irls",
        max_iter=100,
        tol=1e-4,
        warm_start=False,
        start_params="guess",
        random_state=None,
        copy_X=True,
        verbose=0,
    ):

        super().__init__(
            alpha=alpha,
            fit_intercept=fit_intercept,
            family="poisson",
            link="log",
            fit_dispersion=fit_dispersion,
            solver=solver,
            max_iter=max_iter,
            tol=tol,
            warm_start=warm_start,
            start_params=start_params,
            random_state=random_state,
            copy_X=copy_X,
            verbose=verbose,
        )

    def _more_tags(self):
        return {"requires_positive_y": True}
