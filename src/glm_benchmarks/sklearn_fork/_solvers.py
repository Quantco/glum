from __future__ import division

import time
import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
from scipy import linalg, sparse
from scipy.optimize import fmin_l_bfgs_b
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils import check_array
from sklearn.utils.validation import check_random_state

from ._cd_fast import _norm_min_subgrad, enet_coordinate_descent_gram
from ._distribution import ExponentialDispersionModel
from ._link import Link
from ._util import _safe_lin_pred, _safe_sandwich_dot


def _least_squares_solver(
    d: np.ndarray,
    X,
    weights,
    y,
    coef: np.ndarray,
    score,
    fisher_W,
    P1,
    P2,
    inner_tol: float,
    max_inner_iter,
    selection,
    random_state,
):
    intercept = coef.size == X.shape[1] + 1
    idx = 1 if intercept else 0  # offset if coef[0] is intercept

    fisher = _safe_sandwich_dot(X, fisher_W, intercept)
    coef_P2 = add_P2_fisher(fisher, P2, coef, idx)

    # TODO:
    S = score.copy()
    S[idx:] -= coef_P2

    # TODO: In cases where we have lots of columns, we might want to avoid the
    # sandwich product and use something like iterative lsqr or lsmr.
    d = linalg.solve(fisher, S, overwrite_a=True, overwrite_b=True, assume_a="pos")
    return d, coef_P2, 1


def _cd_solver(
    d: np.ndarray,
    X,
    weights,
    y,
    coef: np.ndarray,
    score,
    fisher_W,
    P1,
    P2,
    inner_tol: float,
    max_inner_iter=50000,
    selection="cyclic",
    random_state=None,
):
    intercept = coef.size == X.shape[1] + 1
    idx = 1 if intercept else 0  # offset if coef[0] is intercept

    fisher = _safe_sandwich_dot(X, fisher_W, intercept)
    coef_P2 = add_P2_fisher(fisher, P2, coef, idx)

    rhs = -score
    rhs[idx:] += coef_P2

    random = selection == "random"
    new_coef = coef.copy()
    new_coef, gap, _, n_cycles = enet_coordinate_descent_gram(
        new_coef,
        P1,
        fisher,
        rhs,
        max_inner_iter,
        inner_tol,
        random_state,
        intercept,
        random,
    )
    return new_coef - coef, coef_P2, n_cycles


def add_P2_fisher(fisher, P2, coef, idx):
    if P2.ndim == 1:
        coef_P2 = coef[idx:] * P2
        idiag = np.arange(start=idx, stop=fisher.shape[0])
        fisher[(idiag, idiag)] += P2
    else:
        coef_P2 = coef[idx:] @ P2
        if sparse.issparse(P2):
            fisher[idx:, idx:] += P2.toarray()
        else:
            fisher[idx:, idx:] += P2
    return coef_P2


def make_coef_P2(coef, P2, idx):
    out = np.empty_like(coef)
    if idx == 1:
        out[0] = 0
    if P2.ndim == 1:
        out[idx:] = coef[idx:] * P2
    else:
        out[idx:] = coef[idx:] @ P2
    return out


def _irls_solver(
    inner_solver,
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
    max_inner_iter: int = 100000,
    gradient_tol: Optional[float] = 1e-4,
    step_size_tol: Optional[float] = 1e-4,
    fixed_inner_tol: Optional[Tuple] = None,
    selection="cyclic",
    random_state=None,
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

    gradient_tol : float, optional (default=1e-4)
        Convergence criterion is
        sum_i(|minimum of norm of subgrad of objective_i|)<=tol.

    step_size_tol : float, optional (default=1e-4)

    selection : str, optional (default='cyclic')
        If 'random', randomly chose features in inner loop.

    random_state : {int, RandomState instance, None}, optional (default=None)

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
    iteration_start = time.time()

    eta, mu, score, fisher_W = family._eta_mu_score_fisher(
        coef=coef, phi=1, X=X, y=y, weights=weights, link=link, offset=offset,
    )
    coef_P2 = make_coef_P2(coef, P2, idx)

    # set up space for search direction d for inner loop
    d = np.zeros_like(coef)

    # minimum subgradient norm
    def calc_mn_subgrad_norm():
        return _norm_min_subgrad(coef, -score + coef_P2, P1, idx)

    # the ratio of inner tolerance to the minimum subgradient norm
    # This wasn't explored in the newGLMNET paper linked above.
    # That paper essentially uses inner_tol_ratio = 1.0, but using a slightly
    # lower value is much faster.
    # By comparison, the original GLMNET paper uses inner_tol = tol.
    # So, inner_tol_ratio < 1 is sort of a compromise between the two papers.
    # The value should probably be between 0.01 and 0.5. 0.1 works well for many problems
    inner_tol_ratio = 0.1

    def calc_inner_tol(mn_subgrad_norm):
        if fixed_inner_tol is None:
            # Another potential rule limits the inner tol to be no smaller than tol
            # return max(mn_subgrad_norm * inner_tol_ratio, tol)
            return mn_subgrad_norm * inner_tol_ratio
        else:
            return fixed_inner_tol[0]

    mn_subgrad_norm = calc_mn_subgrad_norm()

    Fw = None
    diagnostics = []
    # outer loop
    while n_iter < max_iter:
        # stopping tolerance of inner loop
        # use L1-norm of minimum of norm of subgradient of F
        inner_tol = calc_inner_tol(mn_subgrad_norm)

        n_iter += 1
        # initialize search direction d (to be optimized) with zero
        d.fill(0)

        # inner loop = _cd_cycle
        d, coef_P2, n_cycles_this_iter = inner_solver(
            d,
            X,
            weights,
            y,
            coef,
            score,
            fisher_W,
            P1,
            P2,
            inner_tol,
            max_inner_iter=max_inner_iter,
            selection=selection,
            random_state=random_state,
        )
        n_cycles += n_cycles_this_iter

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
            eta_wd = eta + la * X_dot_d
            mu_wd = link.inverse(eta_wd)

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
        step = la * d
        coef += step

        # We can avoid a matrix-vector product inside _eta_mu_score_fisher by
        # updating eta here.
        # NOTE: This might accumulate some numerical error over a sufficient
        # number of iterations, maybe we should completely recompute eta every
        # N iterations?
        eta = eta_wd
        mu = mu_wd

        # calculate eta, mu, score, Fisher matrix for next iteration
        eta, mu, score, fisher_W = family._eta_mu_score_fisher(
            coef=coef,
            phi=1,
            X=X,
            y=y,
            weights=weights,
            link=link,
            eta=eta,
            mu=mu,
            offset=offset,
        )
        coef_P2 = make_coef_P2(coef, P2, idx)

        converged, mn_subgrad_norm = check_convergence(
            step, coef, -score + coef_P2, P1, idx, gradient_tol, step_size_tol
        )

        iteration_runtime = time.time() - iteration_start
        coef_l1 = np.sum(np.abs(coef))
        coef_l2 = np.linalg.norm(coef)
        step_l2 = np.linalg.norm(d)
        diagnostics.append(
            [
                mn_subgrad_norm,
                coef_l1,
                coef_l2,
                step_l2,
                n_iter,
                n_cycles,
                iteration_runtime,
                coef[0],
            ]
        )
        iteration_start = time.time()

        # stopping criterion for outer loop
        # sum_i(|minimum-norm of subgrad of F(w)_i|)
        # fp_wP2 = f'(w) + w*P2
        # Note: eta, mu and score are already updated
        # this also updates the inner tolerance for the next loop!
        if converged:
            break
        # end of outer loop

    if not converged:
        warnings.warn(
            "IRLS failed to converge. Increase"
            " the maximum number of iterations max_iter"
            " (currently {})".format(max_iter),
            ConvergenceWarning,
        )
    return coef, n_iter, n_cycles, diagnostics


def check_convergence(
    step,
    coef,
    grad,
    P1,
    idx,
    gradient_tol: Optional[float],
    step_size_tol: Optional[float],
):
    # minimum subgradient norm
    mn_subgrad_norm = _norm_min_subgrad(coef, grad, P1, idx)
    step_size = linalg.norm(step)
    converged = (gradient_tol is not None and mn_subgrad_norm < gradient_tol) or (
        step_size_tol is not None and step_size < step_size_tol
    )
    return converged, mn_subgrad_norm


def _lbfgs_solver(
    coef,
    X,
    y: np.ndarray,
    weights: np.ndarray,
    P2: Union[np.ndarray, sparse.spmatrix],
    verbose: bool,
    family: ExponentialDispersionModel,
    link: Link,
    max_iter: int = 100,
    tol: float = 1e-4,
    offset: np.ndarray = None,
):
    def get_obj_and_derivative(coef):
        mu, devp = family._mu_deviance_derivative(coef, X, y, weights, link, offset)
        dev = family.deviance(y, mu, weights)
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
        iprint=(verbose > 0) - 1,
        pgtol=tol,
        maxiter=max_iter,
        factr=1e2,
    )
    if info["warnflag"] == 1:
        warnings.warn(
            "lbfgs failed to converge." " Increase the number of iterations.",
            ConvergenceWarning,
        )
    elif info["warnflag"] == 2:
        warnings.warn("lbfgs failed for the reason: {}".format(info["task"]))
    n_iter_ = info["nit"]

    return coef, n_iter_, -1, None
