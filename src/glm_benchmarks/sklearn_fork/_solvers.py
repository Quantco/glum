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

from glm_benchmarks.matrix import MatrixBase

from ._distribution import ExponentialDispersionModel
from ._link import Link
from ._util import _safe_lin_pred, _safe_toarray


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


def _least_squares_solver(
    d: np.ndarray,
    X: MatrixBase,
    coef: np.ndarray,
    score,
    fisher,
    P1,
    P2,
    n_cycles: int,
    inner_tol: float,
    max_inner_iter,
    selection,
    random_state,
    diag_fisher=False,
):
    S = score.copy()
    intercept = coef.size == X.shape[1] + 1
    idx = 1 if intercept else 0  # offset if coef[0] is intercept

    coef_P2 = add_P2_fisher(fisher, P2, coef, idx, diag_fisher)

    # TODO:
    S[idx:] -= coef_P2

    # TODO: In cases where we have lots of columns, we might want to avoid the
    # sandwich product and use something like iterative lsqr or lsmr.
    # TODO: need to only pass X and W to _ls_solver and _cd_solver. Then, we
    # can calculate fisher and score or use other solvers internal to the inner
    # solver.
    d = linalg.solve(fisher, S, overwrite_a=True, overwrite_b=True, assume_a="pos")
    return d, coef_P2, 1, inner_tol


def _cd_solver(
    d: np.ndarray,
    X: MatrixBase,
    coef: np.ndarray,
    score,
    fisher,
    P1,
    P2,
    n_cycles: int,
    inner_tol: float,
    max_inner_iter=50000,
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

    coef_P2 = add_P2_fisher(fisher, P2, coef, idx, diag_fisher)

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
                x_j = np.squeeze(np.array(X.getcol(j).toarray()))
                Bj = np.zeros_like(A)
                if intercept:
                    Bj[0] = fisher @ x_j
                Bj[idx:] = _safe_toarray((fisher * x_j) @ X).ravel()

                if P2.ndim == 1:
                    Bj[idx + j] += P2[j]
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


def add_P2_fisher(fisher, P2, coef, idx, diag_fisher):
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
    return coef_P2


def _irls_solver(
    inner_solver,
    coef,
    X: MatrixBase,
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

    gradient_tol : float, optional (default=1e-4)
        Convergence criterion is
        sum_i(|minimum of norm of subgrad of objective_i|)<=tol.

    step_size_tol : float, optional (default=1e-4)

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
        d, coef_P2, n_cycles, inner_tol = inner_solver(
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
        eta, mu, score, fisher = family._eta_mu_score_fisher(
            coef=coef,
            phi=1,
            X=X,
            y=y,
            weights=weights,
            link=link,
            diag_fisher=diag_fisher,
            eta=eta,
            mu=mu,
            offset=offset,
        )

        converged, mn_subgrad_norm = check_convergence(
            step, coef, -score, P2, P1, gradient_tol, step_size_tol
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
    P2,
    P1,
    gradient_tol: Optional[float],
    step_size_tol: Optional[float],
):
    # minimum subgradient norm
    mn_subgrad_norm = linalg.norm(
        _min_norm_sugrad(coef=coef, grad=grad, P2=P2, P1=P1), ord=1
    )
    step_size = linalg.norm(step)
    converged = (gradient_tol is not None and mn_subgrad_norm < gradient_tol) or (
        step_size_tol is not None and step_size < step_size_tol
    )
    return converged, mn_subgrad_norm


def _lbfgs_solver(
    coef,
    X: MatrixBase,
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
