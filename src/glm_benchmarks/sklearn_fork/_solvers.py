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


def _least_squares_solver(state, data):
    fisher = build_fisher(data.X, state.fisher_W, data.fit_intercept, data.P2)

    # TODO: In cases where we have lots of columns, we might want to avoid the
    # sandwich product and use something like iterative lsqr or lsmr.
    d = linalg.solve(
        fisher, state.score, overwrite_a=True, overwrite_b=True, assume_a="pos"
    )
    return d, 1


def _cd_solver(state, data):
    fisher = build_fisher(data.X, state.fisher_W, data.fit_intercept, data.P2)
    new_coef, gap, _, n_cycles = enet_coordinate_descent_gram(
        state.coef.copy(),
        data.P1,
        fisher,
        -state.score,
        data.max_inner_iter,
        state.inner_tol,
        data.random_state,
        data.fit_intercept,
        data.selection == "random",
    )
    return new_coef - state.coef, n_cycles


def build_fisher(X, fisher_W, intercept, P2):
    idx = 1 if intercept else 0
    fisher = _safe_sandwich_dot(X, fisher_W, intercept)
    if P2.ndim == 1:
        idiag = np.arange(start=idx, stop=fisher.shape[0])
        fisher[(idiag, idiag)] += P2
    else:
        if sparse.issparse(P2):
            fisher[idx:, idx:] += P2.toarray()
        else:
            fisher[idx:, idx:] += P2
    return fisher


def _irls_solver(inner_solver, coef, data) -> Tuple[np.ndarray, int, int, List[List]]:
    """Solve GLM with L1 and L2 penalty by IRLS

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
    3a. Coordinate descent by updating coordinate j (d -> d+z*e_j):
       min_z q(d+z*e_j)
       = min_z q(d+z*e_j) - q(d)
       = min_z A_j z + 1/2 B_jj z^2
               + ||P1_j (w_j+d_j+z)||_1 - ||P1_j (w_j+d_j)||_1
       A = f'(w) + d*H(w) + (w+d)*P2
       B = H + P2
    3b. Least squares solve of the quadratic approximation.

    Repeat steps 1-3 until convergence.
    Note: Use Fisher matrix instead of Hessian for H.
    Note: f' = -score, H = Fisher matrix

    Parameters
    ----------
    inner_solver
        A least squares solver that can handle the appropriate penalties. With
        an L1 penalty, this will _cd_solver. With only an L2 penalty,
        _least_squares_solver will be more efficient.
    coef : ndarray, shape (c,)
        If fit_intercept=False, shape c=X.shape[1].
        If fit_intercept=True, then c=X.shape[1] + 1.
    data : IRLSData
        Data object containing all the data and solver parameters.
    """

    state = IRLSState(coef, data)
    state.eta, state.mu, state.score, state.fisher_W, state.coef_P2 = update_quadratic(
        state, data
    )
    state.converged, state.mn_subgrad_norm, state.inner_tol = check_convergence(
        state, data
    )

    while state.n_iter < data.max_iter and not state.converged:

        # 1) Solve the L1 and L2 penalized least squares problem
        d, n_cycles_this_iter = inner_solver(state, data)
        state.n_cycles += n_cycles_this_iter

        # 2) Line search
        state.coef, state.step, state.Fw, state.eta, state.mu = line_search(
            state, data, d
        )

        # 3) Update the quadratic approximation
        (
            state.eta,
            state.mu,
            state.score,
            state.fisher_W,
            state.coef_P2,
        ) = update_quadratic(state, data)

        # 4) Check if we've converged
        state.converged, state.mn_subgrad_norm, state.inner_tol = check_convergence(
            state, data
        )
        state.record_iteration()

    if not state.converged:
        warnings.warn(
            "IRLS failed to converge. Increase"
            " the maximum number of iterations max_iter"
            " (currently {})".format(data.max_iter),
            ConvergenceWarning,
        )
    return state.coef, state.n_iter, state.n_cycles, state.diagnostics


class IRLSData:
    def __init__(
        self,
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
    ):
        """
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
        self.X = X
        self.y = y
        self.weights = weights
        self.P1 = P1

        # Note: we already set P2 = l2*P2, P1 = l1*P1
        # Note: we already symmetrized P2 = 1/2 (P2 + P2')
        self.P2 = P2

        self.fit_intercept = fit_intercept
        self.family = family
        self.link = link
        self.max_iter = max_iter
        self.max_inner_iter = max_inner_iter
        self.gradient_tol = gradient_tol
        self.step_size_tol = step_size_tol
        self.fixed_inner_tol = fixed_inner_tol
        self.selection = selection
        self.random_state = random_state
        self.offset = offset

        self.intercept_offset = 1 if self.fit_intercept else 0

        self.check_data()

    def check_data(self):
        if self.P2.ndim == 2:
            self.P2 = check_array(self.P2, "csc", dtype=[np.float64, np.float32])

        if sparse.issparse(self.X):
            if not sparse.isspmatrix_csc(self.P2):
                raise ValueError(
                    "If X is sparse, P2 must also be sparse csc"
                    "format. Got P2 not sparse."
                )

        self.random_state = check_random_state(self.random_state)


class IRLSState:
    def __init__(self, coef, data):
        self.data = data

        # some precalculations
        self.iteration_start = time.time()

        # number of outer iterations
        self.n_iter = 0

        # number of inner iterations (for CD, this is the number of cycles over
        # all the features)
        self.n_cycles = 0

        self.converged = False

        self.diagnostics = []

        self.coef = coef
        self.step = np.empty_like(self.coef)
        self.Fw = None
        self.eta = None
        self.mu = None
        self.score = None
        self.fisher_W = None
        self.coef_P2 = None
        self.mn_subgrad_norm = None
        self.inner_tol = None

    def record_iteration(self):
        self.n_iter += 1

        iteration_runtime = time.time() - self.iteration_start
        self.iteration_start = time.time()

        coef_l1 = np.sum(np.abs(self.coef))
        coef_l2 = np.linalg.norm(self.coef)
        step_l2 = np.linalg.norm(self.step)
        self.diagnostics.append(
            [
                self.mn_subgrad_norm,
                coef_l1,
                coef_l2,
                step_l2,
                self.n_iter,
                self.n_cycles,
                iteration_runtime,
                self.coef[0],
            ]
        )


def check_convergence(state, data):
    # stopping criterion for outer loop is a mix of a subgradient tolerance
    # and a step size tolerance
    # sum_i(|minimum-norm of subgrad of F(w)_i|)
    # fp_wP2 = f'(w) + w*P2
    # Note: eta, mu and score are already updated
    # this also updates the inner tolerance for the next loop!

    # L1 norm of the minimum norm subgradient
    mn_subgrad_norm = _norm_min_subgrad(
        state.coef, -state.score, state.data.P1, state.data.intercept_offset
    )
    gradient_converged = (
        state.data.gradient_tol is not None
        and mn_subgrad_norm < state.data.gradient_tol
    )

    # Simple L2 step length convergence criteria
    step_size = linalg.norm(state.step)
    step_size_converged = (
        state.data.step_size_tol is not None and step_size < state.data.step_size_tol
    )

    converged = gradient_converged or step_size_converged

    # the ratio of inner tolerance to the minimum subgradient norm
    # This wasn't explored in the newGLMNET paper linked above.
    # That paper essentially uses inner_tol_ratio = 1.0, but using a slightly
    # lower value is much faster.
    # By comparison, the original GLMNET paper uses inner_tol = tol.
    # So, inner_tol_ratio < 1 is sort of a compromise between the two papers.
    # The value should probably be between 0.01 and 0.5. 0.1 works well for many problems
    inner_tol_ratio = 0.1

    if state.data.fixed_inner_tol is None:
        # Another potential rule limits the inner tol to be no smaller than tol
        # return max(mn_subgrad_norm * inner_tol_ratio, tol)
        inner_tol = mn_subgrad_norm * inner_tol_ratio
    else:
        inner_tol = state.data.fixed_inner_tol[0]

    return converged, mn_subgrad_norm, inner_tol


def update_quadratic(state, data):
    eta, mu, score, fisher_W = data.family._eta_mu_score_fisher(
        coef=state.coef,
        phi=1,
        X=data.X,
        y=data.y,
        weights=data.weights,
        link=data.link,
        offset=data.offset,
        eta=state.eta,
        mu=state.mu,
    )
    coef_P2 = make_coef_P2(data, state.coef)
    score -= coef_P2
    return eta, mu, score, fisher_W, coef_P2


def make_coef_P2(data, coef):
    out = np.empty_like(coef)

    if data.intercept_offset == 1:
        out[0] = 0

    C = coef[data.intercept_offset :]
    if data.P2.ndim == 1:
        out[data.intercept_offset :] = C * data.P2
    else:
        out[data.intercept_offset :] = C @ data.P2

    return out


def line_search(state, data, d):
    # line search parameters
    (beta, sigma) = (0.5, 0.01)

    # line search by sequence beta^k, k=0, 1, ..
    # F(w + lambda d) - F(w) <= lambda * bound
    # bound = sigma * (f'(w)*d + w*P2*d
    #                  +||P1 (w+d)||_1 - ||P1 w||_1)
    P1w_1 = linalg.norm(data.P1 * state.coef[data.intercept_offset :], ord=1)
    P1wd_1 = linalg.norm(data.P1 * (state.coef + d)[data.intercept_offset :], ord=1)
    # Note: coef_P2 already calculated and still valid
    bound = sigma * (-(state.score @ d) + P1wd_1 - P1w_1)

    # In the first iteration, we must compute Fw explicitly.
    # In later iterations, we just use Fwd from the previous iteration
    # as set after the line search loop below.
    if state.Fw is None:
        Fw = (
            0.5 * data.family.deviance(data.y, state.mu, data.weights)
            + 0.5 * (state.coef_P2 @ state.coef)
            + P1w_1
        )
    else:
        Fw = state.Fw

    la = 1.0 / beta

    # TODO: if we keep track of X_dot_coef, we can add this to avoid a
    # _safe_lin_pred in _eta_mu_score_fisher every loop
    X_dot_d = _safe_lin_pred(data.X, d)

    # Try progressively shorter line search steps.
    for k in range(20):
        la *= beta  # starts with la=1
        coef_wd = state.coef + la * d

        # The simple version of the next line is:
        # mu_wd = link.inverse(_safe_lin_pred(X, coef_wd))
        # but because coef_wd can be factored as
        # coef_wd = coef + la * d
        # we can rewrite to only perform one dot product with the data
        # matrix per loop which is substantially faster
        eta_wd = state.eta + la * X_dot_d
        mu_wd = data.link.inverse(eta_wd)

        # TODO - optimize: for Tweedie that isn't one of the special cases
        # (gaussian, poisson, gamma), family.deviance is quite slow! Can we
        # fix that somehow?
        Fwd = 0.5 * data.family.deviance(data.y, mu_wd, data.weights) + linalg.norm(
            data.P1 * coef_wd[data.intercept_offset :], ord=1
        )
        coef_wd_P2 = make_coef_P2(data, coef_wd)
        Fwd += 0.5 * (coef_wd_P2 @ coef_wd)
        if Fwd - Fw <= sigma * la * bound:
            break

    # Fw in the next iteration will be equal to Fwd this iteration.
    Fw = Fwd

    # update coefficients
    step = la * d

    # We can avoid a matrix-vector product inside _eta_mu_score_fisher by
    # returning the new eta and mu calculated here.
    # NOTE: This might accumulate some numerical error over a sufficient
    # number of iterations, maybe we should completely recompute eta every
    # N iterations?
    return state.coef + step, step, Fwd, eta_wd, mu_wd


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
