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
    if data.has_lower_bounds or data.has_upper_bounds:
        raise ValueError("Bounds are not supported with the least squares solver.")
    hessian = build_hessian(data.X, state.hessian_rows, data.fit_intercept, data.P2)

    # TODO: In cases where we have lots of columns, we might want to avoid the
    # sandwich product and use something like iterative lsqr or lsmr.
    d = linalg.solve(
        hessian, state.score, overwrite_a=True, overwrite_b=True, assume_a="pos"
    )
    return d, 1


def _cd_solver(state, data):
    hessian = build_hessian(data.X, state.hessian_rows, data.fit_intercept, data.P2)
    new_coef, gap, _, _, n_cycles = enet_coordinate_descent_gram(
        state.active_set,
        state.coef.copy(),
        data.P1,
        hessian,
        -state.score,
        data.max_inner_iter,
        state.inner_tol,
        data.random_state,
        data.fit_intercept,
        data.selection == "random",
        data.has_lower_bounds,
        data._lower_bounds,
        data.has_upper_bounds,
        data._upper_bounds,
    )
    return new_coef - state.coef, n_cycles


def build_hessian(X, hessian_rows, intercept, P2):
    idx = 1 if intercept else 0
    hessian = _safe_sandwich_dot(X, hessian_rows, intercept)
    if P2.ndim == 1:
        idiag = np.arange(start=idx, stop=hessian.shape[0])
        hessian[(idiag, idiag)] += P2
    else:
        if sparse.issparse(P2):
            hessian[idx:, idx:] += P2.toarray()
        else:
            hessian[idx:, idx:] += P2
    return hessian


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
    Note: Use hessian matrix instead of Hessian for H.
    Note: f' = -score, H = hessian matrix

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

    References
    ----------

    Guo-Xun Yuan, Chia-Hua Ho, Chih-Jen Lin
    An Improved GLMNET for L1-regularized Logistic Regression,
    Journal of Machine Learning Research 13 (2012) 1999-2030
    https://www.csie.ntu.edu.tw/~cjlin/papers/l1_glmnet/long-glmnet.pdf
    """

    state = IRLSState(coef, data)

    state.eta, state.mu, state.obj_val, coef_P2 = update_predictions(
        state, data, state.coef
    )
    state.score, state.hessian_rows = update_quadratic(state, data, coef_P2)
    (
        state.converged,
        state.norm_min_subgrad,
        state.max_min_subgrad,
        state.inner_tol,
    ) = check_convergence(state, data)

    state.record_iteration()

    while state.n_iter < data.max_iter and not state.converged:

        state.active_set = identify_active_set(state, data)

        # 1) Solve the L1 and L2 penalized least squares problem
        d, n_cycles_this_iter = inner_solver(state, data)
        state.n_cycles += n_cycles_this_iter

        # 2) Line search
        (
            state.coef,
            state.step,
            state.eta,
            state.mu,
            state.obj_val,
            coef_P2,
        ) = line_search(state, data, d)

        # 3) Update the quadratic approximation
        state.score, state.hessian_rows = update_quadratic(state, data, coef_P2)

        # 4) Check if we've converged
        (
            state.converged,
            state.norm_min_subgrad,
            state.max_min_subgrad,
            state.inner_tol,
        ) = check_convergence(state, data)
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
        offset: Optional[np.ndarray] = None,
        lower_bounds: Optional[np.ndarray] = None,
        upper_bounds: Optional[np.ndarray] = None,
    ):
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
        self.has_lower_bounds, self._lower_bounds = setup_bounds(
            lower_bounds, self.X.dtype
        )
        self.has_upper_bounds, self._upper_bounds = setup_bounds(
            upper_bounds, self.X.dtype
        )

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


def setup_bounds(bounds, dtype):
    _out_bounds = bounds
    if _out_bounds is None:
        _out_bounds = np.array([], dtype=dtype)
    return bounds is not None, _out_bounds


class IRLSState:
    def __init__(self, coef, data):
        self.data = data

        # some precalculations
        self.iteration_start = time.time()

        # number of outer iterations
        self.n_iter = -1

        # number of inner iterations (for CD, this is the number of cycles over
        # all the features)
        self.n_cycles = 0

        self.converged = False

        self.diagnostics = []

        self.coef = coef

        # We need to have an initial step value to make sure that the step size
        # convergence criteria fails on the first pass
        initial_step = data.step_size_tol
        if initial_step is None:
            initial_step = 0.0
        self.step = np.full_like(self.coef, initial_step)

        self.obj_val = None
        self.eta = np.zeros(data.X.shape[0], dtype=data.X.dtype)
        self.mu = None
        self.score = None
        self.hessian_rows = None
        self.coef_P2 = None
        self.norm_min_subgrad = None
        self.max_min_subgrad = None
        self.inner_tol = None
        self.active_set = np.arange(self.coef.shape[0])

    def record_iteration(self):
        self.n_iter += 1

        iteration_runtime = time.time() - self.iteration_start
        self.iteration_start = time.time()

        coef_l1 = np.sum(np.abs(self.coef))
        coef_l2 = np.linalg.norm(self.coef)
        step_l2 = np.linalg.norm(self.step)
        self.diagnostics.append(
            {
                "convergence": self.norm_min_subgrad,
                "L1(coef)": coef_l1,
                "L2(coef)": coef_l2,
                "L2(step)": step_l2,
                "n_active": self.active_set.shape[0],
                "n_iter": self.n_iter,
                "n_cycles": self.n_cycles,
                "runtime": iteration_runtime,
                "intercept": self.coef[0],
            }
        )


def check_convergence(state, data):
    # stopping criterion for outer loop is a mix of a subgradient tolerance
    # and a step size tolerance
    # sum_i(|minimum-norm of subgrad of F(w)_i|)
    # fp_wP2 = f'(w) + w*P2
    # Note: eta, mu and score are already updated
    # this also updates the inner tolerance for the next loop!

    # L1 norm of the minimum norm subgradient
    norm_min_subgrad, max_min_subgrad = _norm_min_subgrad(
        np.arange(state.coef.shape[0], dtype=np.int32),
        state.coef,
        -state.score,
        state.data.P1,
        state.data.intercept_offset,
        data.has_lower_bounds,
        data._lower_bounds,
        data.has_upper_bounds,
        data._upper_bounds,
    )
    gradient_converged = (
        state.data.gradient_tol is not None
        and norm_min_subgrad < state.data.gradient_tol
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
        # return max(norm_min_subgrad * inner_tol_ratio, tol)
        inner_tol = norm_min_subgrad * inner_tol_ratio
    else:
        inner_tol = state.data.fixed_inner_tol[0]

    return converged, norm_min_subgrad, max_min_subgrad, inner_tol


def update_predictions(state, data, coef, X_dot_step=None, factor=1.0):
    if X_dot_step is None:
        X_dot_step = _safe_lin_pred(data.X, coef, data.offset)

    eta, mu, loglikelihood = data.family.eta_mu_loglikelihood(
        data.link, factor, state.eta, X_dot_step, data.y, data.weights
    )
    obj_val = 0.5 * loglikelihood
    obj_val += linalg.norm(data.P1 * coef[data.intercept_offset :], ord=1)
    coef_P2 = make_coef_P2(data, coef)
    obj_val += 0.5 * (coef_P2 @ coef)
    return eta, mu, obj_val, coef_P2


def update_quadratic(state, data, coef_P2):
    gradient_rows, hessian_rows = data.family.rowwise_gradient_hessian(
        data.link,
        coef=state.coef,
        phi=1,
        X=data.X,
        y=data.y,
        weights=data.weights,
        offset=data.offset,
        eta=state.eta,
        mu=state.mu,
    )

    grad = gradient_rows @ data.X
    if data.fit_intercept:
        grad = np.concatenate(([gradient_rows.sum()], grad))
    grad -= coef_P2
    return grad, hessian_rows


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


def identify_active_set(state, data):
    # This criteria is from section 5.3 of:
    # An Improved GLMNET for L1-regularized LogisticRegression.
    # Yuan, Ho, Lin. 2012
    # https://www.csie.ntu.edu.tw/~cjlin/papers/l1_glmnet/long-glmnet.pdf
    T = data.P1 - state.max_min_subgrad
    abs_score = np.abs(state.score[data.intercept_offset :])
    active = abs_score >= T

    active_set = np.concatenate(
        ([0] if data.fit_intercept else [], np.where(active)[0] + data.intercept_offset)
    ).astype(np.int32)

    return active_set


def line_search(state, data, d):
    # line search parameters
    (beta, sigma) = (0.5, 0.01)

    # line search by sequence beta^k, k=0, 1, ..
    # F(w + lambda d) - F(w) <= lambda * bound
    # bound = sigma * (f'(w)*d + w*P2*d
    #                  +||P1 (w+d)||_1 - ||P1 w||_1)
    # This is a standard Armijo-Goldstein backtracking line search algorithm:
    # https://en.wikipedia.org/wiki/Backtracking_line_search
    P1w_1 = linalg.norm(data.P1 * state.coef[data.intercept_offset :], ord=1)
    P1wd_1 = linalg.norm(data.P1 * (state.coef + d)[data.intercept_offset :], ord=1)

    # Note: the L2 penalty term is included in the score.
    bound = sigma * (-(state.score @ d) + P1wd_1 - P1w_1)

    # The step direction in row space. We'll be multiplying this by varying
    # step sizes during the line search. Factoring this matrix-vector product
    # out of the inner loop improve performance a lot!
    X_dot_d = _safe_lin_pred(data.X, d)

    # Try progressively shorter line search steps.
    # variables suffixed with wd are for the new coefficient values
    factor = 1.0
    for k in range(20):
        step = factor * d
        coef_wd = state.coef + step
        eta_wd, mu_wd, obj_val_wd, coef_wd_P2 = update_predictions(
            state, data, coef_wd, X_dot_d, factor=factor
        )
        if obj_val_wd - state.obj_val <= sigma * factor * bound:
            break
        factor *= beta

    # obj_val in the next iteration will be equal to obj_val_wd this iteration.
    # We can avoid a matrix-vector product inside _eta_mu_score_hessian by
    # returning the new eta and mu calculated here.
    # NOTE: This might accumulate some numerical error over a sufficient number
    # of iterations since we aren't calculating eta or mu from scratch but
    # instead adding the delta from the previous iteration. Maybe we should
    # completely recompute eta every N iterations?
    return state.coef + step, step, eta_wd, mu_wd, obj_val_wd, coef_wd_P2


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
