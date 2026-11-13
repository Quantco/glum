import functools
import sys
import time
import warnings
from dataclasses import InitVar, dataclass
from typing import Any, Optional, Union

import numpy as np
from scipy import linalg, sparse
from scipy.optimize import SR1, LinearConstraint, fmin_l_bfgs_b, minimize
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils import check_array
from sklearn.utils.validation import check_random_state

from ._cd_fast import (
    _norm_min_subgrad,
    enet_coordinate_descent_gram,
    identify_active_rows,
)
from ._distribution import ExponentialDispersionModel, get_one_over_variance
from ._linalg import _safe_lin_pred, _safe_sandwich_dot, _solve_least_squares_tikhonov
from ._link import Link


def timeit(runtime_attr: str):
    """
    Decorate a function to compute its run time and update the \
    :class:`IRLSState` instance attribute called ``runtime_attr`` with the run \
    time of the function.

    The first argument of ``fct`` should be an :class:`IRLSState` instance.
    """

    def fct_wrap(fct):
        @functools.wraps(fct)
        def inner_fct(*args, **kwargs):
            start = time.perf_counter()
            out = fct(*args, **kwargs)
            setattr(args[0], runtime_attr, time.perf_counter() - start)
            return out

        return inner_fct

    return fct_wrap


@timeit("inner_solver_runtime")
def _least_squares_solver(state, data, hessian):
    if data.has_lower_bounds or data.has_upper_bounds:
        raise ValueError("Bounds are not supported with the least squares solver.")

    # TODO: In cases where we have lots of columns, we might want to avoid the
    # sandwich product and use something like iterative lsqr or lsmr.
    d = linalg.solve(hessian, state.score, assume_a="pos")
    return d, 1


# Number of coordinate descent cycles between two rounds of null space moves. An
# inner solve that converges within this many cycles never looks for a null space.
_NULL_MOVE_CYCLES = 20


@timeit("inner_solver_runtime")
def _cd_solver(state, data, active_hessian):
    """Minimise the quadratic model by coordinate descent.

    If coordinate descent has not converged after ``_NULL_MOVE_CYCLES`` cycles and
    the active Hessian has (near) null directions, e.g. because of full one-hot
    categoricals, coordinate descent is interleaved with exact line
    minimisations along a sparse basis of them; see :class:`_NullSpaceMoves`.
    Otherwise, coordinate descent runs as before. The only difference is that it
    can stop after ``_NULL_MOVE_CYCLES`` cycles if it has converged by then,
    because the kernel checks convergence at the end of every call.
    """
    coef = state.coef.copy()
    q = -state.score  # updated in place by the kernel and the null space moves
    tol = state.inner_tol

    def run_cd(max_iter):
        _, norm_min_subgrad, _, _, n_cycles = enet_coordinate_descent_gram(
            state.active_set,
            coef,
            data.P1,
            active_hessian,
            q,
            max_iter,
            tol,
            data.random_state,
            data.fit_intercept,
            data.selection == "random",
            data.has_lower_bounds,
            data._lower_bounds,
            data.has_upper_bounds,
            data._upper_bounds,
            np.finfo(state.coef.dtype).eps * 16,
        )
        return norm_min_subgrad <= tol, n_cycles

    # A random run would draw a new seed with every call, so it is not split.
    if data.selection == "random" or data.max_inner_iter <= _NULL_MOVE_CYCLES:
        converged, n_cycles = run_cd(data.max_inner_iter)
    else:
        converged, n_cycles = run_cd(_NULL_MOVE_CYCLES)
        if not converged:
            null_basis = _cached_null_space_basis(state, active_hessian)
            if null_basis is None:
                converged, n_more = run_cd(data.max_inner_iter - n_cycles)
                n_cycles += n_more
            else:
                moves = _NullSpaceMoves(state, data, active_hessian, null_basis)
                chunk = _NULL_MOVE_CYCLES
                while not converged and n_cycles < data.max_inner_iter:
                    before = moves.model(coef, q)
                    moves.apply(coef, q)
                    after_moves = moves.model(coef, q)
                    converged, n_more = run_cd(
                        min(chunk, data.max_inner_iter - n_cycles)
                    )
                    n_cycles += n_more
                    # Along dense, merely ill-conditioned directions, the moves
                    # can cost as much as the cycles in between without helping.
                    # Back off while they lower the model less than those cycles.
                    if before - after_moves < after_moves - moves.model(coef, q):
                        chunk *= 2
                    else:
                        chunk = _NULL_MOVE_CYCLES

    if not converged:
        warnings.warn(
            "Coordinate descent did not converge. You might want to increase the "
            f"number of iterations (currently max_inner_iter={data.max_inner_iter}).",
            ConvergenceWarning,
        )
    return coef - state.coef, n_cycles


# Directions along which the curvature of the diagonally scaled active Hessian
# is below this are handled by null space moves.
_NULL_SPACE_RTOL = 1e-3


def _null_space_basis(Q: np.ndarray) -> Optional[np.ndarray]:
    """Sparse basis of the (near) null space of the PSD matrix ``Q``.

    Works on the scaled matrix ``S = D^{-1/2} Q D^{-1/2}``, ``D = diag(Q)``, whose
    small eigenvalues are what makes coordinate descent slow. A pivoted Cholesky
    factorisation ``S[p, p] = R.T @ R`` stops once the largest remaining pivot
    falls below ``_NULL_SPACE_RTOL``. If it stops at rank ``r < n``, the columns
    of ``[-R11^{-1} R12; I]`` (in pivoted order) span the remaining directions.
    Each is supported on one non-pivot coordinate and the pivots it is
    (nearly) collinear with. For a full set of one-hot columns next to an
    intercept, that is one categorical and the intercept, which is what makes a
    one-dimensional move along it effective. A small L2 penalty makes these
    directions only nearly flat, which is why the tolerance is not at the
    rounding level.

    Returns ``None`` if there are no such directions, which is the common case
    and costs about a third of a Cholesky factorisation.
    """
    n = Q.shape[0]
    if n < 2:
        return None
    d = np.diag(Q)
    scale = np.ones_like(d)
    np.divide(1.0, np.sqrt(d), out=scale, where=d > 0)
    (pstrf,) = linalg.get_lapack_funcs(("pstrf",), (Q,))
    R, piv, rank, info = pstrf(Q * scale[:, None] * scale, tol=_NULL_SPACE_RTOL)
    if info < 0 or rank >= n:
        return None
    piv = piv - 1  # LAPACK pivots are 1-based
    basis = np.zeros((n, n - rank), dtype=Q.dtype)
    basis[piv[:rank]] = -linalg.solve_triangular(
        R[:rank, :rank], R[:rank, rank:], check_finite=False
    )
    basis[piv[rank:], np.arange(n - rank)] = 1.0
    basis = _reduced_echelon_basis(basis, order=np.argsort(d, kind="stable"))
    basis[np.abs(basis) < np.sqrt(np.finfo(Q.dtype).eps)] = 0.0
    return basis * scale[:, None]


def _reduced_echelon_basis(basis: np.ndarray, order: np.ndarray) -> np.ndarray:
    """Another basis of the same space that is the identity on ``k`` coordinates.

    The identity coordinates are picked greedily in the given ``order``,
    skipping those that would make the pivot (nearly) zero. With the
    coordinates ordered by increasing Hessian diagonal, the intercept and other
    dense columns are picked last. For full one-hot categoricals, this gives one
    vector per categorical, so that the moves along them are independent.
    """
    M = basis.T.copy()
    free = np.ones(len(M), dtype=bool)
    threshold = 1e-3 * np.abs(M).max()
    for j in order:
        if not free.any():
            break
        candidates = np.where(free, np.abs(M[:, j]), 0.0)
        i = np.argmax(candidates)
        if candidates[i] <= threshold:
            continue
        M[i] /= M[i, j]
        others = np.arange(len(M)) != i
        M[others] -= np.outer(M[others, j], M[i])
        free[i] = False
    return M.T


def _cached_null_space_basis(state, Q) -> Optional[np.ndarray]:
    """:func:`_null_space_basis`, reused while the active set does not change.

    The null space of ``X_A.T @ W @ X_A`` does not depend on the (positive)
    weights ``W``, so it usually stays the same over outer iterations with the
    same active set. Using a basis from an earlier Hessian is always safe: the
    moves minimise the model exactly along each direction, curvature included.
    """
    cache = state._null_space_cache
    if cache is not None and np.array_equal(cache[0], state.active_set):
        return cache[1]
    basis = _null_space_basis(np.asarray(Q))
    state._null_space_cache = (state.active_set.copy(), basis)
    return basis


class _NullSpaceMoves:
    """Exact moves along a basis of (near) null directions of the active Hessian.

    Along a null direction ``v`` the smooth part of the coordinate descent model
    is (nearly) flat, so coordinate descent can only creep along it, one
    coordinate at a time. Instead, each move minimises the model exactly over
    ``w + t * v``::

        phi(t) = t * q.v + 0.5 * t**2 * v.Q.v + sum_i P1_i |w_i + t v_i|

    which is a one-dimensional, convex, piecewise quadratic problem. Each move
    can only lower the model, so the moves are safe for any direction ``v``.
    Bounds are respected by restricting ``t`` to the feasible interval.
    """

    def __init__(self, state, data, Q, basis):
        a = state.active_set
        self.active_set = a
        offset = data.intercept_offset
        is_penalized = a >= offset
        P1 = np.zeros(len(a), dtype=Q.dtype)
        P1[is_penalized] = data.P1[a[is_penalized] - offset]
        lb = np.full(len(a), -np.inf)
        ub = np.full(len(a), np.inf)
        if data.has_lower_bounds:
            lb[is_penalized] = data._lower_bounds[a[is_penalized] - offset]
        if data.has_upper_bounds:
            ub[is_penalized] = data._upper_bounds[a[is_penalized] - offset]

        Q = np.asarray(Q)
        Q_basis = Q @ basis
        # Curvature at the rounding level of the diagonally scaled Hessian is
        # treated as noise, i.e. as an exact null direction.
        flat = np.sqrt(np.finfo(Q.dtype).eps)
        diag = np.diag(Q)
        self.directions = []
        for j in range(basis.shape[1]):
            nz = np.flatnonzero(basis[:, j])
            v = basis[nz, j]
            vQv = float(v @ Q_basis[nz, j])
            if vQv <= flat * (diag[nz] @ v**2):
                vQv = 0.0
            self.directions.append((nz, v, Q_basis[:, j], vQv, P1[nz], lb[nz], ub[nz]))
        self.P1 = P1
        self.coef0 = state.coef[a]
        self.score = state.score[a]

    def model(self, coef, q) -> float:
        """Value of the quadratic model, up to a constant, in ``O(n_active)``.

        With ``delta = coef - coef0``, the kernel keeps ``q = -score + Q delta``,
        so the smooth part ``-score.delta + 0.5 delta.Q.delta`` equals
        ``0.5 delta.(q - score)``.
        """
        a = self.active_set
        w = coef[a]
        return float(0.5 * (w - self.coef0) @ (q[a] - self.score) + self.P1 @ np.abs(w))

    def apply(self, coef, q):
        """Move along each direction in turn, updating ``coef`` and ``q`` in place."""
        a = self.active_set
        for nz, v, Qv, vQv, P1, lb, ub in self.directions:
            idx = a[nz]
            w = coef[idx]
            t = _l1_line_minimum(w, v, P1, float(q[idx] @ v), vQv, lb, ub)
            if t != 0.0:
                coef[idx] = w + t * v
                q[a] += t * Qv


def _l1_line_minimum(w, v, P1, gv, vQv, lb, ub) -> float:
    """Minimiser of ``phi(t) = t*gv + 0.5*vQv*t**2 + sum_i P1_i |w_i + t v_i|``
    subject to ``lb <= w + t*v <= ub``.

    All ``v_i`` must be nonzero. The point returned is the one nearest to
    ``t = 0`` where ``|phi'|`` is below a small tolerance. Along a null
    direction (``vQv = 0``), ``phi`` can be flat over an interval; jumping across
    it would not lower the model, only add a pure null space component to the
    step, which the line search then rejects on rounding. If ``phi`` is
    unbounded below, the step stops at the last breakpoint (``phi`` decreases
    all the way to it) and coordinate descent takes it from there.
    """
    # feasible interval for t; it contains 0
    lo = min(np.max(np.where(v > 0, (lb - w) / v, (ub - w) / v)), 0.0)
    hi = max(np.min(np.where(v > 0, (ub - w) / v, (lb - w) / v)), 0.0)

    # phi'(t) = gv + vQv*t + sum_i wt_i sign(t - bp_i), with breakpoints bp_i
    is_penalized = P1 > 0
    bp = -w[is_penalized] / v[is_penalized]
    wt = P1[is_penalized] * np.abs(v[is_penalized])
    total = wt.sum()
    vQv = max(vQv, 0.0)
    tie = 4096 * np.finfo(w.dtype).eps * (total + abs(gv))
    if tie == 0.0:
        return 0.0

    # one-sided derivatives at t = 0
    d_right = gv + 2 * wt[bp <= 0].sum() - total
    d_left = gv + 2 * wt[bp < 0].sum() - total
    if d_right < -tie:
        t = _first_root_right(bp, wt, gv, vQv, total, d_right, tie, hi)
    elif d_left > tie:
        t = -_first_root_right(-bp, wt, -gv, vQv, total, -d_left, tie, -lo)
    else:
        return 0.0
    return float(np.clip(t, lo, hi))


def _first_root_right(bp, wt, gv, vQv, total, d_right0, tie, t_max) -> float:
    """Smallest ``0 < t <= t_max`` with ``phi'(t) >= -tie`` (or ``t_max``), given
    that ``phi'(0+) < -tie``."""
    pos = (bp > 0) & (bp < t_max)
    bp, wt = bp[pos], wt[pos]
    order = np.argsort(bp)
    bp, wt = bp[order], wt[order]
    # constant part of phi' just left and just right of each breakpoint
    base_left = d_right0 - gv + 2 * np.concatenate(([0.0], np.cumsum(wt)[:-1]))
    base_right = base_left + 2 * wt
    hit = np.flatnonzero(gv + vQv * bp + base_right >= -tie)
    if len(hit):
        k = hit[0]
        base, t_end = base_left[k], bp[k]
    else:
        base = base_right[-1] if len(bp) else d_right0 - gv
        t_end = t_max
    if vQv > 0:
        # phi' may reach zero before t_end
        t_end = min(t_end, (-tie - gv - base) / vQv)
    if np.isinf(t_end):  # unbounded below
        return float(bp[-1]) if len(bp) else 0.0
    return float(t_end)


@timeit("build_hessian_runtime")
def update_hessian(state, data, active_set):
    """
    Update the approximate Hessian.

    The goal is to compute ``H = X^T @ diag(hessian_rows) @ X``. We will refer
    to ``H`` as the Hessian, even though technically we are computing a
    Gauss-Newton approximation to the Hessian.

    Instead of computing ``H`` directly, we will compute updates to ``H``:
    ``dH``. So, given ``H0`` from a previous iteration,
    ``H0 = X^T @ diag(hessian_rows_0) @ X``, we want to compute ``H1`` from this
    iteration: ``H1 = X^T @ diag(hessian_rows_1) @ X``.

    However, we will instead compute ``H1 = H0 + dH``, where
    ``dH = X^T @ diag(hessian_rows_1 - hessian_rows_0) @ X``.

    We will also refer to
    ``hessian_rows_diff = hessian_rows_1 - hessian_rows_0``.

    The advantage of reframing the computation of ``H`` as an update is that the
    values in ``hessian_rows_diff`` will vary depending on how large the
    influence of that last coefficient update was on that row. As a result, in
    lots of problems, many of the entries in ``hessian_rows_diff`` will be very
    very small.

    So, the goal with ``identify_active_rows`` is to filter to a subset of
    ``hessian_rows_diff`` that we will use to compute the sandwich product for
    ``dH``. If ``threshold/data.hessian_approx == 0.0``, then we will always use
    every row. However, for ``data.hessian_approx != 0``, we include rows for
    which
    ``include = (np.abs(hessian_rows_diff[i]) >= T *
    np.max(np.abs(hessian_rows_diff)))``.

    Essentially, this criterion ignores data matrix rows that have not seen the
    second derivatives of their predictions change very much in the last
    iteration.

    Critically, we set
    ``hessian_rows_old[include] += hessian_rows_diff[include]``.
    That way, ``hessian_rows_diff`` is no longer the change since the last
    iteration, but, instead, the change since the last iteration that a row was
    active. This ensures that we don't miss the situation where a row changes a
    small amount over several iterations which accumulates into a large change.
    """
    # The simplest, but expensive option is that if the active set increases in
    # size, we can just recompute the full hessian for the new active set
    # instead of an approximate update. This is what is currently implemented
    # here.
    #
    # Another interesting option would be to keep track of two old sets of
    # hessian_rows: one for the full dataset and one for the current reduced
    # dataset. Restart from the full dataset version if we increase the size of
    # the active_set. Though expensive, that would be a rare operation.
    #
    # Third thing: we could have a flag that swaps between computing H and
    # delta H. If we just computed H directly, then this wouldn't be an issue.
    # A slight modification of this: we could still use a baseline H from the
    # first iteration with the entire column set.
    first_iteration = not state.hessian_initialized
    reset_iteration = not _is_subset(state.old_active_set, active_set)
    if first_iteration or reset_iteration:
        # In the first iteration or in a reset iteration, we need to:
        # 1) use hessian_rows, not the difference
        # 2) use all the rows
        # 3) Include the P2 components
        # 4) just like an update, we only update the active_set
        hessian_init = build_hessian_delta(
            data.X,
            state.hessian_rows,
            data.fit_intercept,
            data.P2,
            np.arange(data.X.shape[0], dtype=np.int32),
            active_set,
        )
        state.hessian[np.ix_(active_set, active_set)] = hessian_init
        state.hessian_initialized = True
        n_active_rows = data.X.shape[0]
    else:
        # In an update iteration, we want to:
        # 1) use the difference in hessian_rows from the last iteration
        # 2) filter for active_rows in case data.hessian_approx != 0
        # 3) Ignore the P2 components because those don't change and have
        #    already been added
        # 4) only update the active set subset of the hessian.
        hessian_rows_diff, active_rows = identify_active_rows(
            state.gradient_rows,
            state.hessian_rows,
            state.old_hessian_rows,
            data.hessian_approx,
        )
        hessian_delta = build_hessian_delta(
            data.X,
            hessian_rows_diff,
            data.fit_intercept,
            P2=None,
            active_rows=active_rows,
            active_cols=active_set,
        )
        state.hessian[np.ix_(active_set, active_set)] += hessian_delta
        n_active_rows = active_rows.shape[0]

    return (
        state.hessian[np.ix_(active_set, active_set)],
        n_active_rows,
    )


def _is_subset(x, y):
    # NOTE: This functions assumes entries in x and y are unique
    intersection = np.intersect1d(x, y)
    return intersection.size == y.size


def build_hessian_delta(
    X, hessian_rows, intercept, P2, active_rows, active_cols
) -> np.ndarray:
    """
    Get the Hessian "sandwich" matrix for active rows and columns.

    Parameters
    ----------
    X
    hessian_rows
    intercept
    P2
    active_rows
    active_cols
    """
    idx = 1 if intercept else 0
    active_cols_non_intercept = active_cols[idx:] - idx
    delta = _safe_sandwich_dot(
        X, hessian_rows, active_rows, active_cols_non_intercept, intercept
    )
    if P2 is None:
        return delta

    if sparse.issparse(P2) and P2.nnz == 0:
        return delta

    if P2.ndim == 1:
        idiag = np.arange(idx, int(delta.shape[0]))
        delta[(idiag, idiag)] += P2[active_cols_non_intercept]
    else:
        if sparse.issparse(P2):
            is_diagonal = P2.nnz == P2.shape[0] and (P2.data == P2.diagonal()).all()
            if is_diagonal:
                idiag = np.arange(idx, int(delta.shape[0]))
                delta[(idiag, idiag)] += P2.data[active_cols_non_intercept]
            else:
                delta[idx:, idx:] += P2.toarray()[
                    np.ix_(active_cols_non_intercept, active_cols_non_intercept)
                ]
        else:
            delta[idx:, idx:] += P2[
                np.ix_(active_cols_non_intercept, active_cols_non_intercept)
            ]
    return delta


def _irls_solver(inner_solver, coef, data) -> tuple[np.ndarray, int, int, list[list]]:
    """
    Solve GLM with L1 and L2 penalty by IRLS.

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

    state.eta, state.mu, state.obj_val, coef_P2 = _update_predictions(
        state, data, state.coef
    )
    state.gradient_rows, state.score, state.hessian_rows = update_quadratic(
        state, data, coef_P2
    )
    (
        state.converged,
        state.norm_min_subgrad,
        state.max_min_subgrad,
        state.inner_tol,
    ) = check_convergence(state, data)

    state._record_iteration()

    with ProgressBar(state.norm_min_subgrad, data.gradient_tol, data.verbose) as pb:
        while state.n_iter < data.max_iter and not state.converged:
            pb._update(state.n_iter, state.iteration_runtime, state.norm_min_subgrad)

            state.old_active_set = state.active_set
            state.active_set = identify_active_set(state, data)

            # 0) Build the hessian
            hessian, state.n_active_rows = update_hessian(state, data, state.active_set)

            # 1) Solve the L1 and L2 penalized least squares problem
            d, n_cycles_this_iter = inner_solver(state, data, hessian)
            state.n_cycles += n_cycles_this_iter

            # 2) Line search
            state.old_hessian_rows[:] = state.hessian_rows
            (
                state.coef,
                state.step,
                state.eta,
                state.mu,
                state.obj_val,
                coef_P2,
            ) = line_search(state, data, d)
            state.n_updated = np.sum(np.abs(d) > 0)

            # 3) Update the quadratic approximation
            state.gradient_rows, state.score, state.hessian_rows = update_quadratic(
                state, data, coef_P2
            )

            # 4) Check if we've converged
            (
                state.converged,
                state.norm_min_subgrad,
                state.max_min_subgrad,
                state.inner_tol,
            ) = check_convergence(state, data)

            state._record_iteration()

    if not state.converged:
        warnings.warn(
            "IRLS failed to converge. Increase"
            " the maximum number of iterations max_iter"
            f" (currently {data.max_iter})",
            ConvergenceWarning,
        )
    return state.coef, state.n_iter, state.n_cycles, state.diagnostics


def _tikhonov_solver(
    X,
    y: np.ndarray,
    sample_weight: np.ndarray,
    P2,
    fit_intercept: bool,
    random_state: Union[None, int, np.random.RandomState],
    offset: Optional[np.ndarray] = None,
    verbose: int = 0,
) -> tuple[np.ndarray, int, int, None]:
    """Direct Gaussian identity-link solve for ridge/OLS/WLS."""
    # Preserve sklearn-style random_state validation side effect even though
    # this solver path is deterministic.
    check_random_state(random_state)
    coef = _solve_least_squares_tikhonov(
        X=X,
        y=y,
        sample_weight=sample_weight,
        P2=P2,
        fit_intercept=fit_intercept,
        offset=offset,
    )
    if verbose > 0:
        print("Iteration 1: closed-form Gaussian solve.", file=sys.stderr)
    return coef, 1, 1, None


class ProgressBar:
    """
    Display the current progress of the solver using tqdm.

    Current format is that:
        * The "Iteration #" is the IRLS iteration #.
        * The bar itself measures the log base 10 progress towards the
            gradient_tol. So, if we start at gradient_norm = 10 and are going to
            gradient_norm = 1e-4, the progress bar will be out of 5. If the
            current gradient_norm is 0.1, then we will be at 2/5.
        * On the right, show the time for the most recent iteration and the
            current gradient norm.
    """

    def __init__(self, start_norm, tol, verbose):
        self.start_norm = start_norm
        self.tol = tol
        self.verbose = verbose

    def __enter__(self):
        """Run the tqdm progress bar."""
        if not self.verbose:
            return self
        bar_start_loggrad = np.log10(self.start_norm)
        bar_end_loggrad = np.log10(self.tol)
        self.n_bar_steps = np.ceil(bar_start_loggrad - bar_end_loggrad)
        # Wait to import so that if verbose=False, we don't need tqdm installed
        from tqdm import tqdm

        self.t = tqdm(
            total=self.n_bar_steps,
            bar_format="Iteration {postfix[0]}: {l_bar}{bar}| {n_fmt}/{total_fmt} [{pos"
            "tfix[1]}s/it, gradient norm={postfix[2]}]",
            postfix=[0, "", self.start_norm],
        )
        return self

    def __exit__(self, *exc):
        """Stop tracking progress."""
        if self.verbose:
            self.t.close()

    def _update(self, n_iter, iteration_runtime, cur_grad_norm):
        if not self.verbose:
            return
        self.t.postfix[0] = n_iter
        self.t.postfix[1] = f"{iteration_runtime:.2f}"
        self.t.postfix[2] = cur_grad_norm
        # clip to 0 in case we take a step in the wrong direction at the start
        # without this, tqdm will print an annoying warning.
        step = max(self.n_bar_steps - (np.log10(cur_grad_norm) - np.log10(self.tol)), 0)
        # round to two digits for beauty
        self.t.n = np.round(step, 2)
        self.t.update(0)


@dataclass
class IRLSData:
    """Store parameters for the IRLS optimizer."""

    X: Any
    y: np.ndarray
    sample_weight: np.ndarray
    # Note: we already set P2 = l2*P2, P1 = l1*P1 and symmetrized P2 = 1/2 (P2 + P2')
    P1: Union[np.ndarray, sparse.spmatrix]
    P2: Union[np.ndarray, sparse.spmatrix]
    fit_intercept: bool
    family: ExponentialDispersionModel
    link: Link
    max_iter: int = 100
    max_inner_iter: int = 100000
    gradient_tol: Optional[float] = 1e-4
    step_size_tol: Optional[float] = 1e-4
    hessian_approx: float = 0.0
    fixed_inner_tol: Optional[tuple] = None
    selection: str = "cyclic"
    random_state: Union[None, int, np.random.RandomState] = None
    offset: Optional[np.ndarray] = None
    lower_bounds: InitVar[Optional[np.ndarray]] = None
    upper_bounds: InitVar[Optional[np.ndarray]] = None
    verbose: bool = False

    def __post_init__(self, lower_bounds, upper_bounds):
        self.has_lower_bounds, self._lower_bounds = _setup_bounds(
            lower_bounds, self.X.dtype
        )
        self.has_upper_bounds, self._upper_bounds = _setup_bounds(
            upper_bounds, self.X.dtype
        )

        self.intercept_offset = 1 if self.fit_intercept else 0

        # Check data
        if self.P2.ndim == 2:
            self.P2 = check_array(self.P2, "csc", dtype=[np.float64, np.float32])

        if sparse.issparse(self.X):
            if not sparse.isspmatrix_csc(self.P2):
                raise ValueError(
                    "If X is sparse, P2 must also be sparse CSC format. It is not."
                )

        self.random_state = check_random_state(self.random_state)


def _setup_bounds(
    bounds: Optional[np.ndarray], dtype
) -> tuple[bool, Optional[np.ndarray]]:
    _out_bounds = bounds
    if _out_bounds is None:
        _out_bounds = np.array([], dtype=dtype)
    return bounds is not None, _out_bounds


class IRLSState:
    """Track many parameters, such as score and Hessian, used in the IRLS solver."""

    def __init__(self, coef, data):
        self.data = data

        # some precalculations
        self.iteration_start = time.time()
        self.iteration_runtime = 0.0

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
        self.score = np.empty_like(self.coef)
        self.old_hessian_rows = np.zeros(data.X.shape[0], dtype=data.X.dtype)
        self.gradient_rows = None
        self.hessian_rows = None
        self.hessian = np.zeros(
            (self.coef.shape[0], self.coef.shape[0]), dtype=data.X.dtype
        )
        self.hessian_initialized = False
        self.coef_P2 = None
        self.norm_min_subgrad = None
        self.max_min_subgrad = None
        self.inner_tol = None

        self.n_line_search = None

        self.n_updated = 0
        self.n_active_rows = data.X.shape[0]
        self.old_active_set = None
        self.active_set = np.arange(self.coef.shape[0], dtype=np.int32)
        # (active set, null space basis of its Hessian or None), see _cd_solver
        self._null_space_cache: Optional[tuple] = None

        self.build_hessian_runtime = None
        self.inner_solver_runtime = None
        self.line_search_runtime = None
        self.quadratic_update_runtime = None

        # used in the line-search Armijo stopping criterion
        self.large_number = 1e30 if data.X.dtype == np.float32 else 1e43

    def _record_iteration(self):
        self.n_iter += 1

        self.iteration_runtime = time.time() - self.iteration_start
        self.iteration_start = time.time()

        coef_l1 = np.sum(np.abs(self.coef))  # type: ignore
        coef_l2 = np.linalg.norm(self.coef)
        step_l2 = np.linalg.norm(self.step)
        self.diagnostics.append(
            {
                "n_iter": self.n_iter,
                "convergence": self.norm_min_subgrad,
                "objective_fct": self.obj_val,
                "L1(coef)": coef_l1,
                "L2(coef)": coef_l2,
                "L2(step)": step_l2,
                "first_coef": self.coef[0],
                "n_coef_updated": self.n_updated,
                "n_active_cols": self.active_set.shape[0],
                "n_active_rows": self.n_active_rows,
                "n_cycles": self.n_cycles,
                "n_line_search": self.n_line_search,
                "iteration_runtime": self.iteration_runtime,
                "build_hessian_runtime": self.build_hessian_runtime,
                "inner_solver_runtime": self.inner_solver_runtime,
                "line_search_runtime": self.line_search_runtime,
                "quadratic_update_runtime": self.quadratic_update_runtime,
            }
        )


def check_convergence(
    state: IRLSState, data: IRLSData
) -> tuple[bool, float, float, float]:
    """Calculate parameters needed to determine whether we have converged."""
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
    gradient_converged = norm_min_subgrad < state.data.gradient_tol

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
    # Lower values tend to increase the time spent in the inner solver while
    # reducing the time spent computing quadratic approximations. We find that
    # 0.001 is generally a good number.
    inner_tol_ratio = 0.001

    if state.data.fixed_inner_tol is None:
        # Another potential rule limits the inner tol to be no smaller than tol
        # return max(norm_min_subgrad * inner_tol_ratio, tol)
        inner_tol = max(norm_min_subgrad * inner_tol_ratio, data.gradient_tol)
    else:
        inner_tol = state.data.fixed_inner_tol[0]

    return converged, norm_min_subgrad, max_min_subgrad, inner_tol


def _update_predictions(state, data, coef, X_dot_step=None, factor=1.0):
    if X_dot_step is None:
        X_dot_step = _safe_lin_pred(data.X, coef, data.offset)
    return eta_mu_objective(
        data.family,
        data.link,
        X_dot_step,
        factor,
        coef,
        state.eta,
        data.y,
        data.sample_weight,
        data.P1,
        data.P2,
        data.intercept_offset,
    )


def eta_mu_objective(
    family,
    link,
    X_dot_step,
    factor,
    coef,
    cur_eta,
    y,
    sample_weight,
    P1,
    P2,
    intercept_offset,
):
    """Calculate eta, mu, and the objective value."""
    eta, mu, deviance = family.eta_mu_deviance(
        link, factor, cur_eta, X_dot_step, y, sample_weight
    )
    obj_val = 0.5 * deviance
    obj_val += linalg.norm(P1 * coef[intercept_offset:], ord=1)
    coef_P2 = _make_coef_P2(intercept_offset, P2, coef)
    obj_val += 0.5 * (coef_P2 @ coef)
    return eta, mu, obj_val, coef_P2


@timeit("quadratic_update_runtime")
def update_quadratic(
    state: IRLSState, data: IRLSData, coef_P2
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Update the quadratic approximation."""
    gradient_rows, hessian_rows = data.family.rowwise_gradient_hessian(
        data.link,
        coef=state.coef,
        dispersion=1,
        X=data.X,
        y=data.y,
        sample_weight=data.sample_weight,
        offset=data.offset,
        eta=state.eta,
        mu=state.mu,
    )

    grad = gradient_rows @ data.X
    if data.fit_intercept:
        grad = np.concatenate(([gradient_rows.sum()], grad))
    grad -= coef_P2
    return gradient_rows, grad, hessian_rows


def _make_coef_P2(intercept_offset, P2, coef):
    out = np.empty_like(coef)

    if intercept_offset == 1:
        out[0] = 0

    C = coef[intercept_offset:]
    if P2.ndim == 1:
        out[intercept_offset:] = C * P2
    else:
        out[intercept_offset:] = C @ P2

    return out


def identify_active_set(state: IRLSState, data: IRLSData):
    """
    Find the coefficients in the active set for this iteration.

    This criteria is from section 5.3 of:
    An Improved GLMNET for L1-regularized LogisticRegression.
    Yuan, Ho, Lin. 2012
    https://www.csie.ntu.edu.tw/~cjlin/papers/l1_glmnet/long-glmnet.pdf

    Parameters
    ----------
    state
    data

    Returns
    -------
    np.ndarray
    """
    T = data.P1 - state.max_min_subgrad
    abs_score = np.abs(state.score[data.intercept_offset :])
    active = np.logical_or(state.coef[data.intercept_offset :] != 0, abs_score >= T)

    active_set = np.concatenate(
        [[0] if data.fit_intercept else [], np.where(active)[0] + data.intercept_offset]
    ).astype(np.int32)

    return active_set


@timeit("line_search_runtime")
def line_search(state: IRLSState, data: IRLSData, d: np.ndarray):
    """
    Run a backtracking line search.

    Parameters
    ----------
    state
    data
    d

    Returns
    -------
    tuple
    """
    # line search parameters
    (beta, sigma) = (0.5, 0.0001)
    # Use np.finfo(state.coef.dtype).eps instead np.finfo(state.obj_val), as
    # state.obj_val is np.float64, even if the data is np.float32.
    eps = 16 * np.finfo(state.coef.dtype).eps  # type: ignore

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
    for _k in range(20):
        step = factor * d
        coef_wd = state.coef + step
        eta_wd, mu_wd, obj_val_wd, coef_wd_P2 = _update_predictions(
            state, data, coef_wd, X_dot_d, factor=factor
        )
        # 1. Check Armijo / sufficient decrease condition.
        loss_improvement = obj_val_wd - state.obj_val

        if mu_wd.max() < state.large_number and loss_improvement <= factor * bound:
            break
        # 2. Deal with relative loss differences around machine precision.
        tiny_loss = np.abs(state.obj_val * eps)  # type: ignore
        if np.abs(loss_improvement) <= tiny_loss:
            # 2.1 Check the L1 norm of the minimum norm subgradient (the
            # convergence criterion) as alternative condition. With an L1
            # penalty or bounds, the plain gradient does not vanish at the
            # optimum, so its norm says nothing about progress there.
            # Therefore, we need the recent gradient, see update_quadratic.
            sigma_inv = get_one_over_variance(
                data.family, data.link, mu_wd, eta_wd, 1.0, data.sample_weight
            )
            d1 = data.link.inverse_derivative(eta_wd)  # = h'(eta)
            d1_sigma_inv = d1 * sigma_inv
            gradient_rows = d1_sigma_inv * (data.y - mu_wd)
            grad = gradient_rows @ data.X
            if data.fit_intercept:
                grad = np.concatenate(([gradient_rows.sum()], grad))
            grad -= coef_wd_P2
            sum_abs_grad, _ = _norm_min_subgrad(
                np.arange(coef_wd.shape[0], dtype=np.int32),
                coef_wd,
                -grad,
                data.P1,
                data.intercept_offset,
                data.has_lower_bounds,
                data._lower_bounds,
                data.has_upper_bounds,
                data._upper_bounds,
            )
            if sum_abs_grad < state.norm_min_subgrad:
                break
        factor *= beta
    else:
        # If the loss no longer changes up to machine precision, the iterate is
        # converged to within numerical precision and there is nothing to warn
        # about. Only warn if the loss is still moving.
        if np.abs(loss_improvement) > tiny_loss:
            warnings.warn(
                "Line search failed. Next iteration will be very close to current "
                "iteration. Might result in more convergence issues.",
                ConvergenceWarning,
            )

    state.n_line_search = _k

    # obj_val in the next iteration will be equal to obj_val_wd this iteration.
    # We can avoid a matrix-vector product inside _eta_mu_score_hessian by
    # returning the new eta and mu calculated here.
    # NOTE: This might accumulate some numerical error over a sufficient number
    # of iterations since we aren't calculating eta or mu from scratch but
    # instead adding the delta from the previous iteration. Maybe we should
    # completely recompute eta every N iterations?
    return state.coef + step, step, eta_wd, mu_wd, obj_val_wd, coef_wd_P2


def _get_obj_and_derivative(
    coef,
    X,
    y: np.ndarray,
    sample_weight: np.ndarray,
    P2: Union[np.ndarray, sparse.spmatrix],
    family: ExponentialDispersionModel,
    link: Link,
    offset: np.ndarray = None,
):
    mu, devp = family._mu_deviance_derivative(coef, X, y, sample_weight, link, offset)
    dev = family.deviance(y, mu, sample_weight)
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


def _lbfgs_solver(
    coef,
    X,
    y: np.ndarray,
    sample_weight: np.ndarray,
    P2: Union[np.ndarray, sparse.spmatrix],
    verbose: bool,
    family: ExponentialDispersionModel,
    link: Link,
    max_iter: int = 100,
    tol: float = 1e-4,
    offset: np.ndarray = None,
):
    func = functools.partial(
        _get_obj_and_derivative,
        X=X,
        y=y,
        sample_weight=sample_weight,
        P2=P2,
        family=family,
        link=link,
        offset=offset,
    )

    coef, loss, info = fmin_l_bfgs_b(
        func,
        coef,
        fprime=None,
        pgtol=tol,
        maxiter=max_iter,
        factr=1e2,
    )
    if info["warnflag"] == 1:
        warnings.warn(
            "lbfgs failed to converge. Increase the number of iterations.",
            ConvergenceWarning,
        )
    elif info["warnflag"] == 2:
        warnings.warn("lbfgs failed for the reason: {}".format(info["task"]))
    n_iter_ = info["nit"]

    return coef, n_iter_, -1, None


def _trust_constr_solver(
    coef,
    X,
    y: np.ndarray,
    sample_weight: np.ndarray,
    P2: Union[np.ndarray, sparse.spmatrix],
    fit_intercept: bool,
    verbose: bool,
    family: ExponentialDispersionModel,
    link: Link,
    max_iter: int = 100,
    xtol: Optional[float] = 1e-8,
    gtol: Optional[float] = 1e-8,
    offset: np.ndarray = None,
    A_ineq: Optional[np.ndarray] = None,
    b_ineq: Optional[np.ndarray] = None,
):
    fun = functools.partial(
        _get_obj_and_derivative,
        X=X,
        y=y,
        sample_weight=sample_weight,
        P2=P2,
        family=family,
        link=link,
        offset=offset,
    )

    if (A_ineq is not None) and (b_ineq is not None):
        if fit_intercept:
            # add one column of 0's from the left
            # the intercept will not be constrained
            A_ineq_intercept = np.zeros(shape=(A_ineq.shape[0], 1))
            A_ineq_ = np.concatenate((A_ineq_intercept, A_ineq), axis=1)
        else:
            A_ineq_ = A_ineq

        # we express constraints in the form A theta <= b
        constraints = LinearConstraint(
            A=A_ineq_,
            lb=-np.inf,
            ub=b_ineq,
        )
    else:
        constraints = ()

    res = minimize(
        fun=fun,
        x0=coef,
        jac=True,
        method="trust-constr",
        hess=SR1(),
        constraints=constraints,
        options={
            "xtol": xtol,
            "gtol": gtol,
            "maxiter": max_iter,
            "verbose": 2 if verbose else 0,
        },
    )

    if not res["success"]:
        warnings.warn(
            f"trust-constr failed with message: {res['message']}",
            ConvergenceWarning,
        )

    return res["x"], res["nit"], -1, None


def _build_monotonic_penalty(
    coef_no_intercept: np.ndarray,
    A_ineq: np.ndarray,
    b_ineq: np.ndarray,
    constraint_lam: float = 1e9,
) -> Optional[sparse.csc_matrix]:
    """Return ``lam * A_v.T @ A_v`` as sparse, or None if no violations."""
    violations = A_ineq @ coef_no_intercept - b_ineq
    violated = violations > 0
    if not np.any(violated):
        return None
    A_v = sparse.csc_matrix(A_ineq[violated])
    return constraint_lam * (A_v.T @ A_v)


def _monotonic_irls_solver(
    coef,
    data: IRLSData,
    A_ineq: np.ndarray,
    b_ineq: np.ndarray,
    constraint_lam_start: float = 1e5,
    constraint_lam_max: float = 1e9,
    constraint_lam_factor: float = 100.0,
) -> tuple[np.ndarray, int, int, list[list]]:
    """IRLS with progressive quadratic penalty for monotonic constraints."""
    if np.any(b_ineq != 0):
        raise NotImplementedError(
            "irls-ls-monotonic only supports b_ineq=0. "
            "Use solver='trust-constr' for general A_ineq @ w <= b_ineq."
        )

    idx = 1 if data.fit_intercept else 0
    constraint_lam = constraint_lam_start

    P2_base = data.P2.copy()

    def _update_constraint_penalty(coef, lam) -> bool:
        """Update data.P2 with penalty; return True if P2 changed."""
        C = _build_monotonic_penalty(coef[idx:], A_ineq, b_ineq, lam)
        if C is None:
            changed = data.P2 is not P2_base
            data.P2 = P2_base
        else:
            if P2_base.ndim == 1:
                data.P2 = sparse.diags(P2_base) + C
            elif sparse.issparse(P2_base):
                data.P2 = P2_base + C
            else:
                data.P2 = P2_base + C.toarray()
            changed = True
        return changed

    _update_constraint_penalty(coef, constraint_lam)

    state = IRLSState(coef, data)
    state.eta, state.mu, state.obj_val, coef_P2 = _update_predictions(
        state, data, state.coef
    )
    state.gradient_rows, state.score, state.hessian_rows = update_quadratic(
        state, data, coef_P2
    )
    (
        _,
        state.norm_min_subgrad,
        state.max_min_subgrad,
        state.inner_tol,
    ) = check_convergence(state, data)
    state.converged = False
    state._record_iteration()

    with ProgressBar(state.norm_min_subgrad, data.gradient_tol, data.verbose) as pb:
        while state.n_iter < data.max_iter and not state.converged:
            pb._update(state.n_iter, state.iteration_runtime, state.norm_min_subgrad)

            violations = A_ineq @ state.coef[idx:] - b_ineq
            if np.any(violations > 0):
                constraint_lam = min(
                    constraint_lam * constraint_lam_factor, constraint_lam_max
                )

            old_coef_P2 = _make_coef_P2(data.intercept_offset, data.P2, state.coef)
            penalty_changed = _update_constraint_penalty(state.coef, constraint_lam)

            if penalty_changed:
                state.hessian_initialized = False
                # The objective value and score include the P2 term, so they are
                # stale after the penalty changed. The line search compares against
                # them, so bring them up to date; otherwise it can fail spuriously.
                new_coef_P2 = _make_coef_P2(data.intercept_offset, data.P2, state.coef)
                state.obj_val += 0.5 * ((new_coef_P2 - old_coef_P2) @ state.coef)
                state.score = state.score + old_coef_P2 - new_coef_P2

            state.old_active_set = state.active_set
            state.active_set = identify_active_set(state, data)

            hessian, state.n_active_rows = update_hessian(state, data, state.active_set)

            d, n_cycles_this_iter = _least_squares_solver(state, data, hessian)
            state.n_cycles += n_cycles_this_iter

            state.old_hessian_rows[:] = state.hessian_rows
            (
                state.coef,
                state.step,
                state.eta,
                state.mu,
                state.obj_val,
                coef_P2,
            ) = line_search(state, data, d)
            state.n_updated = np.sum(np.abs(d) > 0)

            state.gradient_rows, state.score, state.hessian_rows = update_quadratic(
                state, data, coef_P2
            )

            (
                state.converged,
                state.norm_min_subgrad,
                state.max_min_subgrad,
                state.inner_tol,
            ) = check_convergence(state, data)

            if state.converged:
                violations = A_ineq @ state.coef[idx:] - b_ineq
                if np.any(violations > 1e-6):
                    state.converged = False

            state._record_iteration()

    if not state.converged:
        warnings.warn(
            "IRLS failed to converge. Increase"
            " the maximum number of iterations max_iter"
            f" (currently {data.max_iter})",
            ConvergenceWarning,
        )

    max_violation = np.max(A_ineq @ state.coef[idx:] - b_ineq)
    if max_violation > 1e-6:
        warnings.warn(
            f"Monotonic IRLS: max constraint violation {max_violation:.2e} "
            f"after {state.n_iter} iterations.",
            ConvergenceWarning,
        )

    return state.coef, state.n_iter, state.n_cycles, state.diagnostics
