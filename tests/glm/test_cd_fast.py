import numpy as np
import pytest

from glum._cd_fast import enet_coordinate_descent_gram


def _reference_cd(active_set, w, P1, Q, q, n_cycles, intercept, lb, ub):
    """Cyclic coordinate descent, written out as in ``_cd_fast.pyx``."""
    w, q = w.copy(), q.copy()
    for _ in range(n_cycles):
        for a, i in enumerate(active_set):
            P1_i = 0.0 if i < intercept else P1[i - intercept]
            w_old = w[i]
            tmp = q[i] - w[i] * Q[a, a]
            w[i] = np.sign(-tmp) * max(abs(tmp) - P1_i, 0) / Q[a, a]
            if i >= intercept:
                if lb is not None:
                    w[i] = max(w[i], lb[i - intercept])
                if ub is not None:
                    w[i] = min(w[i], ub[i - intercept])
            q[active_set] += (w[i] - w_old) * Q[a]
    return w, q


def _problem(dtype, intercept, rng):
    n_full = 9
    # a non-contiguous active set; glum puts the intercept first
    active_set = np.array([0, 2, 3, 5, 8] if intercept else [1, 2, 4, 7, 8], np.int32)
    A = rng.normal(size=(40, n_full))
    if intercept:
        A[:, 0] = 1.0
    H = A.T @ A / 40 + 0.1 * np.eye(n_full)
    Q = np.ascontiguousarray(H[np.ix_(active_set, active_set)]).astype(dtype)
    w = rng.normal(scale=0.3, size=n_full).astype(dtype)
    q = rng.normal(size=n_full).astype(dtype)
    P1 = np.full(n_full - intercept, 0.2, dtype=dtype)
    return active_set, w, P1, Q, q


def _run(active_set, w, P1, Q, q, max_iter, tol, intercept, random, lb, ub):
    empty = np.array([], dtype=w.dtype)
    return enet_coordinate_descent_gram(
        active_set, w, P1, Q, q, max_iter, w.dtype.type(tol),
        np.random.RandomState(0), intercept, random,
        lb is not None, empty if lb is None else lb,
        ub is not None, empty if ub is None else ub,
        w.dtype.type(16 * np.finfo(w.dtype).eps),
    )  # fmt: skip


@pytest.mark.filterwarnings("ignore:Coordinate descent did not converge")
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("intercept", [True, False])
@pytest.mark.parametrize("bounds", [False, True])
def test_enet_cd_gram_matches_reference(dtype, intercept, bounds):
    """A fixed number of cycles on a non-contiguous active set matches the
    plain-Python update rule, and leaves inactive coordinates untouched.

    With tol=0 the kernel runs all cycles unless the iterate is exactly
    stationary (e.g. every coordinate at a bound), so the reference is run
    for the number of cycles the kernel reports.
    """
    rng = np.random.default_rng(0)
    active_set, w0, P1, Q, q0 = _problem(dtype, intercept, rng)
    n_bounded = len(P1)
    lb = np.full(n_bounded, -0.15, dtype) if bounds else None
    ub = np.full(n_bounded, 0.25, dtype) if bounds else None

    w, q = w0.copy(), q0.copy()
    w_out, *_, n_iter = _run(active_set, w, P1, Q, q, 7, 0.0, intercept, False, lb, ub)
    w_ref, q_ref = _reference_cd(active_set, w0, P1, Q, q0, n_iter, intercept, lb, ub)

    rtol = 1e-10 if dtype == np.float64 else 1e-4
    assert 1 <= n_iter <= 7
    np.testing.assert_allclose(w_out, w_ref, rtol=rtol, atol=rtol)
    np.testing.assert_allclose(q, q_ref, rtol=rtol, atol=rtol)  # q is updated in place
    inactive = np.setdiff1d(np.arange(len(w0)), active_set)
    np.testing.assert_array_equal(w[inactive], w0[inactive])
    np.testing.assert_array_equal(q[inactive], q0[inactive])


@pytest.mark.parametrize("intercept", [True, False])
def test_enet_cd_gram_random_selection_converges(intercept):
    """Random and cyclic selection converge to the same (unique) minimiser."""
    rng = np.random.default_rng(1)
    active_set, w0, P1, Q, q0 = _problem(np.float64, intercept, rng)
    w_cyclic, *_ = _run(
        active_set, w0.copy(), P1, Q, q0.copy(), 10000, 1e-12, intercept, False,
        None, None,
    )  # fmt: skip
    w_random, *_ = _run(
        active_set, w0.copy(), P1, Q, q0.copy(), 10000, 1e-12, intercept, True,
        None, None,
    )  # fmt: skip
    np.testing.assert_allclose(w_random, w_cyclic, rtol=1e-8, atol=1e-10)
