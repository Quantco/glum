import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import ConvergenceWarning

import glum._solvers as solvers
from glum import GeneralizedLinearRegressor
from glum._solvers import _l1_line_minimum, _null_space_basis
from glum_benchmarks.data.simulated_glm import simulate_mixed_data


def _phi(t, w, v, P1, gv, vQv):
    t = np.atleast_1d(t)[:, None]
    return (t * gv + 0.5 * vQv * t**2 + (P1 * np.abs(w + t * v)).sum(1, keepdims=True))[
        :, 0
    ]


@pytest.mark.parametrize("seed", range(200))
def test_l1_line_minimum_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    n = rng.integers(1, 7)
    w = rng.normal(size=n) * (rng.random(n) < 0.6)
    v = rng.normal(size=n)
    P1 = np.abs(rng.normal(size=n)) * (rng.random(n) < 0.8)
    gv = 2 * rng.normal()
    vQv = 0.0 if seed % 2 else abs(rng.normal())
    lb = np.full(n, -np.inf)
    ub = np.full(n, np.inf)
    if seed % 3 == 0:
        lb = np.where(rng.random(n) < 0.5, w - np.abs(rng.normal(size=n)), -np.inf)
        ub = np.where(rng.random(n) < 0.5, w + np.abs(rng.normal(size=n)), np.inf)

    t = _l1_line_minimum(w, v, P1, gv, vQv, lb, ub)

    lo = np.max(np.where(v > 0, (lb - w) / v, (ub - w) / v))
    hi = np.min(np.where(v > 0, (ub - w) / v, (lb - w) / v))
    assert lo - 1e-12 <= t <= hi + 1e-12
    assert np.all(w + t * v >= lb - 1e-12) and np.all(w + t * v <= ub + 1e-12)
    assert _phi(t, w, v, P1, gv, vQv)[0] <= _phi(0.0, w, v, P1, gv, vQv)[0] + 1e-12

    total = np.sum(P1 * np.abs(v))
    unbounded = vQv == 0 and (
        (gv + total < 0 and hi == np.inf) or (gv - total > 0 and lo == -np.inf)
    )
    if not unbounded:
        grid = np.linspace(max(lo, -30), min(hi, 30), 20001)
        bp = -w / v
        cand = np.concatenate([grid, bp[(bp >= lo) & (bp <= hi)], [0.0]])
        best = _phi(cand, w, v, P1, gv, vQv).min()
        assert _phi(t, w, v, P1, gv, vQv)[0] <= best + 1e-9 * (1 + abs(best))


def test_l1_line_minimum_stays_put_on_flat_interval():
    # phi(t) = |1 + t| + |1 - t| is flat on [-1, 1]: do not move
    w, v, P1 = np.array([1.0, 1.0]), np.array([1.0, -1.0]), np.ones(2)
    assert _l1_line_minimum(w, v, P1, 0.0, 0.0, -np.inf * P1, np.inf * P1) == 0.0
    # with a slope it moves to the nearest edge of the flat region of the
    # minimum, not across it
    assert _l1_line_minimum(w, v, P1, -1.0, 0.0, -np.inf * P1, np.inf * P1) == 1.0


def _one_hot_design(rng, n=400):
    X = np.zeros((n, 1 + 4 + 3 + 2))
    X[:, 0] = 1  # intercept
    X[np.arange(n), 1 + rng.integers(0, 4, n)] = 1
    X[np.arange(n), 5 + rng.integers(0, 3, n)] = 1
    X[:, 8:] = rng.normal(size=(n, 2))
    return X


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_null_space_basis_one_hot(dtype):
    rng = np.random.default_rng(0)
    X = _one_hot_design(rng)
    Q = ((X.T * rng.uniform(0.5, 1.5, len(X))) @ X).astype(dtype)

    basis = _null_space_basis(Q)

    assert basis is not None and basis.shape == (10, 2)
    rtol = 1e-12 if dtype == np.float64 else 1e-5
    np.testing.assert_allclose(Q @ basis, 0, atol=rtol * np.abs(Q).max())
    # one direction per categorical (plus the intercept), nothing else
    supports = sorted(tuple(np.flatnonzero(b)) for b in basis.T)
    assert supports == [(0, 1, 2, 3, 4), (0, 5, 6, 7)]


def test_null_space_basis_full_rank():
    rng = np.random.default_rng(0)
    X = _one_hot_design(rng)[:, [0, 1, 2, 3, 5, 6, 8, 9]]  # drop one level each
    assert _null_space_basis(X.T @ X) is None
    assert _null_space_basis(np.eye(1)) is None


def _categorical_data(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "a": pd.Categorical(rng.integers(0, 12, n)),
            "b": pd.Categorical(rng.integers(0, 8, n)),
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
        }
    )
    eta = 0.3 * df["x1"] - 0.2 * df["x2"] + 0.1 * df["a"].cat.codes % 3
    y = rng.poisson(np.exp(eta))
    return df, y


def _fit(X, y, monkeypatch, null_moves, **kwargs):
    with monkeypatch.context() as m:
        if not null_moves:
            m.setattr(solvers, "_NULL_MOVE_CYCLES", 10**9)
        model = GeneralizedLinearRegressor(
            family="poisson", alpha=1e-3, l1_ratio=1.0, solver="irls-cd", **kwargs
        )
        return model.fit(X, y)


@pytest.mark.parametrize("fit_intercept", [True, False])
def test_null_moves_singular_hessian(monkeypatch, fit_intercept):
    """Full one-hot categoricals make the Hessian singular (with an intercept,
    or with more than one categorical). The null moves reach the same optimum in
    far fewer coordinate descent cycles."""
    np.random.seed(0)
    data = simulate_mixed_data(
        ohe_categorical=False, random_seed=0, family="poisson", n_rows=2000,
        dense_features=5, sparse_features=20, sparse_density=0.05,
        categorical_features=2, categorical_levels=10,
    )  # fmt: skip
    X, y = data["X"], data["y"]
    kwargs = dict(drop_first=False, fit_intercept=fit_intercept)
    with_moves = _fit(X, y, monkeypatch, True, **kwargs)
    without = _fit(X, y, monkeypatch, False, **kwargs)

    obj_with = with_moves.diagnostics_[-1]["objective_fct"]
    obj_without = without.diagnostics_[-1]["objective_fct"]
    assert obj_with <= obj_without + 1e-8 * abs(obj_without)
    np.testing.assert_allclose(with_moves.predict(X), without.predict(X), rtol=1e-4)
    assert with_moves.diagnostics_[-1]["n_cycles"] < (
        without.diagnostics_[-1]["n_cycles"] / 5
    )


@pytest.mark.parametrize("drop_first", [True, False])
def test_null_moves_respect_bounds(monkeypatch, drop_first):
    df, y = _categorical_data()
    model = _fit(
        df,
        y,
        monkeypatch,
        True,
        drop_first=drop_first,
        lower_bounds=np.full(22 - drop_first * 2, -0.05),
        upper_bounds=np.full(22 - drop_first * 2, 0.1),
    )
    assert np.all(model.coef_ >= -0.05) and np.all(model.coef_ <= 0.1)


def test_full_rank_unaffected(monkeypatch):
    """Without collinearity, there are no moves and the result is the same.

    It is not bitwise the same: an inner solve that is already converged after
    ``_NULL_MOVE_CYCLES`` cycles now stops there.
    """
    df, y = _categorical_data()
    # small gradient_tol so that some inner solves need more than
    # _NULL_MOVE_CYCLES cycles
    kwargs = dict(drop_first=True, gradient_tol=1e-10, max_iter=200)
    without = _fit(df, y, monkeypatch, False, **kwargs)
    with monkeypatch.context() as m:
        m.setattr(solvers, "_NullSpaceMoves", None)  # fails if it is used
        with_moves = _fit(df, y, monkeypatch, True, **kwargs)
    cycles_per_solve = np.diff([d["n_cycles"] for d in with_moves.diagnostics_])
    assert cycles_per_solve.max() > solvers._NULL_MOVE_CYCLES
    np.testing.assert_allclose(with_moves.coef_, without.coef_, rtol=1e-8)
    np.testing.assert_allclose(with_moves.intercept_, without.intercept_, rtol=1e-8)
    assert with_moves.n_iter_ == without.n_iter_
    assert (
        with_moves.diagnostics_[-1]["n_cycles"] <= without.diagnostics_[-1]["n_cycles"]
    )


@pytest.mark.filterwarnings("ignore:IRLS failed to converge")
def test_inner_non_convergence_warns():
    df, y = _categorical_data()
    model = GeneralizedLinearRegressor(
        family="poisson", alpha=1e-3, l1_ratio=1.0, max_inner_iter=1, max_iter=3
    )
    with pytest.warns(ConvergenceWarning, match="Coordinate descent did not converge"):
        model.fit(df, y)


def test_model_value_matches_quadratic_model():
    """``_NullSpaceMoves.model`` is the coordinate descent model up to a constant."""
    rng = np.random.default_rng(0)
    A = rng.normal(size=(30, 6))
    Q = A.T @ A
    a = np.array([0, 2, 3, 5], dtype=np.int32)
    Qa = Q[np.ix_(a, a)]
    coef0 = rng.normal(size=6)
    score = rng.normal(size=6)
    P1 = np.abs(rng.normal(size=5))  # with an intercept at index 0

    class _State:
        active_set = a
        coef = coef0

    _State.score = score

    class _Data:
        intercept_offset = 1
        has_lower_bounds = has_upper_bounds = False

    _Data.P1 = P1
    moves = solvers._NullSpaceMoves(_State, _Data, Qa, np.eye(4)[:, :1])

    def exact(coef):
        delta = (coef - coef0)[a]
        return (
            -score[a] @ delta
            + 0.5 * delta @ Qa @ delta
            + P1[a[1:] - 1] @ np.abs(coef[a[1:]])
        )

    values = []
    for _ in range(3):
        coef = coef0.copy()
        coef[a] += rng.normal(size=4)
        q = -score.copy()
        q[a] += Qa @ (coef - coef0)[a]
        values.append(moves.model(coef, q) - exact(coef))
    np.testing.assert_allclose(values, values[0], atol=1e-10)
