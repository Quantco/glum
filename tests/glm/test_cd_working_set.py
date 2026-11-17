import warnings

import numpy as np
import pytest
import scipy.sparse as sps
from sklearn.exceptions import ConvergenceWarning

import glum._glm
import glum._solvers
from glum import GeneralizedLinearRegressor


def _wide_problem(family, sparse=False, seed=0):
    # More columns than rows, so irls-cd uses the working set solver.
    rng = np.random.default_rng(seed)
    n, p = 60, 250
    X = rng.normal(size=(n, p)) * rng.uniform(0.5, 3, p)
    if sparse:
        X = sps.csc_matrix(X * (rng.uniform(size=(n, p)) < 0.3))
    beta = np.zeros(p)
    beta[:5] = [1.0, -0.8, 0.6, 0.5, -0.4]
    eta = 0.5 * (X @ beta)
    if family == "normal":
        y = eta + rng.normal(size=n)
    elif family == "poisson":
        y = rng.poisson(np.exp(0.3 * eta)).astype(float)
    else:
        y = (rng.uniform(size=n) < 1 / (1 + np.exp(-eta))).astype(float)
    return X, y


def _fit(X, y, **kwargs):
    return GeneralizedLinearRegressor(gradient_tol=1e-10, max_iter=200, **kwargs).fit(
        X, y
    )


def _objective(model, X, y, alpha, l1_ratio):
    dev = model._family_instance.deviance(
        y, model.predict(X), sample_weight=np.full(len(y), 1 / len(y))
    )
    b = model.coef_
    return 0.5 * dev + alpha * (
        l1_ratio * np.abs(b).sum() + 0.5 * (1 - l1_ratio) * (b @ b)
    )


@pytest.mark.parametrize("family", ["normal", "poisson", "binomial"])
@pytest.mark.parametrize("sparse", [False, True], ids=["dense", "sparse"])
@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("l1_ratio", [1.0, 0.5])
def test_working_set_matches_gram_solver(
    monkeypatch, family, sparse, fit_intercept, l1_ratio
):
    # The working set solver reaches the same optimum as coordinate descent on
    # the full active Hessian. poisson and binomial take several IRLS
    # iterations, so the inner solves start away from zero.
    X, y = _wide_problem(family, sparse)
    alpha = {"normal": 0.05, "poisson": 0.02, "binomial": 0.02}[family]
    kwargs = dict(
        family=family, alpha=alpha, l1_ratio=l1_ratio, fit_intercept=fit_intercept
    )
    ws = _fit(X, y, **kwargs)
    with monkeypatch.context() as m:
        m.setattr(glum._glm, "_cd_working_set_solver", glum._solvers._cd_solver)
        gram = _fit(X, y, **kwargs)

    obj_ws = _objective(ws, X, y, alpha, l1_ratio)
    obj_gram = _objective(gram, X, y, alpha, l1_ratio)
    assert obj_ws <= obj_gram * (1 + 1e-10)
    np.testing.assert_allclose(ws.coef_, gram.coef_, atol=1e-6)
    np.testing.assert_allclose(ws.intercept_, gram.intercept_, atol=1e-6)


def test_working_set_respects_bounds(monkeypatch):
    # With bounds, the solution is not polished; it must stay feasible and
    # match the Gram solver.
    X, y = _wide_problem("normal")
    kwargs = dict(family="normal", alpha=0.02, l1_ratio=1.0, lower_bounds=-0.2)
    ws = _fit(X, y, **kwargs)
    with monkeypatch.context() as m:
        m.setattr(glum._glm, "_cd_working_set_solver", glum._solvers._cd_solver)
        gram = _fit(X, y, **kwargs)
    assert ws.coef_.min() >= -0.2 - 1e-12
    assert (ws.coef_ <= -0.2 + 1e-12).any()  # some bound is active
    np.testing.assert_allclose(ws.coef_, gram.coef_, atol=1e-6)


def test_working_set_does_not_build_full_hessian(monkeypatch):
    # The point of the working set solver: no Hessian of the full active set.
    def fail(*args, **kwargs):
        raise AssertionError("update_hessian called")

    monkeypatch.setattr(glum._solvers, "update_hessian", fail)
    X, y = _wide_problem("poisson")
    model = _fit(X, y, family="poisson", alpha=0.02, l1_ratio=1.0)
    assert 0 < (model.coef_ != 0).sum() < X.shape[0]


@pytest.mark.parametrize("sparse", [False, True], ids=["dense", "sparse"])
def test_default_feature_names(sparse):
    # Unnamed numpy and scipy inputs get the default names.
    X, y = _wide_problem("normal", sparse)
    model = GeneralizedLinearRegressor(alpha=0.1).fit(X[:, :4], y)
    expected = [f"_col_{i}" for i in range(4)]
    assert model.feature_names_ == expected
    assert model.term_names_ == expected
    assert model.term_names_ is not model.feature_names_


@pytest.mark.parametrize("intercept", [False, True])
def test_norm_min_subgrad_max_without_intercept(intercept):
    # The maximum of the minimum norm subgradient used to stay at its initial
    # value of +inf without an intercept, which made every column active.
    from glum._cd_fast import _norm_min_subgrad

    grad = np.array([0.1, 1.0, -3.0, 0.5])
    coef = np.zeros(4)
    P1 = np.zeros(4 - intercept)
    norm, max_ = _norm_min_subgrad(
        np.arange(4, dtype=np.int32),
        coef,
        grad,
        P1,
        intercept,
        False,
        np.empty(0),
        False,
        np.empty(0),
    )
    assert norm == pytest.approx(np.abs(grad).sum())
    assert max_ == pytest.approx(3.0)


def _screenable_problem(family, seed=0):
    # Wide, sparse, non-negative, with a few frequent columns and many rare
    # ones of small norm, like tf-idf text features.
    rng = np.random.default_rng(seed)
    n, p = 300, 3000
    density = np.where(np.arange(p) < 30, 0.3, 0.003)
    mask = rng.uniform(size=(n, p)) < density
    scale = np.where(np.arange(p) < 30, 1.0, 0.1)  # rare columns have small values
    X = sps.csc_matrix(mask * rng.uniform(0.1, 1.0, size=(n, p)) * scale)
    beta = np.zeros(p)
    beta[:8] = [1.5, -1.2, 1.0, 0.8, -0.8, 0.6, 0.5, -0.4]
    eta = X @ beta
    if family == "normal":
        y = eta + 0.3 * rng.normal(size=n)
    elif family == "poisson":
        y = rng.poisson(np.exp(0.5 * eta)).astype(float)
    else:
        y = (rng.uniform(size=n) < 1 / (1 + np.exp(-eta))).astype(float)
    return X, y


@pytest.mark.parametrize("family", ["normal", "poisson", "binomial"])
@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("weights", [False, True])
def test_screening_matches_unscreened(monkeypatch, family, fit_intercept, weights):
    # Screening fits only the columns that can be nonzero and certifies the
    # rest, so it must reach the optimum of the full problem.
    X, y = _screenable_problem(family)
    rng = np.random.default_rng(1)
    sw = rng.uniform(0.5, 2.0, size=len(y)) if weights else None
    offset = 0.1 * rng.normal(size=len(y)) if family != "binomial" else None
    alpha = {"normal": 0.01, "poisson": 0.01, "binomial": 0.005}[family]
    kwargs = dict(family=family, alpha=alpha, l1_ratio=1.0, fit_intercept=fit_intercept)

    calls = []
    solve_screened = GeneralizedLinearRegressor._solve_screened

    def spy(self, *args, **kw):
        out = solve_screened(self, *args, **kw)
        calls.append(out is not None)
        return out

    monkeypatch.setattr(GeneralizedLinearRegressor, "_solve_screened", spy)
    screened = GeneralizedLinearRegressor(gradient_tol=1e-10, max_iter=200, **kwargs)
    screened.fit(X, y, sample_weight=sw, offset=offset)
    assert calls == [True]  # screening was used

    monkeypatch.setattr(
        GeneralizedLinearRegressor, "_can_screen", lambda self, *a, **k: False
    )
    full = GeneralizedLinearRegressor(gradient_tol=1e-10, max_iter=200, **kwargs)
    full.fit(X, y, sample_weight=sw, offset=offset)
    np.testing.assert_allclose(screened.coef_, full.coef_, atol=1e-7)
    np.testing.assert_allclose(screened.intercept_, full.intercept_, atol=1e-7)
    assert len(screened.coef_) == X.shape[1]
    np.testing.assert_allclose(screened.col_means_, full.col_means_, rtol=1e-12)
    np.testing.assert_allclose(screened.col_stds_, full.col_stds_, rtol=1e-12)


@pytest.mark.parametrize("dense", [False, True])
@pytest.mark.parametrize("signed", [False, True])
@pytest.mark.parametrize("scale_predictors", [False, True])
def test_screening_input_variants(monkeypatch, dense, signed, scale_predictors):
    # The bound uses the absolute values of the columns and, when the model is
    # estimated as if scaled, their standard deviations. Screen even if it keeps
    # most columns, as it does with scale_predictors.
    monkeypatch.setattr(glum._glm, "_SCREEN_MAX_RATIO", 1)
    X, y = _screenable_problem("normal")
    if signed:
        X = X.copy()
        X.data *= np.where(np.arange(X.nnz) % 3 == 0, -1.0, 1.0)
    if dense:
        X = X.toarray()
    kwargs = dict(
        family="normal",
        alpha=0.01,
        l1_ratio=1.0,
        gradient_tol=1e-10,
        scale_predictors=scale_predictors,
    )
    calls = []
    screen_columns = GeneralizedLinearRegressor._screen_columns

    def spy(self, *args, **kw):
        out = screen_columns(self, *args, **kw)
        calls.append(out is not None)
        return out

    monkeypatch.setattr(GeneralizedLinearRegressor, "_screen_columns", spy)
    screened = GeneralizedLinearRegressor(**kwargs).fit(X, y)
    assert calls == [True]
    monkeypatch.setattr(
        GeneralizedLinearRegressor, "_can_screen", lambda self, *a, **k: False
    )
    full = GeneralizedLinearRegressor(**kwargs).fit(X, y)
    assert (full.coef_ != 0).sum() > 3
    np.testing.assert_allclose(screened.coef_, full.coef_, atol=1e-7)
    np.testing.assert_allclose(screened.intercept_, full.intercept_, atol=1e-7)
    np.testing.assert_allclose(screened.col_stds_, full.col_stds_, rtol=1e-12)


def test_screening_certificate_adds_missed_columns(monkeypatch):
    # With a guess that is far too small, the first subproblem misses columns
    # of the solution (here: all of them); the full-problem check must add them.
    monkeypatch.setattr(glum._glm, "_SCREEN_GUESS", 0.01)
    added = []
    add = glum._glm._Screen.add

    def spy(self, cols):
        added.append(len(cols))
        add(self, cols)

    monkeypatch.setattr(glum._glm._Screen, "add", spy)
    X, y = _screenable_problem("normal")
    kwargs = dict(family="normal", alpha=0.01, l1_ratio=1.0, gradient_tol=1e-10)
    low = GeneralizedLinearRegressor(**kwargs).fit(X, y)
    assert sum(added) > 0
    monkeypatch.setattr(
        GeneralizedLinearRegressor, "_can_screen", lambda self, *a, **k: False
    )
    full = GeneralizedLinearRegressor(**kwargs).fit(X, y)
    assert (full.coef_ != 0).sum() > 3
    np.testing.assert_allclose(low.coef_, full.coef_, atol=1e-7)


@pytest.mark.parametrize("family", ["normal", "binomial"])
@pytest.mark.parametrize("fit_intercept", [True, False])
def test_screening_keeps_no_column(family, fit_intercept):
    # Far above the largest useful alpha, no column passes the bound.
    X, y = _screenable_problem(family)
    model = GeneralizedLinearRegressor(
        family=family, alpha=100.0, l1_ratio=1.0, fit_intercept=fit_intercept
    ).fit(X, y)
    assert not model.coef_.any()
    if fit_intercept:
        mean = y.mean()
        expected = mean if family == "normal" else np.log(mean / (1 - mean))
        np.testing.assert_allclose(model.intercept_, expected, rtol=1e-6)


def test_working_set_stops_at_unreachable_tolerance():
    # In float32, gradient_tol=1e-10 cannot be reached; the solver must give up
    # with a warning instead of running round after round.
    X, y = _screenable_problem("normal")
    model = GeneralizedLinearRegressor(
        alpha=0.01, l1_ratio=1.0, gradient_tol=1e-10, max_inner_iter=1000
    )
    with pytest.warns(ConvergenceWarning):
        model.fit(X.astype(np.float32), y.astype(np.float32))
    assert (model.coef_ != 0).sum() > 3


def test_screening_zero_weights_do_not_warn():
    X, y = _screenable_problem("poisson")
    sw = np.ones(len(y))
    sw[:10] = 0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        GeneralizedLinearRegressor(family="poisson", alpha=0.01, l1_ratio=1.0).fit(
            X, y, sample_weight=sw
        )


def test_screening_counts_iterations_of_all_subproblems(monkeypatch):
    monkeypatch.setattr(glum._glm, "_SCREEN_GUESS", 0.01)
    X, y = _screenable_problem("binomial")
    model = GeneralizedLinearRegressor(
        family="binomial", alpha=0.005, l1_ratio=1.0, gradient_tol=1e-10
    ).fit(X, y)
    # The diagnostics of each solve start with its initial point.
    n_solves = sum(d["n_iter"] == 0 for d in model.diagnostics_)
    assert n_solves >= 2
    assert model.n_iter_ == len(model.diagnostics_) - n_solves
