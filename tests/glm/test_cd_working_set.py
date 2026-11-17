import numpy as np
import pytest
import scipy.sparse as sps

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
