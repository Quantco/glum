import numpy as np
from scipy import sparse as sps

from glm_benchmarks.glmnet_qc.irls import (
    get_grad,
    get_grad_wrt_mean,
    get_hess,
    get_minus_ll,
)
from glm_benchmarks.glmnet_qc.model import GlmnetModel, update_params

n_rows = 100
n_cols = 5
eps = 1e-4


def make_model(
    distribution: str, link_name: str = None, sparse: bool = False
) -> GlmnetModel:
    np.random.seed(0)
    y = np.random.choice(np.arange(2), n_rows)
    if sparse:
        x = sps.hstack((np.ones((n_rows, 1)), sps.random(n_rows, n_cols, density=0.2)))
    else:
        x = np.hstack((np.ones((len(y), 1)), np.random.normal(0, 1, (n_rows, n_cols))))

    model = GlmnetModel(
        y,
        x,
        distribution,
        0,
        0,
        params=np.random.normal(0, 1, n_cols + 1),
        link_name=link_name,
    )
    return model


def grad_test(distribution: str, link_name: str = None, sparse: bool = False) -> None:
    model = make_model(distribution, link_name, sparse)

    i = 0
    grad = get_grad(model)[i]

    new_params = model.params.copy()
    new_params[i] -= eps
    obj_low = get_minus_ll(update_params(model, new_params))

    new_params = model.params.copy()
    new_params[i] += eps
    obj_high = get_minus_ll(update_params(model, new_params))
    np.testing.assert_allclose(grad, (obj_high - obj_low) / (2 * eps), rtol=1e-4)


def hessian_test(
    distribution: str, link_name: str = None, sparse: bool = False
) -> np.ndarray:
    model = make_model(distribution, link_name, sparse)

    i = 0
    j = 1
    whole_hess = get_hess(model)
    hess = whole_hess[i, j]

    new_params = model.params.copy()
    new_params[i] -= eps
    grad_low = get_grad(update_params(model, new_params))[j]

    new_params = model.params.copy()
    new_params[i] += eps
    grad_high = get_grad(update_params(model, new_params))[j]
    np.testing.assert_allclose(hess, (grad_high - grad_low) / (2 * eps))
    return whole_hess


def test_gaussian_grad() -> None:
    grad_test("gaussian")


def test_poisson_grad() -> None:
    grad_test("poisson")


def test_gaussian_custom_link_grad():
    grad_test("gaussian", "log")


def test_bernoulli_grad():
    grad_test("bernoulli")


def test_gaussian_grad_sparse() -> None:
    grad_test("gaussian", sparse=True)


def test_poisson_grad_sparse() -> None:
    grad_test("poisson", sparse=True)


def test_gaussian_custom_link_grad_sparse():
    grad_test("gaussian", "log", True)


def test_bernoulli_grad_sparse():
    grad_test("bernoulli", sparse=True)


def test_gaussian_hess():
    hess = hessian_test("gaussian")
    assert (np.diag(hess) > 0).all()


def test_poisson_hess():
    hess = hessian_test("poisson")
    assert (np.diag(hess) > 0).all()


def test_gaussian_custom_link_hess():
    hessian_test("gaussian", "log")


def test_gaussian_hess_sparse():
    hess = hessian_test("gaussian", sparse=True)
    assert (np.diag(hess) > 0).all()


def test_poisson_hess_sparse():
    hess = hessian_test("poisson", sparse=True)
    assert (np.diag(hess) > 0).all()


def test_gaussian_custom_link_hess_sparse():
    hessian_test("gaussian", "log", sparse=True)


def test_bernoulli_hess():
    hess = hessian_test("bernoulli")
    assert (np.diag(hess) > 0).all()


def test_bernoulli_hess_sparse():
    hess = hessian_test("bernoulli", sparse=True)
    assert (np.diag(hess) > 0).all()


def grad_wrt_mean_test(distribution: str, sparse: bool = False):
    model = make_model(distribution, sparse=sparse)
    grad = get_grad_wrt_mean(model)

    high_params = model.params.copy()
    high_params[0] += eps
    high_model = update_params(model, params=high_params)
    high_mean = high_model.predict()
    obj_high = get_minus_ll(high_model)

    low_params = model.params.copy()
    low_params[0] -= eps
    low_model = update_params(model, params=low_params)
    low_mean = low_model.predict()
    obj_low = get_minus_ll(low_model)

    mean_change = high_mean - low_mean

    np.testing.assert_allclose(
        model.weights.dot(grad * mean_change), obj_high - obj_low
    )


def test_grad_wrt_mean_gaussian():
    grad_wrt_mean_test("gaussian")


def test_grad_wrt_mean_poission():
    grad_wrt_mean_test("poisson")


def test_grad_wrt_mean_bernoulli():
    grad_wrt_mean_test("bernoulli")


def test_grad_wrt_mean_gaussian_sparse():
    grad_wrt_mean_test("gaussian", sparse=True)


def test_grad_wrt_mean_poission_sparse():
    grad_wrt_mean_test("poisson", sparse=True)


def test_grad_wrt_mean_bernoulli_sparse():
    grad_wrt_mean_test("bernoulli", sparse=True)
