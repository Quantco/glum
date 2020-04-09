import numpy as np

from glm_benchmarks.glmnet_qc.glmnet_qc import GlmnetModel, update_params
from glm_benchmarks.glmnet_qc.irls import get_grad, get_grad_wrt_mean, get_hess, get_obj

n_rows = 100
n_cols = 5
eps = 1e-4


def make_model(distribution: str, link_name: str = None) -> GlmnetModel:
    np.random.seed(0)
    y = np.random.choice(np.arange(2), n_rows)
    x = np.random.normal(0, 1, (n_rows, n_cols))
    model = GlmnetModel(
        y,
        x,
        distribution,
        0,
        0,
        params=np.random.normal(0, 1, n_cols),
        link_name=link_name,
    )
    return model


def grad_test(distribution: str, link_name: str = None) -> None:
    model = make_model(distribution, link_name)

    i = 0
    grad = get_grad(model)[i]

    new_params = model.params.copy()
    new_params[i] -= eps
    obj_low = get_obj(update_params(model, new_params))

    new_params = model.params.copy()
    new_params[i] += eps
    obj_high = get_obj(update_params(model, new_params))
    np.testing.assert_allclose(grad, (obj_high - obj_low) / (2 * eps), rtol=1e-4)


def hessian_test(distribution: str, link_name: str = None) -> np.ndarray:
    model = make_model(distribution, link_name)

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


def test_gaussian_hess():
    hess = hessian_test("gaussian")
    assert (np.diag(hess) > 0).all()


def test_poisson_hess():
    hess = hessian_test("poisson")
    assert (np.diag(hess) > 0).all()


def test_gaussian_custom_link_hess():
    hessian_test("gaussian", "log")


def test_bernoulli_grad():
    grad_test("bernoulli")


def test_bernoulli_hess():
    hess = hessian_test("bernoulli")
    assert (np.diag(hess) > 0).all()


def grad_wrt_mean_test(distribution: str):
    model = make_model(distribution)
    grad = get_grad_wrt_mean(model)

    high_model = update_params(model, intercept=model.intercept + eps)
    high_mean = high_model.predict()
    obj_high = get_obj(high_model)

    low_model = update_params(model, intercept=model.intercept - eps)
    low_mean = low_model.predict()
    obj_low = get_obj(low_model)

    mean_change = high_mean - low_mean

    np.testing.assert_allclose(grad.dot(mean_change), obj_high - obj_low)


def test_grad_wrt_mean_gaussian():
    grad_wrt_mean_test("gaussian")


def test_grad_wrt_mean_poission():
    grad_wrt_mean_test("poisson")


def test_grad_wrt_mean_bernoulli():
    grad_wrt_mean_test("bernoulli")
