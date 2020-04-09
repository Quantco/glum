import numpy as np

from glm_benchmarks.glmnet_qc.glmnet_qc import GlmnetModel, update_params
from glm_benchmarks.glmnet_qc.irls import get_grad, get_hess, get_obj

n_rows = 100
n_cols = 5
eps = 1e-4


def make_model(distribution: str, link_name: str = None) -> GlmnetModel:
    np.random.seed(0)
    y = np.random.choice(np.arange(3), n_rows)
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


def hessian_test(distribution: str, link_name: str = None) -> None:
    model = make_model(distribution, link_name)

    i = 0
    j = 1
    hess = get_hess(model)[i, j]

    new_params = model.params.copy()
    new_params[i] -= eps
    grad_low = get_grad(update_params(model, new_params))[j]

    new_params = model.params.copy()
    new_params[i] += eps
    grad_high = get_grad(update_params(model, new_params))[j]
    np.testing.assert_allclose(hess, (grad_high - grad_low) / (2 * eps))


def test_gaussian_grad() -> None:
    grad_test("gaussian")


def test_poisson_grad() -> None:
    grad_test("poisson")


def test_gaussian_custom_link_grad():
    grad_test("gaussian", "log")


def test_gaussian_hess():
    hessian_test("gaussian")


def test_poisson_hess():
    hessian_test("poisson")


def test_gaussian_custom_link_hess():
    hessian_test("gaussian", "log")
