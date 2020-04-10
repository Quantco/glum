from typing import Tuple, Union

import numpy as np
import statsmodels.api as sm
from scipy import sparse as sps

from glm_benchmarks.glmnet_qc.model import GlmnetModel, update_params


def get_obj(model: GlmnetModel) -> float:
    minus_ll = get_minus_ll(model)
    penalty = model.alpha * (
        model.l1_ratio * model.penalty_scaling.dot(np.abs(model.params))
        + (1 - model.l1_ratio) * model.penalty_scaling.dot(model.params ** 2) / 2
    )
    return minus_ll + penalty


def get_minus_ll(model: GlmnetModel) -> float:
    expected_y = model.predict()
    if model.distribution == "gaussian":
        resids = model.y - expected_y
        ll_i = (resids ** 2) / 2
    elif model.distribution == "poisson":
        ll_i = expected_y - model.y * np.log(expected_y)
    elif model.distribution == "bernoulli":
        ll_i = -model.y * np.log(expected_y) - (1 - model.y) * np.log(1 - expected_y)
    else:
        raise NotImplementedError
    return model.weights.dot(ll_i)


def get_grad_wrt_mean(model: GlmnetModel) -> np.ndarray:
    """ returns length-N vector: - dLLi / d mu_i """
    expected_y = model.predict()
    if model.distribution == "gaussian":
        return expected_y - model.y
    if model.distribution == "poisson":
        return 1 - model.y / expected_y
    if model.distribution == "bernoulli":
        return (1 - model.y) / (1 - expected_y) - model.y / expected_y
    else:
        raise NotImplementedError


def get_d_inv_link_d_eta(model: GlmnetModel) -> Union[np.ndarray, float]:
    """ Returns length N vector: d mu / d eta """
    if model.link_name == "identity":
        return 1
    if model.link_name == "log":
        return model.predict()
    if model.link_name == "logit":
        expected_y = model.predict()
        return -expected_y * (1 - expected_y)
    raise NotImplementedError


def get_grad(model: GlmnetModel) -> np.ndarray:
    """
    returns k-length vector:
    sum_i grad_wrt_theta_i * grad_theta_wrt_eta_i * x_i
    """
    d_pred_mean_d_eta = get_d_inv_link_d_eta(model)
    grad = model.x.T.dot(get_grad_wrt_mean(model) * d_pred_mean_d_eta * model.weights)
    return grad


def get_hess_wrt_mean(model: GlmnetModel) -> Union[float, np.ndarray]:
    if model.distribution == "gaussian":
        return 1
    if model.distribution == "poisson":
        return model.y / (model.predict() ** 2)
    if model.distribution == "bernoulli":
        expected_y = model.predict()
        return model.y / expected_y ** 2 + (1 - model.y) / (1 - expected_y) ** 2
    raise NotImplementedError


def get_d2_inv_link_d_eta(model: GlmnetModel) -> Union[np.ndarray, float]:

    if model.link_name == "identity":
        return 0
    if model.link_name == "log":
        return model.predict()
    if model.link_name == "logit":
        expected_y = model.predict()
        return expected_y * (1 - expected_y) * (1 - 2 * expected_y)
    raise NotImplementedError


def get_hess(model: GlmnetModel) -> np.ndarray:

    n_length_parts = get_hess_wrt_mean(model) * get_d_inv_link_d_eta(
        model
    ) ** 2 + get_grad_wrt_mean(model) * get_d2_inv_link_d_eta(model)
    x = model.x.A if sps.issparse(model.x) else model.x
    ll_hess = (x * (n_length_parts * model.weights)[:, None]).T.dot(x)
    return ll_hess


def get_irls_z_and_weights_unregularized(
    model: GlmnetModel,
) -> Tuple[np.ndarray, np.ndarray]:
    """ TODO: write a test checking that this is a good likelihood approximation. """
    w = get_hess_wrt_mean(model) * (
        get_d_inv_link_d_eta(model) ** 2
    ) + get_grad_wrt_mean(model) * get_d2_inv_link_d_eta(model)

    xb = model.x.dot(model.params)
    z = xb - get_grad_wrt_mean(model) * get_d_inv_link_d_eta(model) / w
    return w * model.weights, z


def do_backtracking_line_search(
    model: GlmnetModel, new_param: np.ndarray, max_n_tries: int = 10
) -> np.ndarray:
    old_obj = get_minus_ll(model)
    i = 0
    while old_obj < get_minus_ll(update_params(model, new_param)) and i < max_n_tries:
        new_param = (model.params + new_param) / 2
    return new_param


def get_one_irls_step(model: GlmnetModel) -> np.ndarray:
    w, z = get_irls_z_and_weights_unregularized(model)
    new_param = sm.WLS(z, model.x, w).fit().params
    return do_backtracking_line_search(model, new_param)


def get_one_newton_update(model: GlmnetModel) -> np.ndarray:
    step = -np.linalg.lstsq(get_hess(model), get_grad(model), rcond=None)[0]
    new_param = model.params + step
    return do_backtracking_line_search(model, new_param)


def main():
    sim = True
    distribution = "gaussian"
    link_name = None

    if sim:
        n_cols = 20
        n_rows = 1000
        np.random.seed(0)
        x = np.random.normal(0, 1, (n_rows, n_cols))
        true_params = np.random.normal(0, 0.1, n_cols)
        xb = x.dot(true_params)

        if distribution == "gaussian":
            y = np.random.normal(xb, 1)
        elif distribution == "poisson":
            y = np.random.poisson(np.exp(xb))
            assert np.all(np.isfinite(y))
        elif distribution == "bernoulli":
            p = 1 / (1 + np.exp(xb))
            y = np.random.uniform(0, 1, n_rows) < p
        else:
            raise NotImplementedError

    else:
        x = np.array([[-2, -1, 1, 2], [0, 0, 1, 1.0]]).T
        y = np.array([0, 1, 1, 2.0])

    alpha = 0
    l1_ratio = 0.5

    model_one = GlmnetModel(y, x, distribution, alpha, l1_ratio, link_name=link_name)
    n_iters = 4

    print("initial")
    print(get_minus_ll(model_one))
    for _ in range(n_iters):
        model_one.params = get_one_newton_update(model_one)

    print("newton")
    print(get_minus_ll(model_one))
    print("dist from true params", np.abs(model_one.params - true_params))

    model = GlmnetModel(y, x, distribution, alpha, l1_ratio, link_name=link_name)
    for _ in range(n_iters):
        model.params = get_one_irls_step(model)

    print("irls")
    print(get_minus_ll(model))
    print("dist from truth", np.abs(model.params - true_params))


if __name__ == "__main__":
    main()
