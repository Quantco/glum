from typing import Tuple, Union

import numpy as np
import statsmodels.api as sm

from glm_benchmarks.glmnet_qc.glmnet_qc import GlmnetModel, update_params


def get_penalty(model: GlmnetModel) -> float:
    l1_part = model.l1_ratio * np.abs(model.params).dot(model.penalty_scaling)
    l2_part = (1 - model.l1_ratio) * (model.params ** 2).dot(model.penalty_scaling) / 2
    penalty = model.alpha * (l1_part + l2_part)
    return penalty


def get_penalty_grad(model: GlmnetModel) -> np.ndarray:
    l1_part = model.l1_ratio * np.sign(model.params) * model.penalty_scaling
    l2_part = (1 - model.l1_ratio) * model.params * model.penalty_scaling
    penalty_grad = model.alpha * (l1_part + l2_part)
    return penalty_grad


def get_penalty_hess(model: GlmnetModel) -> np.ndarray:
    hess = np.diag(model.alpha * (1 - model.l1_ratio) * model.penalty_scaling)
    return hess


def get_obj(model: GlmnetModel) -> float:
    expected_y = model.predict()
    if model.distribution == "gaussian":
        resids = model.y - expected_y
        minus_ll = (resids ** 2).mean() / 2
    elif model.distribution == "poisson":
        minus_ll = (expected_y.sum() - model.y.dot(np.log(expected_y))) / len(model.y)
    elif model.distribution == "bernoulli":
        minus_ll = -(
            model.y.dot(np.log(expected_y)) + (1 - model.y).dot(np.log(1 - expected_y))
        ) / len(model.y)
    else:
        raise NotImplementedError
    return minus_ll + get_penalty(model)


def get_grad_wrt_mean(model: GlmnetModel) -> np.ndarray:
    """ returns length-N vector: - dLLi / d mu_i """
    expected_y = model.predict()
    if model.distribution == "gaussian":
        return (expected_y - model.y) / len(model.y)
    if model.distribution == "poisson":
        return (1 - model.y / expected_y) / len(model.y)
    if model.distribution == "bernoulli":
        return ((1 - model.y) / (1 - expected_y) - model.y / expected_y) / len(model.y)
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
    grad = model.x.T.dot(get_grad_wrt_mean(model) * d_pred_mean_d_eta)
    return grad + get_penalty_grad(model)


def get_hess_wrt_mean(model: GlmnetModel) -> Union[float, np.ndarray]:
    if model.distribution == "gaussian":
        return 1 / len(model.y)
    if model.distribution == "poisson":
        return model.y / (model.predict() ** 2 * len(model.y))
    if model.distribution == "bernoulli":
        expected_y = model.predict()
        return (
            model.y / expected_y ** 2 + (1 - model.y) / (1 - expected_y) ** 2
        ) / len(model.y)
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
    ll_hess = (model.x * n_length_parts[:, None]).T.dot(model.x)
    return ll_hess + get_penalty_hess(model)


def get_irls_z_and_weights_unregularized(
    model: GlmnetModel,
) -> Tuple[np.ndarray, np.ndarray]:
    w = (
        get_hess_wrt_mean(model) * (get_d_inv_link_d_eta(model) ** 2)
        + get_grad_wrt_mean(model) * get_d2_inv_link_d_eta(model)
    ) / 2

    xb = model.x.dot(model.params) + model.intercept
    z = xb - get_grad_wrt_mean(model) * get_d_inv_link_d_eta(model) / (2 * w)
    return w, z


def do_backtracking_line_search(
    model: GlmnetModel, new_param: np.ndarray, max_n_tries: int = 10
) -> np.ndarray:
    old_obj = get_obj(model)
    i = 0
    while old_obj < get_obj(update_params(model, new_param)) and i < max_n_tries:
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
    print(get_obj(model_one))
    for _ in range(n_iters):
        model_one.params = get_one_newton_update(model_one)

    print("newton")
    print(get_obj(model_one))
    print("dist from true params", np.abs(model_one.params - true_params))

    model = GlmnetModel(y, x, distribution, alpha, l1_ratio, link_name=link_name)
    for _ in range(n_iters):
        model.params = get_one_irls_step(model)

    print("irls")
    print(get_obj(model))
    print("dist from truth", np.abs(model.params - true_params))


if __name__ == "__main__":
    main()
