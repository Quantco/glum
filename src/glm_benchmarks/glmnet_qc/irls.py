import copy
from typing import Tuple, Union

import numpy as np
import statsmodels.api as sm
from glmnet_python import glmnet

from glm_benchmarks.glmnet_qc.glmnet_qc import GlmnetModel, update_params
from glm_benchmarks.glmnet_qc.run_simulated_example import sim_data


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
    if model.distribution == "gaussian":
        resids = model.y - model.predict()
        minus_ll = (resids ** 2).mean() / 2
    elif model.distribution == "poisson":
        prediction = model.predict()
        minus_ll = (prediction.sum() - model.y.dot(np.log(prediction))) / len(model.y)
    else:
        raise NotImplementedError
    return minus_ll + get_penalty(model)


def get_grad_wrt_mean(model: GlmnetModel) -> np.ndarray:
    """ returns length-N vector: - dLLi / d theta_i """
    pred_mean = model.predict()
    if model.distribution == "gaussian":
        return pred_mean - model.y
    if model.distribution == "poisson":
        return 1 - model.y / pred_mean
    else:
        raise NotImplementedError


def get_d_inv_link_d_eta(model: GlmnetModel) -> Union[np.ndarray, float]:
    if model.link_name == "identity":
        return 1
    if model.link_name == "log":
        return model.predict()
    raise NotImplementedError


def get_grad(model: GlmnetModel) -> np.ndarray:
    """
    returns k-length vector:
    sum_i grad_wrt_theta_i * grad_theta_wrt_eta_i * x_i
    """
    d_pred_mean_d_eta = get_d_inv_link_d_eta(model)
    grad = model.x.T.dot(get_grad_wrt_mean(model) * d_pred_mean_d_eta) / len(model.y)
    return grad + get_penalty_grad(model)


def get_hess_wrt_mean(model: GlmnetModel) -> Union[float, np.ndarray]:
    if model.distribution == "gaussian":
        return 1
    if model.distribution == "poisson":
        return model.y / model.predict() ** 2
    raise NotImplementedError


def get_d2_inv_link_d_eta(model: GlmnetModel) -> Union[np.ndarray, float]:

    if model.link_name == "identity":
        return 0
    if model.link_name == "log":
        return model.predict()
    raise NotImplementedError


def get_hess(model: GlmnetModel) -> np.ndarray:

    n_length_parts = get_hess_wrt_mean(model) * get_d_inv_link_d_eta(
        model
    ) ** 2 + get_grad_wrt_mean(model) * get_d2_inv_link_d_eta(model)

    ll_hess = (model.x * n_length_parts[:, None]).T.dot(model.x)
    print(np.diag(ll_hess)[:5])
    assert (n_length_parts > 0).all()
    assert (np.diag(ll_hess) > 0).all()
    return ll_hess / len(model.y) + get_penalty_hess(model)


def get_irls_z_and_weights_unregularized(
    model: GlmnetModel,
) -> Tuple[np.ndarray, np.ndarray]:
    w = (
        get_hess_wrt_mean(model) * (get_d_inv_link_d_eta(model) ** 2)
        + get_grad_wrt_mean(model) * get_d2_inv_link_d_eta(model)
    ) / 2
    assert (w > 0).all()
    assert np.isfinite(w).all()

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
    nonzero = np.where(w != 0)
    print(w.shape)
    print(w[nonzero].shape)
    assert np.isfinite(z[nonzero]).all()
    new_param = sm.WLS(z[nonzero], model.x[nonzero], w[nonzero]).fit().params
    return do_backtracking_line_search(model, new_param)


def get_one_newton_update(model: GlmnetModel) -> np.ndarray:
    step = -np.linalg.lstsq(get_hess(model), get_grad(model), rcond=None)[0]
    new_param = model.params + step
    return do_backtracking_line_search(model, new_param)


def main():
    sim = True
    distribution = "gaussian"
    link_name = "log"

    if sim:
        n_cols = 100
        data = sim_data(1000, n_cols, False)
        np.random.seed(0)
        if distribution == "poisson" or link_name == "log":
            true_params = np.random.normal(0, 1, n_cols)

            data["y"] = np.random.poisson(np.exp(data["x"].dot(true_params))).astype(
                "float"
            )
    else:
        data = {
            "x": np.array([[-2, -1, 1, 2], [0, 0, 1, 1.0]]).T,
            "y": np.array([0, 1, 1, 2.0]),
        }

    x = np.hstack([np.ones((len(data["y"]), 1)), data["x"]])

    alpha = 0
    l1_ratio = 0.5

    model_one = GlmnetModel(
        data["y"], x, distribution, alpha, l1_ratio, link_name=link_name
    )

    print("initial")
    print(get_obj(model_one))
    model_one.params = get_one_newton_update(model_one)

    print("newton")
    print(get_obj(model_one))

    model = GlmnetModel(
        data["y"], x, distribution, alpha, l1_ratio, link_name=link_name
    )
    model.params = get_one_irls_step(model)

    print("irls")
    print(get_obj(model))

    if False:
        glmnet_m = glmnet(
            x=data["x"],
            y=data["y"],
            family=distribution,
            alpha=l1_ratio,
            lambdau=np.array([alpha]),
            standardize=False,
            thresh=1e-7,
        )

        print("obj with our params according to us")
        print(get_obj(model))

        theirs = np.squeeze(np.concatenate(([glmnet_m["a0"]], glmnet_m["beta"])))
        print("obj with their params according to us")
        new_model = copy.copy(model)
        new_model.params = theirs
        print(get_obj(new_model))


if __name__ == "__main__":
    main()
