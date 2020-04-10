"""
This implements the algorithm given in the glmnet paper.
TODO:
- stop normalizing and un-normalizing (make scale/shift methods of model
and store the scaling and shifting params, and give it an 'unnormalize' method)
- profile again
- n_alphas option
- active set convergence
- stopping criteria
"""
from typing import Tuple, Union

import numpy as np
from scipy import sparse as sps

from glm_benchmarks.glmnet_qc.irls import get_irls_z_and_weights_unregularized, get_obj
from glm_benchmarks.glmnet_qc.model import (
    GaussianCanonicalModel,
    GlmnetModel,
    update_params,
)


def soft_threshold(z: float, threshold: float) -> Union[float, int]:
    if np.abs(z) <= threshold:
        return 0
    if z > 0:
        return z - threshold
    return z + threshold


def _get_new_param(
    mean_resid_times_x: float,
    model: GlmnetModel,
    j: int,
    x_j_squared_times_weights: float,
) -> float:
    numerator = soft_threshold(
        mean_resid_times_x, model.l1_ratio * model.alpha * model.penalty_scaling[j]
    )
    if numerator == 0:
        new_param_j = 0.0
    else:
        denominator = x_j_squared_times_weights + model.alpha * model.penalty_scaling[
            j
        ] * (1 - model.l1_ratio)
        new_param_j = numerator / denominator
    return new_param_j


def _get_coordinate_wise_update_naive(
    model: GaussianCanonicalModel, j: int, resid: np.ndarray,
) -> Tuple[float, np.ndarray]:

    x_j = model.x[:, j]
    w_times_xj_squared = model.weights.dot(x_j ** 2)

    mean_resid_times_x = (
        x_j.dot(model.weights * resid) + model.params[j] * w_times_xj_squared
    )

    new_param_j = _get_new_param(mean_resid_times_x, model, j, w_times_xj_squared)

    if new_param_j != 0 or model.params[j] != 0:
        resid += x_j * (model.params[j] - new_param_j)
    return new_param_j, resid


def _get_coordinate_wise_update_sparse(
    model: GaussianCanonicalModel, j: int, resid: np.ndarray
) -> Tuple[float, np.ndarray]:
    x_j = model.x.getcol(j)
    assert isinstance(model.x, sps.csc_matrix)
    nonzero_rows = x_j.indices

    w_times_xj_squared = (x_j.data ** 2).dot(model.weights[nonzero_rows])

    mean_resid_times_x = (
        x_j.data.dot(resid[nonzero_rows] * model.weights[nonzero_rows])
        + model.params[j] * w_times_xj_squared
    )

    new_param_j = _get_new_param(mean_resid_times_x, model, j, w_times_xj_squared)
    if new_param_j != 0 or model.params[j] != 0:
        resid[nonzero_rows] += x_j.data * (model.params[j] - new_param_j)

    return new_param_j, resid


def _do_cd(model: GaussianCanonicalModel, n_iters: int = 1) -> GaussianCanonicalModel:
    """
    The paper is not terribly clear on what to do in the sparse case. It sounds like the
    variables are scaled but not centered to as not to alter the sparsity.

    In a medium-sized and very sparse example (10k rows, 10k cols, 0.1% nonzero),
    this takes about 44% of the time of the dense example, and does not reach quite
    as high an R-squared.
    """
    model.set_optimal_intercept()
    resid = model.y - model.predict()
    initial_obj = get_obj(model)

    for i in range(n_iters):
        for j in range(len(model.params)):
            if sps.isspmatrix(model.x):
                model.params[j], resid = _get_coordinate_wise_update_sparse(
                    model, j, resid
                )
            else:
                model.params[j], resid = _get_coordinate_wise_update_naive(
                    model, j, resid
                )

            if get_obj(model) > initial_obj + 1e-7:
                raise RuntimeError(
                    f"Initial obj was {initial_obj}, and current is {get_obj(model)}"
                )

            model.set_optimal_intercept()
            if get_obj(model) > initial_obj + 1e-7:
                raise RuntimeError
            resid -= resid.dot(model.weights) / model.weight_sum

    return model


def get_alpha_path(
    x: Union[np.ndarray, sps.spmatrix],
    y: np.ndarray,
    l1_ratio: float,
    min_max_ratio: float = 0.001,
    n_alphas: int = 100,
) -> np.ndarray:

    # with ridge, there is no alpha such that all coefficients are zero
    l1_ratio = np.maximum(l1_ratio, 1e-5)

    # Find minimum alpha such that all coefficients are zero
    max_grad = np.max(np.abs(x.T.dot(y)))
    max_alpha = max_grad / (len(y) * l1_ratio)
    # paper suggests minimum alpha = .001 * max alpha
    min_alpha = min_max_ratio * max_alpha
    # suggested by paper
    n_values = n_alphas
    alpha_path = np.logspace(np.log(max_alpha), np.log(min_alpha), n_values)
    return alpha_path


# TODO: refactor to avoid so many duplicate parameters between fit_pathwise and


def fit_pathwise(
    y: np.ndarray,
    x: Union[np.ndarray, sps.spmatrix],
    l1_ratio: float,
    weights: np.ndarray = None,
    n_iters: int = 10,
    alpha_path: np.ndarray = None,
    penalty_scaling: np.ndarray = None,
    distribution="gaussian",
    link_name: str = None,
) -> GlmnetModel:
    """
    See documentation for fit_glmnet.
    """
    # TODO: stop normalizing and un-normalizing so much.
    if alpha_path is None:
        alpha_path = get_alpha_path(x, y, l1_ratio)
    assert len(alpha_path) > 0

    model = fit_glmnet(
        y,
        x,
        alpha_path[0],
        l1_ratio,
        weights,
        n_iters=n_iters,
        penalty_scaling=penalty_scaling,
        distribution=distribution,
        link_name=link_name,
    )

    for alpha in alpha_path[1:]:
        model = fit_glmnet(
            y,
            x,
            alpha,
            l1_ratio,
            weights,
            n_iters=n_iters,
            start_params=model.params,
            penalty_scaling=penalty_scaling,
            distribution=distribution,
            link_name=link_name,
        )

    return model


def fit_glmnet_gaussian_canonical(
    y: np.ndarray,
    x: Union[np.ndarray, sps.spmatrix],
    alpha: float,
    l1_ratio: float,
    weights: np.ndarray = None,
    n_iters: int = 10,
    start_params: np.ndarray = None,
    penalty_scaling: np.ndarray = None,
) -> GaussianCanonicalModel:
    """
    Parameters
    ----------
    link_name
    weights
    y
    x
    alpha: Overall penalty multiplier. The glmnet paper calls this 'lambda'.
    l1_ratio: penalty on l1 part is multiplied by l1_ratio; penalty on l2 part is
        multiplied by 1 - l1_ratio.
    weights
    n_iters
    start_params: starting parameters
    penalty_scaling: Optional. Relative penalty. The penalty on parameter [i] is
        alpha * penalty_scaling[i] (times either l1_ratio or (1 - l1_ratio))
    """

    if start_params is None:
        start_params = np.zeros(x.shape[1])
    if penalty_scaling is None:
        penalty_scaling = np.ones(x.shape[1])
        penalty_scaling[0] = 0

    model = GaussianCanonicalModel(
        y, x, alpha, l1_ratio, weights, start_params, penalty_scaling=penalty_scaling
    )
    model = _do_cd(model, n_iters)
    return model


def fit_glmnet(
    y: np.ndarray,
    x: Union[np.ndarray, sps.spmatrix],
    alpha: float,
    l1_ratio: float,
    weights: np.ndarray = None,
    n_iters: int = 10,
    start_params: np.ndarray = None,
    penalty_scaling: np.ndarray = None,
    distribution="gaussian",
    link_name: str = None,
) -> GlmnetModel:
    # TODO: allow for normalization
    model = GlmnetModel(
        y,
        x.tocsc() if sps.issparse(x) else x,
        distribution,
        alpha,
        l1_ratio,
        weights,
        params=start_params,
        penalty_scaling=penalty_scaling,
        link_name=link_name,
    )

    for i in range(n_iters):
        # Just for the likelihood part, not the penalty
        irls_weights, z = get_irls_z_and_weights_unregularized(model)

        irls_model = fit_glmnet_gaussian_canonical(
            z,
            x,
            alpha,
            l1_ratio,
            weights=irls_weights,
            n_iters=10,
            start_params=model.params,
            penalty_scaling=penalty_scaling,
        )

        new_params = irls_model.params

        old_obj = get_obj(model)
        j = 0
        while (
            get_obj(update_params(model, params=new_params)) > old_obj + 1e-7 and j < 10
        ):
            new_params = (new_params + model.params) / 2
            j += 1

        model.params = new_params
        if get_obj(model) > old_obj + 1e-7:
            raise RuntimeError("did not converge")

    return model
