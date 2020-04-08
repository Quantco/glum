"""
This implements the algorithm given in the glmnet paper.
"""
import copy
import time
from typing import Any, Dict, Tuple, Union

import numpy as np
from scipy import sparse as sps


class GlmnetModel:
    def __init__(
        self,
        y: np.ndarray,
        x: Union[np.ndarray, sps.spmatrix],
        distribution: str,
        alpha: float,
        l1_ratio: float,
        intercept: float = 0.0,
        params: np.ndarray = None,
        penalty_scaling: Union[np.ndarray, None] = None,
    ):
        """
        Assume x does *not* include an intercept.
        """
        if alpha < 0:
            raise ValueError("alpha must be positive.")
        if not 0 <= l1_ratio <= 1:
            raise ValueError("l1_ratio must be between zero and one.")
        self.distribution = distribution
        self.y = y
        self.x = x
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        if params is None:
            params = np.zeros(x.shape[1])
        else:
            if not params.shape == (x.shape[1],):
                raise ValueError(
                    f"""params should have shape {x.shape[1],}. Do not include the
                    intercept."""
                )
        self.params = params
        self.intercept = intercept
        if penalty_scaling is not None and np.any(penalty_scaling < 0):
            raise ValueError("penalty_scaling must be non-negative.")
        self.penalty_scaling = penalty_scaling
        self.link, self.inv_link = get_link_and_inverse(self.distribution)

    def predict(self):
        return self.inv_link(self.intercept + self.x.dot(self.params))

    def get_overall_penalty_coordinate_j(self, j: int) -> float:
        if self.penalty_scaling is None:
            return self.alpha
        return self.alpha * self.penalty_scaling[j]

    def get_r2(self, y: np.ndarray) -> float:
        return 1 - np.var(y - self.predict()) / np.var(y)

    def _unstandardize(self, x, x_col_means, x_col_stds):
        out = copy.deepcopy(self)
        out.x = x
        out.params /= x_col_stds
        out.intercept -= (x_col_means / x_col_stds).dot(self.params)
        return out


def get_link_and_inverse(distribution):
    if distribution == "gaussian":
        return (lambda x: x, lambda x: x)
    elif distribution == "poisson":
        return (lambda x: np.log(x), lambda x: np.exp(x))
    else:
        raise NotImplementedError(f"{distribution} is not a supported distribution")


def soft_threshold(z: float, threshold: float) -> Union[float, int]:
    if np.abs(z) <= threshold:
        return 0
    if z > 0:
        return z - threshold
    return z + threshold


def _get_new_param(mean_resid_times_x: float, model: GlmnetModel, j: int) -> float:
    p = model.get_overall_penalty_coordinate_j(j)
    numerator = soft_threshold(mean_resid_times_x, model.l1_ratio * p)
    if numerator == 0:
        new_param_j = 0.0
    else:
        denominator = 1 + p * (1 - model.l1_ratio)
        new_param_j = numerator / denominator
    return new_param_j


def _get_coordinate_wise_update_naive(
    model: GlmnetModel, j: int, resid: np.ndarray,
) -> Tuple[float, np.ndarray]:

    x_j = model.x[:, j]
    mean_resid_times_x = x_j.dot(resid) / len(model.y) + model.params[j]
    new_param_j = _get_new_param(mean_resid_times_x, model, j)

    if new_param_j != 0 or model.params[j] != 0:
        resid += x_j * (model.params[j] - new_param_j)
    return new_param_j, resid


# TODO: refactor a bit so that there's less duplication between _do_cd_naive
# and _do_cd_sparse
def _do_cd_naive(
    y: np.ndarray,
    x: np.ndarray,
    alpha: float,
    l1_ratio: float,
    start_params: np.ndarray,
    n_iters: int,
    report_normalized=False,
    penalty_scaling: Union[np.ndarray, None] = None,
    distribution="gaussian",
) -> GlmnetModel:
    """
    This is the "naive algorithm" from the paper.
    """
    x_col_means = x.mean(0)
    x_col_stds = x.std(0)
    x_standardized = (x - x_col_means[None, :]) / x_col_stds[None, :]
    model = GlmnetModel(
        y,
        x_standardized,
        distribution,
        alpha,
        l1_ratio,
        params=start_params * x_col_stds,
        intercept=(y - x_standardized.dot(start_params)).mean(),
        penalty_scaling=penalty_scaling,
    )
    resid = model.y - model.predict()

    for i in range(n_iters):
        for j in range(len(model.params)):
            model.params[j], resid = _get_coordinate_wise_update_naive(model, j, resid)

    # TODO: This assertion is true for gaussian. Will it still be true for Poisson?
    np.testing.assert_almost_equal(
        model.intercept, (y - x_standardized.dot(model.params)).mean()
    )

    return (
        model if report_normalized else model._unstandardize(x, x_col_means, x_col_stds)
    )


def _get_coordinate_wise_update_sparse(
    model: GlmnetModel, j: int, resid: np.ndarray
) -> Tuple[float, np.ndarray]:
    x_j = model.x.getcol(j)
    nonzero_rows = x_j.indices

    mean_resid_times_x = (
        x_j.data.dot(resid[nonzero_rows]) / len(model.y) + model.params[j]
    )

    new_param_j = _get_new_param(mean_resid_times_x, model, j)
    if new_param_j != 0 or model.params[j] != 0:
        resid[nonzero_rows] += x_j.data * (model.params[j] - new_param_j)

    return new_param_j, resid


def _do_cd_sparse(
    y: np.ndarray,
    x: sps.spmatrix,
    alpha: float,
    l1_ratio: float,
    start_params: np.ndarray,
    n_iters: int,
    report_normalized=False,
    penalty_scaling: Union[np.ndarray, None] = None,
    distribution="gaussian",
) -> GlmnetModel:
    """
    The paper is not terribly clear on what to do here. It sounds like the
    variables are scaled but not centered to as not to alter the sparsity.

    In a medium-sized and very sparse example (10k rows, 10k cols, 0.1% nonzero),
    this takes about 44% of the time of the dense example, and does not reach quite
    as high an R-squared.
    """
    x_norm = np.sqrt(x.power(2).sum(0) / x.shape[0])
    z = sps.csc_matrix(1 / x_norm)
    x_scaled = x.multiply(z)
    model = GlmnetModel(
        y,
        x_scaled,
        distribution,
        alpha,
        l1_ratio,
        params=start_params * np.array(x_norm)[0, :],
        intercept=(y - x.dot(start_params)).mean(),
        penalty_scaling=penalty_scaling,
    )
    resid = y - model.predict()
    model.intercept = resid.mean()
    resid -= resid.mean()

    for i in range(n_iters):
        for j in range(len(model.params)):
            model.params[j], resid = _get_coordinate_wise_update_sparse(model, j, resid)
        model.intercept += resid.mean()
        resid -= resid.mean()

    if not report_normalized:
        model.x = x
        model.params /= np.array(x_norm)[0, :]
        model.intercept = (model.y - model.x.dot(model.params)).mean()
        model.intercept += (model.y - model.predict()).mean()

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
# fit_glmnet


def fit_pathwise(
    y: np.ndarray,
    x: Union[np.ndarray, sps.spmatrix],
    l1_ratio: float,
    n_iters: int = 10,
    solver: str = "naive",
    alpha_path: np.ndarray = None,
    penalty_scaling: Union[np.ndarray, None] = None,
    distribution="gaussian",
) -> GlmnetModel:
    """
    See documentation for fit_glmnet.
    """
    if alpha_path is None:
        alpha_path = get_alpha_path(x, y, l1_ratio)
    assert len(alpha_path) > 0

    model = fit_glmnet(y, x, alpha_path[0], l1_ratio, n_iters, solver)

    for alpha in alpha_path[1:]:
        model = fit_glmnet(
            y,
            x,
            alpha,
            l1_ratio,
            n_iters,
            solver,
            start_params=model.params,
            penalty_scaling=penalty_scaling,
            distribution=distribution,
        )

    return model


def fit_glmnet(
    y: np.ndarray,
    x: Union[np.ndarray, sps.spmatrix],
    alpha: float,
    l1_ratio: float,
    n_iters: int = 10,
    solver: str = "naive",
    start_params: np.ndarray = None,
    report_normalized: bool = False,
    penalty_scaling: Union[np.ndarray, None] = None,
    distribution="gaussian",
) -> GlmnetModel:
    """
    Parameters
    ----------
    y
    x
    alpha: Overall penalty multiplier. The glmnet paper calls this 'lambda'.
    l1_ratio: penalty on l1 part is multiplied by l1_ratio; penalty on l2 part is
        multiplied by 1 - l1_ratio.
    n_iters
    solver: 'naive' or 'sparse'
    start_params: starting parameters for an *unscaled* model
    report_normalized: Whether to keep the model in normalized form or not.
    penalty_scaling: Optional. Relative penalty. The penalty on parameter [i] is
        alpha * penalty_scaling[i] (times either l1_ratio or (1 - l1_ratio))
    """
    if start_params is None:
        start_params = np.zeros(x.shape[1])

    if solver == "naive":
        assert isinstance(x, np.ndarray)
        model = _do_cd_naive(
            y,
            x,
            alpha,
            l1_ratio,
            start_params,
            n_iters,
            report_normalized,
            penalty_scaling=penalty_scaling,
            distribution=distribution,
        )
    elif solver == "sparse":
        assert sps.issparse(x)
        model = _do_cd_sparse(
            y,
            x.tocsc(),
            alpha,
            l1_ratio,
            start_params,
            n_iters,
            report_normalized,
            penalty_scaling=penalty_scaling,
            distribution=distribution,
        )
    else:
        raise NotImplementedError

    return model


def sim_data(n_rows: int, n_cols: int, sparse: bool) -> Dict[str, Any]:
    intercept = -3
    np.random.seed(0)
    x = sps.random(n_rows, n_cols, format="csc")
    if not sparse:
        x = x.A
    true_coefs = np.random.normal(0, 1, n_cols)
    y = x.dot(true_coefs)
    y = y + intercept + np.random.normal(0, 1, n_rows)
    return {"y": y, "x": x, "coefs": true_coefs}


def test(sparse: bool):
    data = sim_data(10000, 1000, sparse)
    y = data["y"]
    print("\n\n")
    solver = "sparse" if sparse else "naive"
    print(solver)
    x = data["x"]

    start = time.time()
    model = fit_glmnet(y, x, 0.1, 0.5, n_iters=40, solver=solver)
    end = time.time()
    print("time", end - start)
    print("r2", model.get_r2(y))
    print("frac of coefs zero", (model.params == 0).mean())


if __name__ == "__main__":
    test(sparse=False)
    test(sparse=True)
