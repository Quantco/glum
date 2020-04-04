"""
This implements the algorithm given in the glmnet paper.
"""
import time
from typing import Tuple, Union

import numpy as np
from scipy import sparse as sps


class GlmnetGaussianModel:
    def __init__(
        self,
        y: np.ndarray,
        x: np.ndarray,
        alpha: float,
        l1_ratio: float,
        intercept: float = 0.0,
        params: np.ndarray = None,
    ):
        """
        Assume x does *not* include an intercept.
        """
        assert alpha >= 0
        assert 0 <= l1_ratio <= 1
        self.y = y
        self.x = x
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        if params is None:
            self.params = np.zeros(x.shape[1])
        else:
            self.params = np.ma.masked_values(params, 0)
        self.intercept = intercept

    def predict(self):
        return self.intercept + self.x.dot(self.params)


def _get_ssr(model: GlmnetGaussianModel) -> float:
    resid = model.y - model.predict()
    return (resid ** 2).sum()


def soft_threshold(z: float, threshold: float) -> Union[float, int]:
    if np.abs(z) <= threshold:
        return 0
    if z > 0:
        return z - threshold
    return z + threshold


def get_coordinate_wise_update_naive(
    model: GlmnetGaussianModel, j: int, resid: np.ndarray
) -> Tuple[float, np.ndarray]:
    x_j = model.x[:, j]
    if model.params[j] == 0:
        resid_not_j = resid
    else:
        resid_not_j = resid + x_j * model.params[j]

    mean_resid_times_x = x_j.dot(resid_not_j) / len(model.y)
    numerator = soft_threshold(mean_resid_times_x, model.l1_ratio * model.alpha)
    if numerator == 0:
        new_param_j = 0.0
    else:
        denominator = 1 + model.alpha * (1 - model.l1_ratio)
        new_param_j = numerator / denominator

    new_resid = resid_not_j - x_j * new_param_j
    return new_param_j, new_resid


def cd_outer_update_naive(model: GlmnetGaussianModel) -> None:
    """
    This is the "naive algorithm" from the paper.
    """
    resid = model.y - model.predict()

    for i in range(len(model.params)):
        new_param, resid = get_coordinate_wise_update_naive(model, i, resid)
        model.params[i] = new_param


def _do_cd_naive(
    y: np.ndarray, x: np.ndarray, alpha: float, l1_ratio: float, n_iters: int = 10,
) -> GlmnetGaussianModel:
    x_mean = x.mean(0)
    x_sd = x.std(0)
    x_standardized = (x - x_mean[None, :]) / x_sd[None, :]
    model = GlmnetGaussianModel(y, x_standardized, alpha, l1_ratio, intercept=y.mean())

    for i in range(n_iters):
        cd_outer_update_naive(model)
    return model


def _do_cd_covariance(
    y: np.ndarray, x: np.ndarray, alpha: float, l1_ratio: float, n_iters: int = 10,
):
    pass


def fit_glmnet(
    y: np.ndarray,
    x: Union[np.ndarray, sps.csc_matrix],
    alpha: float,
    l1_ratio: float,
    n_iters: int = 10,
    solver: str = "naive",
) -> GlmnetGaussianModel:
    solvers = {"naive": _do_cd_naive, "covariance": _do_cd_covariance}

    model = solvers[solver](y, x, alpha, l1_ratio, n_iters)
    return model


def main():
    # 10000 cols: 19.97s, r2 = 0.841
    # 5000 cols: 8.697s, r2 = 0.805
    n_rows = 10000
    n_cols = 5000
    intercept = -3
    np.random.seed(0)
    x = np.random.normal(1, 2, (n_rows, n_cols))
    true_coefs = np.random.normal(0, 1, n_cols)
    y = intercept + x.dot(true_coefs) + np.random.normal(0, 1, n_rows)
    start = time.time()
    model = fit_glmnet(y, x, 1, 0.5, n_iters=5)
    end = time.time()
    print("time", end - start)
    print("intercept", model.intercept)
    ssr = _get_ssr(model)
    print("r2", 1 - ssr / (np.var(y) * len(y)))
    print("frac of coefs zero:")
    print((model.params == 0).mean())


if __name__ == "__main__":
    main()
