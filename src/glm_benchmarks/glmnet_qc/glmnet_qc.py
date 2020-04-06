"""
This implements the algorithm given in the glmnet paper.
"""
import time
from typing import Any, Dict, Tuple, Union

import numpy as np
from scipy import sparse as sps


class GlmnetGaussianModel:
    def __init__(
        self,
        y: np.ndarray,
        x: Union[np.ndarray, sps.spmatrix],
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
            params = np.zeros(x.shape[1])
        self.params = params
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


def _get_coordinate_wise_update_naive(
    model: GlmnetGaussianModel, j: int, resid: np.ndarray
) -> Tuple[float, np.ndarray]:
    x_j = model.x[:, j]

    # Equation 8. 54% of time
    mean_resid_times_x = x_j.dot(resid) / len(model.y) + model.params[j]
    numerator = soft_threshold(mean_resid_times_x, model.l1_ratio * model.alpha)
    if numerator == 0:
        new_param_j = 0.0
    else:
        denominator = 1 + model.alpha * (1 - model.l1_ratio)
        new_param_j = numerator / denominator

    if new_param_j != 0 or model.params[j] != 0:
        resid += x_j * (model.params[j] - new_param_j)
    return new_param_j, resid


def get_col_j(x: sps.spmatrix, j: int) -> Tuple[np.ndarray, np.ndarray]:
    nz_indices = x.indices[x.indptr[j] : x.indptr[j + 1]]
    nz_data = x.data[x.indptr[j] : x.indptr[j + 1]]
    return nz_indices, nz_data


def _get_coordinate_wise_update_sparse(
    model: GlmnetGaussianModel, j: int, resid: np.ndarray
) -> Tuple[float, np.ndarray]:
    nz_indices, x_j_nz = get_col_j(model.x, j)

    resid_nz = resid[nz_indices]
    grad_part = x_j_nz.dot(resid_nz)

    mean_resid_times_x = grad_part / len(model.y) + model.params[j]
    numerator = soft_threshold(mean_resid_times_x, model.l1_ratio * model.alpha)
    if numerator == 0:
        new_param_j = 0.0
    else:
        denominator = 1 + model.alpha * (1 - model.l1_ratio)
        new_param_j = numerator / denominator

    if new_param_j != 0 or model.params[j] != 0:
        resid[nz_indices] += x_j_nz * (model.params[j] - new_param_j)

    return new_param_j, resid


def _do_cd_naive(
    y: np.ndarray, x: np.ndarray, alpha: float, l1_ratio: float, n_iters: int = 10,
) -> GlmnetGaussianModel:
    """
    This is the "naive algorithm" from the paper.
    """
    x_standardized = (x - x.mean(0)[None, :]) / x.std(0)[None, :]
    model = GlmnetGaussianModel(y, x_standardized, alpha, l1_ratio, intercept=y.mean())
    resid = model.y - model.predict()

    for i in range(n_iters):
        for j in range(len(model.params)):
            model.params[j], resid = _get_coordinate_wise_update_naive(model, j, resid)

    return model


def r2(model: GlmnetGaussianModel, y: np.ndarray) -> float:
    return 1 - np.var(y - model.predict()) / np.var(y)


def _do_cd_sparse(
    y: np.ndarray, x: sps.spmatrix, alpha: float, l1_ratio: float, n_iters: int
) -> GlmnetGaussianModel:
    """
    The paper is not terribly clear on what to do here. It sounds like the
    variables are scaled but not centered to as not to alter the sparsity.

    In a medium-sized and very sparse example (10k rows, 10k cols, 0.1% nonzero),
    this takes about 44% of the time of the dense example, and does not reach quite
    as high an R-squared.
    """
    z = sps.csc_matrix(1 / (np.sqrt(x.power(2).sum(0) / x.shape[0])))
    x_scaled = x.multiply(z)
    model = GlmnetGaussianModel(y, x_scaled, alpha, l1_ratio, intercept=y.mean())
    resid = y - model.predict()

    for i in range(n_iters):
        for j in range(len(model.params)):
            model.params[j], resid = _get_coordinate_wise_update_sparse(model, j, resid)
        model.intercept += resid.mean()
        resid -= resid.mean()

    return model


def fit_glmnet(
    y: np.ndarray,
    x: Union[np.ndarray, sps.csc_matrix],
    alpha: float,
    l1_ratio: float,
    n_iters: int = 10,
    solver: str = "naive",
) -> GlmnetGaussianModel:
    if solver == "naive":
        model = _do_cd_naive(y, x, alpha, l1_ratio, n_iters)
    elif solver == "sparse":
        model = _do_cd_sparse(y, x, alpha, l1_ratio, n_iters)
    else:
        raise NotImplementedError

    return model


def sim_data(n_rows: int, n_cols: int, sparse: bool) -> Dict[str, Any]:
    intercept = -3
    np.random.seed(0)
    x = sps.random(n_rows, n_cols, density=0.001, format="csc")
    if not sparse:
        x = x.A
    true_coefs = np.random.normal(0, 1, n_cols)
    y = x.dot(true_coefs)
    y = y + intercept + np.random.normal(0, 1, n_rows)
    return {"y": y, "x": x, "coefs": true_coefs}


def test(sparse: bool):
    data = sim_data(10000, 10000, sparse)
    y = data["y"]
    print("\n\n")
    solver = "sparse" if sparse else "naive"
    print(solver)
    x = data["x"]

    start = time.time()
    model = fit_glmnet(y, x, 0.1, 0.5, n_iters=40, solver=solver)
    end = time.time()
    print("time", end - start)
    print("r2", r2(model, y))
    print("frac of coefs zero", (model.params == 0).mean())


if __name__ == "__main__":
    test(sparse=False)
    test(sparse=True)
