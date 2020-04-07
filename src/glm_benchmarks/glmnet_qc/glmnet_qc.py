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

from .util import spmatrix_col_sd


class GlmnetGaussianModel:
    def __init__(
        self,
        y: np.ndarray,
        x: Union[np.ndarray, sps.spmatrix],
        alpha: float,
        l1_ratio: float,
        intercept: float = 0.0,
        params: np.ndarray = None,
        penalty_scaling: np.ndarray = None,
        is_x_zero_centered: bool = False,
        is_x_squared_mean_one: bool = False,
    ):
        """
        Assume x does *not* include an intercept.
        """
        if alpha < 0:
            raise ValueError("alpha must be positive.")
        if not 0 <= l1_ratio <= 1:
            raise ValueError("l1_ratio must be between zero and one.")
        if not y.shape == (x.shape[0],):
            raise ValueError("y has the wrong shape")
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
        if penalty_scaling is None:
            penalty_scaling = np.ones(x.shape[1])
        elif (penalty_scaling < 0).any():
            raise ValueError("penalty_scaling must be non-negative.")
        self.penalty_scaling = penalty_scaling
        self.is_x_zero_centered = is_x_zero_centered
        self.is_x_squared_mean_one = is_x_squared_mean_one
        self.original_x_mean = np.squeeze(np.asarray(self.x.mean(0)))

        if sps.issparse(x):
            self.original_x_sd = spmatrix_col_sd(x)
        else:
            self.original_x_sd = self.x.std(0)

    def predict(self):
        return self.intercept + self.x.dot(self.params)

    def set_optimal_intercept(self):
        resid = self.y - self.predict()
        self.intercept += resid.mean()

    def scale_to_mean_squared_one(self):
        if sps.issparse(self.x):
            x_squared = self.x.power(2)
        else:
            x_squared = self.x ** 2

        scale_factor = np.squeeze(np.asarray(np.sqrt(x_squared.sum(0) / len(self.y))))

        if sps.issparse(self.x):
            self.x = self.x.multiply(sps.csc_matrix(1 / scale_factor))
        else:
            self.x /= scale_factor[None, :]
        self.params *= scale_factor
        self.is_x_squared_mean_one = True

    def rescale_to_original_sd(self):
        """
        Undoes scale_to_sum_squared_one, resetting scale to original
        """
        current_sd = spmatrix_col_sd(self.x) if sps.issparse(self.x) else self.x.std(0)
        scale_factor = self.original_x_sd / current_sd

        if sps.issparse(self.x):
            self.x = self.x.multiply(sps.csc_matrix(scale_factor))
        else:
            self.x *= scale_factor[None, :]
        self.params /= scale_factor
        self.is_x_squared_mean_one = False

    def center_around_zero(self):
        if sps.issparse(self.x):
            # Make x dense
            self.x = self.x.A
        self.x -= self.original_x_mean[None, :]
        self.is_x_zero_centered = True

    def shift_to_original_centering(self):
        """
        Undoes center_around_zero, resetting mean to original
        """
        current_mean = np.squeeze(np.asarray(self.x.mean(0)))
        shifter = self.original_x_mean - current_mean

        if sps.issparse(self.x):
            # densify
            self.x = self.x.A

        self.x += shifter[None, :]
        self.is_x_zero_centered = False

    def normalize(self):
        self.center_around_zero()
        self.scale_to_mean_squared_one()

    def undo_normalize(self):
        self.rescale_to_original_sd()
        self.shift_to_original_centering()


def soft_threshold(z: float, threshold: float) -> Union[float, int]:
    if np.abs(z) <= threshold:
        return 0
    if z > 0:
        return z - threshold
    return z + threshold


def _get_new_param(
    mean_resid_times_x: float, model: GlmnetGaussianModel, j: int
) -> float:
    numerator = soft_threshold(
        mean_resid_times_x, model.l1_ratio * model.alpha * model.penalty_scaling[j]
    )
    if numerator == 0:
        new_param_j = 0.0
    else:
        denominator = 1 + model.alpha * model.penalty_scaling[j] * (1 - model.l1_ratio)
        new_param_j = numerator / denominator
    return new_param_j


def _get_coordinate_wise_update_naive(
    model: GlmnetGaussianModel, j: int, resid: np.ndarray,
) -> Tuple[float, np.ndarray]:

    x_j = model.x[:, j]
    mean_resid_times_x = x_j.dot(resid) / len(model.y) + model.params[j]
    new_param_j = _get_new_param(mean_resid_times_x, model, j)

    if new_param_j != 0 or model.params[j] != 0:
        resid += x_j * (model.params[j] - new_param_j)
    return new_param_j, resid


def _get_coordinate_wise_update_sparse(
    model: GlmnetGaussianModel, j: int, resid: np.ndarray
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


def _do_cd_naive(
    y: np.ndarray,
    x: np.ndarray,
    alpha: float,
    l1_ratio: float,
    start_params: np.ndarray,
    n_iters: int,
    report_normalized=False,
    penalty_scaling: np.ndarray = None,
) -> GlmnetGaussianModel:
    """
    This is the "naive algorithm" from the paper.
    """
    model = GlmnetGaussianModel(
        y, x, alpha, l1_ratio, params=start_params, penalty_scaling=penalty_scaling,
    )
    model.normalize()
    model.set_optimal_intercept()

    resid = model.y - model.predict()

    for i in range(n_iters):
        for j in range(len(model.params)):
            model.params[j], resid = _get_coordinate_wise_update_naive(model, j, resid)

    if not report_normalized:
        model.undo_normalize()
        model.set_optimal_intercept()
    return model


def get_r2(model: GlmnetGaussianModel, y: np.ndarray) -> float:
    return 1 - np.var(y - model.predict()) / np.var(y)


def _do_cd_sparse(
    y: np.ndarray,
    x: sps.spmatrix,
    alpha: float,
    l1_ratio: float,
    start_params: np.ndarray,
    n_iters: int,
    report_normalized=False,
    penalty_scaling: np.ndarray = None,
) -> GlmnetGaussianModel:
    """
    The paper is not terribly clear on what to do here. It sounds like the
    variables are scaled but not centered to as not to alter the sparsity.

    In a medium-sized and very sparse example (10k rows, 10k cols, 0.1% nonzero),
    this takes about 44% of the time of the dense example, and does not reach quite
    as high an R-squared.
    """
    model = GlmnetGaussianModel(
        y, x, alpha, l1_ratio, params=start_params, penalty_scaling=penalty_scaling,
    )

    model.scale_to_mean_squared_one()
    model.set_optimal_intercept()
    resid = y - model.predict()

    for i in range(n_iters):
        for j in range(len(model.params)):
            model.params[j], resid = _get_coordinate_wise_update_sparse(model, j, resid)
        model.intercept += resid.mean()
        resid -= resid.mean()

    if not report_normalized:
        model.rescale_to_original_sd()
        model.set_optimal_intercept()

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


def fit_pathwise(
    y: np.ndarray,
    x: Union[np.ndarray, sps.spmatrix],
    l1_ratio: float,
    n_iters: int = 10,
    solver: str = "naive",
    alpha_path: np.ndarray = None,
    penalty_scaling: np.ndarray = None,
) -> GlmnetGaussianModel:
    """
    See documentation for fit_glmnet.
    """
    # TODO: stop normalizing and un-normalizing so much.
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
    penalty_scaling: np.ndarray = None,
) -> GlmnetGaussianModel:
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
    if penalty_scaling is None:
        penalty_scaling = np.ones(x.shape[1])

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
        )
    else:
        raise NotImplementedError

    return model
