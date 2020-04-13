from typing import Union

import numpy as np
from scipy import sparse as sps

from glm_benchmarks.glmnet_qc.util import default_links, get_link_and_inverse


class GlmnetModel:
    def __init__(
        self,
        y: np.ndarray,
        x: Union[np.ndarray, sps.spmatrix],
        distribution: str,
        alpha: float,
        l1_ratio: float,
        weights: np.ndarray = None,
        params: np.ndarray = None,
        penalty_scaling: np.ndarray = None,
        link_name: str = None,
    ):
        """
        Assume x includes an intercept as the first column
        """
        # TODO: add in intercept instead of assuming it is there
        if alpha < 0:
            raise ValueError("alpha must be positive.")
        if not 0 <= l1_ratio <= 1:
            raise ValueError("l1_ratio must be between zero and one.")
        self.distribution = distribution
        if not y.shape == (x.shape[0],):
            raise ValueError("y has the wrong shape")
        self.y = y
        self.x = x
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        if weights is None:
            self.weights = np.ones_like(y) / len(y)
        else:
            self.weights = weights
        self.weight_sum = self.weights.sum()
        if params is None:
            params = np.zeros(x.shape[1])
        else:
            if not params.shape == (x.shape[1],):
                raise ValueError(
                    f"""params should have shape {x.shape[1],}. Actual shape
                    {params.shape}"""
                )
        self.params = params
        if penalty_scaling is None:
            penalty_scaling = np.ones(x.shape[1])
            penalty_scaling[0] = 0
        elif (penalty_scaling < 0).any():
            raise ValueError("penalty_scaling must be non-negative.")
        self.penalty_scaling = penalty_scaling
        self.link_name = default_links[distribution] if link_name is None else link_name
        self.link, self.inv_link = get_link_and_inverse(self.link_name)

    def predict(self):
        return self.inv_link(self.x.dot(self.params))

    def get_r2(self, y: np.ndarray) -> float:
        return 1 - np.var(y - self.predict()) / np.var(y)


class GaussianCanonicalModel(GlmnetModel):
    def __init__(
        self,
        y: np.ndarray,
        x: Union[np.ndarray, sps.spmatrix],
        alpha: float,
        l1_ratio: float,
        weights: np.ndarray = None,
        params: np.ndarray = None,
        penalty_scaling: np.ndarray = None,
    ):
        super().__init__(
            y, x, "gaussian", alpha, l1_ratio, weights, params, penalty_scaling
        )

    def set_optimal_intercept(self):
        resid = self.y - self.predict()
        self.params[0] += self.weights.dot(resid) / self.weight_sum


def update_params(model: GlmnetModel, params: np.ndarray = None) -> GlmnetModel:
    new_params = model.params if params is None else params
    return GlmnetModel(
        model.y,
        model.x,
        model.distribution,
        model.alpha,
        model.l1_ratio,
        params=new_params,
        penalty_scaling=model.penalty_scaling,
        link_name=model.link_name,
    )
