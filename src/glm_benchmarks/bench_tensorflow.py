from typing import Any, Dict, Union

import numpy as np
import pandas as pd
import tensorflow_probability as tfp

from glm_benchmarks.problems import load_simple_insurance_data
from glm_benchmarks.util import runtime


def map_distribution_to_tf_dist(distribution: str):
    """

    Parameters
    ----------
    distribution: e.g. 'poisson', or 'gaussian', according to glmnet convention

    Returns
    -------
    tensorflow probability distribution

    """
    dists = {
        "poisson": tfp.glm.Poisson(),
        "gaussian": tfp.glm.Normal(),
        "binomial": tfp.glm.Bernoulli(),
    }
    return dists[distribution]


def tensorflow_bench(
    dat: Dict[str, Union[pd.Series, pd.DataFrame]],
    distribution: str,
    alpha: float,
    l1_ratio: float,
) -> Dict[str, Any]:
    if l1_ratio != 0:
        raise NotImplementedError

    result = dict()
    x = np.hstack((np.ones((dat["X"].shape[0], 1)), dat["X"].values))
    t, fit = runtime(
        tfp.glm.fit,
        model_matrix=x,
        response=dat["y"].values,
        model=map_distribution_to_tf_dist(distribution),
        l2_regularizer=alpha,
    )
    model_coefficients, predicted_linear_response, is_converged, iter_ = fit
    result["runtime"] = t
    result["model_obj"] = fit
    result["intercept"] = model_coefficients[0]
    result["coef"] = model_coefficients[1]
    result["n_iter"] = iter_
    return result


def main():
    dat = load_simple_insurance_data(1000)
    x = dat["X"]
    print(type(x.values))
    print(x.shape)
    tensorflow_bench(dat)


if __name__ == "__main__":
    main()
