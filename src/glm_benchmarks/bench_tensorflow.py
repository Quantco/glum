from typing import Any, Dict, Union

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp

from glm_benchmarks.problems import load_simple_insurance_data
from glm_benchmarks.util import runtime


def map_distribution_to_tf_dist(distribution: str):
    if distribution.lower() == "poisson":
        return tfp.glm.Poisson()
    elif distribution.lower() == "gaussian":
        return tfp.glm.Normal()
    else:
        raise NotImplementedError


def tensorflow_bench(
    dat: Dict[str, Union[pd.Series, pd.DataFrame]],
    distribution: str,
    alpha: float,
    l1_ratio: float,
) -> Dict[str, Any]:

    result = dict()
    x = tf.convert_to_tensor(
        np.hstack((np.ones((dat["X"].shape[0], 1)), dat["X"].values))
    )
    y = tf.convert_to_tensor(dat["y"].values)

    t, fit = runtime(
        tfp.glm.fit_sparse,
        model_matrix=x,
        response=y,
        model=map_distribution_to_tf_dist(distribution),
        model_coefficients_start=tf.zeros(x.shape[1], dtype=tf.float64),
        tolerance=1.0,
        l1_regularizer=alpha * l1_ratio,
        l2_regularizer=alpha * (1 - l1_ratio),
        maximum_iterations=10,
    )
    model_coefficients, is_converged, iter_ = fit
    result["runtime"] = t
    result["model_obj"] = fit
    result["intercept"] = model_coefficients.numpy()[0]
    result["coef"] = model_coefficients.numpy()[1:]
    result["n_iter"] = iter_.numpy()
    print(result)
    return result


def main():
    dat = load_simple_insurance_data(1000)
    x = dat["X"]
    print(type(x.values))
    print(x.shape)
    tensorflow_bench(dat)


if __name__ == "__main__":
    main()
