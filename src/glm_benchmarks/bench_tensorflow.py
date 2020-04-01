import warnings
from typing import Any, Dict, Union

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from scipy import sparse as sps

from glm_benchmarks.util import runtime


def map_distribution_to_tf_dist(distribution: str):
    if distribution.lower() == "poisson":
        return tfp.glm.Poisson()
    elif distribution.lower() == "gaussian":
        return tfp.glm.Normal()
    else:
        raise NotImplementedError


def format_design_mat(x):
    # Add intercept
    if isinstance(x, sps.spmatrix):
        x = sps.hstack((np.ones(x.shape[0]), x))
    else:
        x = np.hstack((np.ones(x.shape[0]), x))
    x = tf.convert_to_tensor(x)
    return x


def tensorflow_bench(
    dat: Dict[str, Union[np.ndarray, sps.spmatrix]],
    distribution: str,
    alpha: float,
    l1_ratio: float,
) -> Dict[str, Any]:

    if "weights" in dat.keys():
        warnings.warn("Tensorflow doesn't support weights.")
        return {}

    result = dict()

    x = format_design_mat(dat["X"])
    y = tf.convert_to_tensor(dat["y"])

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
    return result
