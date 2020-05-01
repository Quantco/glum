import os
from typing import Dict, Union

import h2o
import numpy as np
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from scipy import sparse as sps

from .util import benchmark_convergence_tolerance, runtime


def build_and_fit(model_args, train_args):
    glm = H2OGeneralizedLinearEstimator(**model_args)
    glm.train(**train_args)
    return glm


def hstack_sparse_or_dense(to_stack):
    if sps.isspmatrix(to_stack[0]):
        return sps.hstack(to_stack)
    else:
        return np.hstack(to_stack)


def h2o_bench(
    dat: Dict[str, Union[np.ndarray, sps.spmatrix]],
    distribution: str,
    alpha: float,
    l1_ratio: float,
    iterations: int,
):

    h2o.init(nthreads=int(os.environ.get("OMP_NUM_THREADS", os.cpu_count())))  # type: ignore

    train_mat = hstack_sparse_or_dense((dat["X"], dat["y"][:, np.newaxis]))

    use_weights = "weights" in dat.keys()
    if use_weights:
        train_mat = hstack_sparse_or_dense((train_mat, dat["weights"][:, np.newaxis]))

    train_h2o = h2o.H2OFrame(train_mat)

    tweedie = distribution == "tweedie_p=1.5"
    model_args = dict(
        model_id="glm",
        # not sure if this is right
        family="tweedie" if tweedie else distribution,
        alpha=l1_ratio,
        lambda_=alpha,
        standardize=False,
        solver="IRLSM",
        objective_epsilon=benchmark_convergence_tolerance,
        beta_epsilon=benchmark_convergence_tolerance,
        gradient_epsilon=benchmark_convergence_tolerance,
        tweedie_variance_power=1.5 if tweedie else None,
    )

    if use_weights:
        train_args = dict(
            x=train_h2o.col_names[:-2],
            y=train_h2o.col_names[-2],
            training_frame=train_h2o,
            weights_column=train_h2o.col_names[-1],
        )
    else:
        train_args = dict(
            x=train_h2o.col_names[:-1],
            y=train_h2o.col_names[-1],
            training_frame=train_h2o,
        )

    result = dict()
    result["runtime"], m = runtime(build_and_fit, iterations, model_args, train_args)
    # un-standardize
    standardized_intercept = m.coef()["Intercept"]

    standardized_coefs = np.array(
        [
            # h2o automatically removes zero-variance columns; impute to 1
            m.coef().get(f"C{i + 1}", 0)
            for i in range(train_mat.shape[1] - (2 if use_weights else 1))
        ]
    )

    result["intercept"] = standardized_intercept
    result["coef"] = standardized_coefs
    result["n_iter"] = m.score_history().iloc[-1]["iterations"]
    return result
