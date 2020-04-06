from typing import Dict, Union

import h2o
import numpy as np
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from scipy import sparse as sps

from .util import runtime


def build_and_fit(model_args, train_args):
    glm = H2OGeneralizedLinearEstimator(**model_args)
    glm.train(**train_args)
    return glm


def h2o_bench(
    dat: Dict[str, Union[np.ndarray, sps.spmatrix]],
    distribution: str,
    alpha: float,
    l1_ratio: float,
):
    h2o.init()

    train_np = np.hstack((dat["X"], dat["y"][:, np.newaxis]))

    use_weights = "weights" in dat.keys()
    if use_weights:
        train_np = np.hstack((train_np, dat["weights"][:, np.newaxis]))

    train_h2o = h2o.H2OFrame(train_np)

    model_args = dict(
        model_id="glm",
        family=distribution,
        alpha=l1_ratio,
        lambda_=alpha,
        standardize=True,
        # solver='COORDINATE_DESCENT',
        solver="IRLSM",
        objective_epsilon=1e-12,
        beta_epsilon=1e-12,
        gradient_epsilon=1e-12,
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
    result["runtime"], m = runtime(build_and_fit, model_args, train_args)
    result["model_obj"] = "h2o objects fail to pickle"

    # un-standardize
    standardized_intercept = m.coef()["Intercept"]
    standardized_coefs = np.array(
        [
            m.coef()[f"C{i + 1}"]
            for i in range(train_np.shape[1] - (2 if use_weights else 1))
        ]
    )
    # import ipdb
    # ipdb.set_trace()
    # coefs = standardized_coefs / (dat['X'].std(0))
    # intercept = (standardized_x - unstandardized_x).dot(coefs)
    # intercept = (dat['y'] - dat['X'].dot(coefs)).mean()

    result["intercept"] = standardized_intercept
    result["coef"] = standardized_coefs
    result["n_iter"] = m.score_history().iloc[-1]["iterations"]
    return result
