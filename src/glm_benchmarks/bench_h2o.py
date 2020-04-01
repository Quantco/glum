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
    train_h2o = h2o.H2OFrame(train_np)

    model_args = dict(
        model_id="glm",
        family=distribution,
        alpha=l1_ratio,
        lambda_=alpha,
        standardize=False,
    )

    train_args = dict(
        x=train_h2o.col_names[:-1], y=train_h2o.col_names[-1], training_frame=train_h2o,
    )

    result = dict()
    result["runtime"], m = runtime(build_and_fit, model_args, train_args)
    result["model_obj"] = "h2o objects fail to pickle"
    result["intercept"] = m.coef()["Intercept"]
    result["coef"] = np.array(
        [m.coef()[f"C{i + 1}"] for i in range(train_np.shape[1] - 1)]
    )
    result["n_iter"] = m.score_history().iloc[-1]["iterations"]
    return result
