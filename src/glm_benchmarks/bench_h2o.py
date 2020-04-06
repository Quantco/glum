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
        lambda_=alpha,  # 0.0006945007199887186,#alpha/1.44,
        standardize=False,
        objective_epsilon=1e-7,
        beta_epsilon=1e-7,
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
    result["intercept"] = m.coef()["Intercept"]
    n_coefs = train_np.shape[1] - (2 if use_weights else 1)
    result["coef"] = extract_coefs(m, n_coefs)
    result["n_iter"] = m.score_history().iloc[-1]["iterations"]
    # print('correct: ', )
    # print(np.sum(np.abs(result['coef'])))

    # def objfnc(L):
    #     MM = model_args.copy()
    #     MM['lambda_'] = L
    #     modelobj = build_and_fit(MM, train_args)
    #     coefs = extract_coefs(modelobj, n_coefs)
    #     l1 = np.sum(np.abs(coefs))
    #     print(L, l1)
    #     return l1 - 13.198392302008212

    # from scipy.optimize import bisect
    # LAMBDA = bisect(objfnc, alpha / 10.0, alpha * 3.0)
    # import ipdb
    # ipdb.set_trace()
    return result


def extract_coefs(m, n_coefs):
    return np.array([m.coef()[f"C{i + 1}"] for i in range(n_coefs)])
