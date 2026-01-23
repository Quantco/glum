import os
import warnings
from typing import Optional, Union

import h2o
import numpy as np
import pandas as pd
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from scipy import sparse as sps

from .util import benchmark_convergence_tolerance, runtime


def _build_and_fit(model_args, train_args):
    glm = H2OGeneralizedLinearEstimator(**model_args)
    glm.train(**train_args)
    return glm


def _hstack_sparse_or_dense(to_stack):
    if sps.isspmatrix(to_stack[0]):
        return sps.hstack(to_stack)
    else:
        return np.hstack(to_stack)


def h2o_bench(
    dat: dict[str, Union[np.ndarray, sps.spmatrix]],
    distribution: str,
    alpha: float,
    l1_ratio: float,
    iterations: int,
    cv: bool,
    reg_multiplier: Optional[float] = None,
    **kwargs,
):
    """
    Run a benchmark problem using h2o's glm.

    Parameters
    ----------
    dat
    distribution
    alpha
    l1_ratio
    iterations
    cv
    reg_multiplier
    kwargs

    Returns
    -------
    dict of data about this run
    """
    result: dict = {}

    if not isinstance(dat["X"], (np.ndarray, sps.spmatrix, pd.DataFrame)):
        warnings.warn(
            "h2o requires data as scipy.sparse matrix, pandas dataframe, or numpy "
            "array. Skipping."
        )
        return result

    h2o.init(nthreads=int(os.environ.get("OMP_NUM_THREADS", os.cpu_count())))  # type: ignore

    train_mat = _hstack_sparse_or_dense((dat["X"], dat["y"][:, np.newaxis]))

    use_weights = "sample_weight" in dat.keys()
    if use_weights:
        train_mat = _hstack_sparse_or_dense(
            (train_mat, dat["sample_weight"][:, np.newaxis])
        )
    if "offset" in dat.keys():
        train_mat = _hstack_sparse_or_dense((train_mat, dat["offset"][:, np.newaxis]))

    train_h2o = h2o.H2OFrame(train_mat)

    # Determine the y column index (it's right after X columns)
    n_extra_cols = int(use_weights) + int("offset" in dat.keys())
    y_col_idx = -(1 + n_extra_cols)
    y_col = train_h2o.col_names[y_col_idx]

    # For binomial, convert target to categorical
    if distribution == "binomial":
        # Round to int and convert to factor (h2o requires 0/1 int for binomial)
        train_h2o[y_col] = train_h2o[y_col].round().ascharacter().asfactor()

    tweedie = "tweedie" in distribution

    model_args = dict(
        model_id="glm",
        # not sure if this is right
        family="tweedie" if tweedie else distribution,
        alpha=l1_ratio,
        lambda_=alpha if reg_multiplier is None else alpha * reg_multiplier,
        standardize=False,
        solver="IRLSM",
        objective_epsilon=benchmark_convergence_tolerance,
        beta_epsilon=benchmark_convergence_tolerance,
        gradient_epsilon=benchmark_convergence_tolerance,
        max_iterations=1000,
        gainslift_bins=0,
    )
    if cv:
        model_args["lambda_search"] = True
        model_args["nfolds"] = 5

    if tweedie:
        p = float(distribution.split("=")[-1])
        model_args["tweedie_variance_power"] = p
        model_args["tweedie_link_power"] = 1 if p == 0 else 0
    if "gamma" in distribution:
        model_args["link"] = "Log"

    if use_weights:
        train_args = dict(
            x=train_h2o.col_names[:y_col_idx],
            y=y_col,
            training_frame=train_h2o,
            weights_column=train_h2o.col_names[y_col_idx + 1],
        )
        if "offset" in dat.keys():
            train_args["offset_column"] = train_h2o.col_names[-1]
    elif "offset" in dat.keys():
        train_args = dict(
            x=train_h2o.col_names[:y_col_idx],
            y=y_col,
            training_frame=train_h2o,
            offset_column=train_h2o.col_names[-1],
        )
    else:
        train_args = dict(
            x=train_h2o.col_names[:-1],
            y=y_col,
            training_frame=train_h2o,
        )

    result["runtime"], m = runtime(_build_and_fit, iterations, model_args, train_args)
    # un-standardize
    standardized_intercept = m.coef()["Intercept"]

    # Number of X columns (excluding y, weights, offset)
    n_x_cols = train_mat.shape[1] - (1 + n_extra_cols)
    standardized_coefs = np.array(
        [
            # h2o automatically removes zero-variance columns; impute to 1
            m.coef().get(f"C{i + 1}", 0)
            for i in range(n_x_cols)
        ]
    )
    if cv:
        result["best_alpha"] = m._model_json["output"]["lambda_best"]
        result["n_alphas"] = m.parms["nlambdas"]["actual_value"]

    result["intercept"] = standardized_intercept
    result["coef"] = standardized_coefs

    result["n_iter"] = m.score_history().iloc[-1]["iteration" if cv else "iterations"]
    return result
