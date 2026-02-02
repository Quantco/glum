import logging
import os
import warnings
from typing import Optional, Union

import h2o
import numpy as np
import pandas as pd
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from scipy import sparse as sps

from glum_benchmarks.util import (
    benchmark_convergence_tolerance,
    runtime,
)

# Suppress H2O's "Closing connection" messages at exit
logging.getLogger("h2o").setLevel(logging.WARNING)


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
    reg_multiplier: Optional[float] = None,
    standardize: bool = True,
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
    reg_multiplier
    standardize
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

    h2o.init(
        nthreads=int(os.environ.get("OMP_NUM_THREADS", os.cpu_count())),  # type: ignore
        verbose=False,
    )
    h2o.no_progress()  # Suppress progress bars

    train_mat = _hstack_sparse_or_dense((dat["X"], dat["y"][:, np.newaxis]))
    train_h2o = h2o.H2OFrame(train_mat)

    # y column is the last column
    y_col = train_h2o.col_names[-1]

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
        standardize=standardize,  # Let h2o handle standardization internally
        solver="IRLSM",
        objective_epsilon=benchmark_convergence_tolerance,
        beta_epsilon=benchmark_convergence_tolerance,
        gradient_epsilon=benchmark_convergence_tolerance,
        gainslift_bins=0,
    )

    if tweedie:
        p = float(distribution.split("=")[-1])
        model_args["tweedie_variance_power"] = p
        model_args["tweedie_link_power"] = 1 if p == 0 else 0
    if "gamma" in distribution:
        model_args["link"] = "Log"

    train_args = dict(
        x=train_h2o.col_names[:-1],
        y=y_col,
        training_frame=train_h2o,
    )

    result["runtime"], m = runtime(_build_and_fit, iterations, model_args, train_args)
    # un-standardize
    standardized_intercept = m.coef()["Intercept"]

    # Number of X columns (excluding y)
    n_x_cols = train_mat.shape[1] - 1
    standardized_coefs = np.array(
        [
            # h2o automatically removes zero-variance columns; impute to 1
            m.coef().get(f"C{i + 1}", 0)
            for i in range(n_x_cols)
        ]
    )

    result["intercept"] = standardized_intercept
    result["coef"] = standardized_coefs

    result["n_iter"] = m.score_history().iloc[-1]["iterations"]
    # h2o default max_iterations is very high, but we can get actual from params
    result["max_iter"] = m.actual_params.get("max_iterations", 100)
    return result
