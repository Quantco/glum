import warnings
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
from scipy import sparse as sps
from sklearn.linear_model import LogisticRegression

from .util import benchmark_convergence_tolerance, runtime


def _build_and_fit(model_args, train_args):
    return LogisticRegression(**model_args).fit(**train_args)


def liblinear_bench(
    dat: Dict[str, Union[sps.spmatrix, np.ndarray]],
    distribution: str,
    alpha: float,
    l1_ratio: float,
    iterations: int,
    cv: bool,
    reg_multiplier: Optional[float] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Run the benchmark for sklearn.linear_model.LogisticRegression.

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
    dict

    """
    result: Dict = {}

    X = dat["X"]
    if not isinstance(X, (np.ndarray, sps.spmatrix, pd.DataFrame)):
        warnings.warn(
            "liblinear requires data as scipy.sparse matrix, pandas dataframe, or numpy "
            "array. Skipping."
        )
        return result

    if distribution != "binomial":
        warnings.warn("liblinear only supports binomial")
        return result

    if l1_ratio == 1 and alpha > 0:
        pen = "l1"
    elif l1_ratio == 0 and alpha > 0:
        pen = "l2"
    else:
        warnings.warn(
            "liblinear only supports lasso and ridge regression with positive alpha"
        )
        return result

    if "offset" in dat.keys():
        warnings.warn("liblinear does not support offsets")
        return result

    if cv:
        warnings.warn("liblinear does not yet support CV")
        return result

    model_args = dict(
        penalty=pen,
        tol=benchmark_convergence_tolerance,
        C=1 / (X.shape[0] * alpha)
        if reg_multiplier is None
        else 1 / (X.shape[0] * alpha * reg_multiplier),
        # Note that when an intercept is fitted, it is subject to regularization, unlike
        # other solvers. intercept_scaling helps combat this by inflating the intercept
        # column, though too low of a value leaves too much regularization and too high
        # of a value results in poor matrix properties.
        # See https://scikit-learn.org/stable/modules/generated/
        # sklearn.linear_model.LogisticRegression.html
        intercept_scaling=1e3,
        solver="liblinear",
    )

    fit_args = dict(
        X=X,
        y=dat["y"].astype(np.int64).copy(),
        sample_weight=dat.get("sample_weight"),
    )

    result["runtime"], m = runtime(_build_and_fit, iterations, model_args, fit_args)
    result["intercept"] = m.intercept_[0]
    result["coef"] = np.squeeze(m.coef_)
    result["n_iter"] = m.n_iter_[0]

    return result
