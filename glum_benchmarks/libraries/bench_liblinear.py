import warnings
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from scipy import sparse as sps
from sklearn.linear_model import LogisticRegression

from glum_benchmarks.util import (
    _standardize_features,
    benchmark_convergence_tolerance,
    runtime,
)


def _build_and_fit(model_args, train_args):
    return LogisticRegression(**model_args).fit(**train_args)


def liblinear_bench(
    dat: dict[str, Union[sps.spmatrix, np.ndarray]],
    distribution: str,
    alpha: float,
    l1_ratio: float,
    iterations: int,
    reg_multiplier: Optional[float] = None,
    standardize: bool = True,
    max_iter: int = 1000,
    **kwargs,
) -> dict[str, Any]:
    """
    Run the benchmark for sklearn.linear_model.LogisticRegression.

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
    dict

    """
    # Standardize features if requested
    if standardize:
        dat = dat.copy()
        dat["X"] = _standardize_features(dat["X"])

    result: dict = {}

    X = dat["X"]
    if not isinstance(X, (np.ndarray, sps.spmatrix, pd.DataFrame)):
        warnings.warn(
            "liblinear requires data as scipy.sparse matrix, pandas dataframe, or "
            "numpy array. Skipping."
        )
        return result

    if distribution != "binomial":
        warnings.warn("liblinear only supports binomial")
        return result

    if l1_ratio == 1 and alpha > 0:
        sklearn_l1_ratio = 1.0
    elif l1_ratio == 0 and alpha > 0:
        sklearn_l1_ratio = 0.0
    else:
        warnings.warn(
            "liblinear only supports lasso and ridge regression with positive alpha"
        )
        return result

    model_args = dict(
        l1_ratio=sklearn_l1_ratio,
        tol=benchmark_convergence_tolerance,
        C=(
            1 / (X.shape[0] * alpha)
            if reg_multiplier is None
            else 1 / (X.shape[0] * alpha * reg_multiplier)
        ),
        # Note that when an intercept is fitted, it is subject to regularization, unlike
        # other solvers. intercept_scaling helps combat this by inflating the intercept
        # column, though too low of a value leaves too much regularization and too high
        # of a value results in poor matrix properties.
        # See https://scikit-learn.org/stable/modules/generated/
        # sklearn.linear_model.LogisticRegression.html
        intercept_scaling=1e3,
        solver="liblinear",
        max_iter=max_iter,
    )

    fit_args = dict(  # type: ignore
        X=X,
        y=dat["y"].astype(np.int64).copy(),
    )

    result["runtime"], m = runtime(_build_and_fit, iterations, model_args, fit_args)
    result["intercept"] = m.intercept_[0]
    result["coef"] = np.squeeze(m.coef_)
    result["n_iter"] = m.n_iter_[0]

    return result
