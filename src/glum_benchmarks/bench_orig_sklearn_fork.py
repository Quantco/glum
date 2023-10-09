import warnings
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from scipy import sparse as sps

from .orig_sklearn_fork import (  # type: ignore
    GeneralizedLinearRegressor,
    TweedieDistribution,
)
from .util import benchmark_convergence_tolerance, runtime

random_seed = 110


def _build_and_fit(model_args, fit_args):
    return GeneralizedLinearRegressor(**model_args).fit(**fit_args)


def orig_sklearn_fork_bench(
    dat: dict[str, Union[np.ndarray, sps.spmatrix]],
    distribution: str,
    alpha: float,
    l1_ratio: float,
    iterations: int,
    cv: bool,
    reg_multiplier: Optional[float] = True,
    **kwargs,
):
    """
    Benchmark the original sklearn-fork.

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
    Dict of
    """
    if cv:
        raise ValueError("original sklearn fork does not support cross-validation")
    result: dict[str, Any] = {}

    X = dat["X"]

    if X.shape[0] > 100000 and not isinstance(X, (np.ndarray, pd.DataFrame)):
        warnings.warn(
            "original sklearn fork is too slow on sparse data sets with more than "
            "100,000 rows. Skipping."
        )
        return result

    fit_args = dict(X=X, y=dat["y"])
    if "sample_weight" in dat.keys():
        fit_args.update({"sample_weight": dat["sample_weight"]})
    if "offset" in dat.keys():
        warnings.warn("Original sklearn_fork does not support offsets.")
        return result

    family = distribution
    if family == "gaussian":
        family = "normal"
    elif "tweedie" in family:
        tweedie_p = float(family.split("-p=")[1])
        family = TweedieDistribution(tweedie_p)  # type: ignore

    model_args = dict(
        family=family,
        alpha=alpha if reg_multiplier is None else alpha * reg_multiplier,
        l1_ratio=l1_ratio,
        max_iter=150,
        random_state=random_seed,
        copy_X=False,
        selection="cyclic",
        tol=benchmark_convergence_tolerance,
    )

    try:
        result["runtime"], m = runtime(_build_and_fit, iterations, model_args, fit_args)
    except ValueError as e:
        warnings.warn(f"Problem failed with this error: {e}")
        return result

    result["intercept"] = m.intercept_
    result["coef"] = m.coef_
    result["n_iter"] = m.n_iter_

    return result
