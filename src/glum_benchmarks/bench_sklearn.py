import warnings
from typing import Any, Optional, Union

import numpy as np
from scipy import sparse as sps
from sklearn.linear_model import (
    ElasticNet,
    LogisticRegression,
    TweedieRegressor,
)

from .util import benchmark_convergence_tolerance, runtime

# TODO: Add cv for logistic and elastic net
# TODO: Design decision: do we set precompute = True ?


def _build_and_fit(model_args, fit_args):
    """Build and fit a sklearn regressor."""
    model_class = model_args.pop("_model_class")
    reg = model_class(**model_args)
    return reg.fit(**fit_args)


def sklearn_bench(
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
    Benchmark scikit-learn GLM regressors.

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

    result: dict[str, Any] = {}
    reg_strength = alpha if reg_multiplier is None else alpha * reg_multiplier

    if "offset" in dat.keys():
        warnings.warn("sklearn doesn't support offset, skipping this problem.")
        return result

    n_samples = dat["X"].shape[0]

    if distribution == "gaussian":
        model_args = {
            "_model_class": ElasticNet,
            "alpha": reg_strength,
            "l1_ratio": l1_ratio,
            "fit_intercept": True,
            "max_iter": 1000,
            "tol": benchmark_convergence_tolerance,
            "precompute": True,
        }
    elif distribution == "binomial":
        # sklearn's LogisticRegression uses C = 1/lambda where lambda is the
        # regularization strength. However, sklearn does NOT scale the regularization
        # by n_samples, while glum's objective is: mean(loss) + alpha * penalty.
        # To match glum's objective, we need: C = 1 / (alpha * n_samples)
        C_value = 1.0 / (reg_strength * n_samples) if reg_strength > 0 else 1e10
        # Determine penalty type based on l1_ratio
        # Penalty is deprecated but we use it here as we run on sk learn 1.6.1 due
        # to the fact that h2o-py requires Python <3.10
        if l1_ratio == 0.0:
            penalty = "l2"
        elif l1_ratio == 1.0:
            penalty = "l1"
        else:
            penalty = "elasticnet"

        model_args = {
            "_model_class": LogisticRegression,
            "C": C_value,
            "penalty": penalty,
            # Use lbfgs for L2 (faster and more reliable), saga for L1/elasticnet
            "solver": "lbfgs" if penalty == "l2" else "saga",
            "fit_intercept": True,
            "max_iter": 1000,
            "tol": benchmark_convergence_tolerance,
        }
        # Only pass l1_ratio for elastic net
        if penalty == "elasticnet":
            model_args["l1_ratio"] = l1_ratio
    else:
        if "tweedie" in distribution:
            power = float(distribution.split("-p=")[1])
        elif distribution == "poisson":
            power = 1.0
        elif distribution == "gamma":
            power = 2.0

        # sklearn's TweedieRegressor only supports L2 regularization
        if l1_ratio > 0:
            warnings.warn(
                f"sklearn only supports L2 regularization for {distribution}, skipping."
            )
            return result

        model_args = {
            "_model_class": TweedieRegressor,
            "power": power,
            "alpha": reg_strength,
            "fit_intercept": True,
            "max_iter": 1000,
            "tol": benchmark_convergence_tolerance,
            "solver": "newton-cholesky",
        }

    fit_args = {"X": dat["X"], "y": dat["y"]}

    if "sample_weight" in dat.keys():
        fit_args["sample_weight"] = dat["sample_weight"]

    try:
        result["runtime"], m = runtime(_build_and_fit, iterations, model_args, fit_args)
    except ValueError as e:
        warnings.warn(f"Problem failed with this error: {e}")
        return result

    if distribution == "binomial":
        result["intercept"] = m.intercept_[0] if m.intercept_.ndim > 0 else m.intercept_
        result["coef"] = m.coef_.ravel()
        result["n_iter"] = m.n_iter_[0] if hasattr(m.n_iter_, "__len__") else m.n_iter_
    else:
        result["intercept"] = m.intercept_
        result["coef"] = m.coef_
        result["n_iter"] = m.n_iter_ if hasattr(m, "n_iter_") else None

    return result
