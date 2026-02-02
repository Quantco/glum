import warnings
from typing import Any, Optional, Union

import numpy as np
from scipy import sparse as sps
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LogisticRegression,
    Ridge,
    TweedieRegressor,
)

from glum_benchmarks.util import (
    _standardize_features,
    _unstandardize_coefficients,
    benchmark_convergence_tolerance,
    runtime,
)


def _build_and_fit(model_class, model_args, fit_args):
    """Build and fit a sklearn regressor."""
    return model_class(**model_args).fit(**fit_args)


def sklearn_bench(
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
    Benchmark scikit-learn GLM regressors.

    Parameters
    ----------
    dat
    distribution
    alpha
    l1_ratio
    iterations
    reg_multiplier
    standardize
        If True, standardize continuous features using sklearn's StandardScaler.
    kwargs

    Returns
    -------
    Dict of
    """
    # Standardize features if requested
    scaler = None
    scaled_indices = None
    if standardize:
        dat = dat.copy()
        dat["X"], scaler, scaled_indices = _standardize_features(dat["X"])

    result: dict[str, Any] = {}
    reg_strength = alpha if reg_multiplier is None else alpha * reg_multiplier

    n_samples = dat["X"].shape[0]
    model_class = None

    if distribution == "gaussian":
        if l1_ratio == 0.0:
            # Pure L2 (Ridge): use closed-form solution
            model_class = Ridge
            model_args = {
                "alpha": reg_strength * n_samples,
                "fit_intercept": True,
                "tol": benchmark_convergence_tolerance,
                "solver": "auto",
            }
        elif l1_ratio == 1.0:
            # Pure L1 (Lasso)
            model_class = Lasso
            model_args = {
                "alpha": reg_strength * n_samples,
                "fit_intercept": True,
                "tol": benchmark_convergence_tolerance,
                "precompute": True,
            }
        else:
            # Scale by n_samples to match glum's objective
            model_class = ElasticNet
            model_args = {
                "alpha": reg_strength * n_samples,
                "l1_ratio": l1_ratio,
                "fit_intercept": True,
                "tol": benchmark_convergence_tolerance,
                "precompute": True,
            }
    elif distribution == "binomial":
        # sklearn's LogisticRegression uses C = 1/lambda where lambda is the
        # regularization strength. However, sklearn does NOT scale the regularization
        # by n_samples, while glum's objective is: mean(loss) + alpha * reg_term.
        # To match glum's objective, we need: C = 1 / (alpha * n_samples)
        C_value = 1.0 / (reg_strength * n_samples) if reg_strength > 0 else 1e10
        # Use lbfgs for L2 (faster and more reliable), saga for L1/elasticnet
        solver = "lbfgs" if l1_ratio == 0.0 else "saga"

        model_class = LogisticRegression
        model_args = {
            "C": C_value,
            "l1_ratio": l1_ratio,
            "solver": solver,
            "fit_intercept": True,
            "tol": benchmark_convergence_tolerance,
        }
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

        model_class = TweedieRegressor
        model_args = {
            "power": power,
            "alpha": reg_strength,
            "fit_intercept": True,
            "tol": benchmark_convergence_tolerance,
            "solver": "newton-cholesky",
        }

    fit_args = {"X": dat["X"], "y": dat["y"]}

    if "sample_weight" in dat.keys():
        fit_args["sample_weight"] = dat["sample_weight"]

    try:
        result["runtime"], m = runtime(
            _build_and_fit, iterations, model_class, model_args, fit_args
        )
    except ValueError as e:
        warnings.warn(f"Problem failed with this error: {e}")
        return result

    if distribution == "binomial":
        intercept = m.intercept_[0] if m.intercept_.ndim > 0 else m.intercept_
        coef = m.coef_.ravel()
        n_iter = m.n_iter_[0] if hasattr(m.n_iter_, "__len__") else m.n_iter_
    else:
        intercept = m.intercept_
        coef = m.coef_
        # Ridge has n_iter_=None (closed-form), treat as 0 iterations (direct solve)
        n_iter = getattr(m, "n_iter_", None)
        n_iter = 0 if n_iter is None else n_iter

    # Unstandardize coefficients to match original data scale
    result["intercept"], result["coef"] = _unstandardize_coefficients(
        intercept, coef, scaler, scaled_indices
    )
    result["n_iter"] = n_iter
    # For convergence detection: get max_iter from model
    result["max_iter"] = getattr(m, "max_iter", None)

    return result
