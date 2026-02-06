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
    standardize: bool = True,
    timeout: Optional[float] = None,
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
    n_samples = dat["X"].shape[0]
    model_class = None

    if distribution == "gaussian":
        if l1_ratio == 0.0:
            # Pure L2 (Ridge): sklearn uses sum-of-squared-errors loss while glum
            # uses mean-squared-error loss. To get the same optimal coefficients,
            # we scale alpha by n_samples to compensate for the 1/n factor in glum.
            model_class = Ridge
            model_args = {
                "alpha": alpha * n_samples,
                "fit_intercept": True,
                "tol": benchmark_convergence_tolerance,
                "solver": "auto",
            }
        elif l1_ratio == 1.0:
            # Pure L1
            model_class = Lasso
            model_args = {
                "alpha": alpha,
                "fit_intercept": True,
                "tol": benchmark_convergence_tolerance,
                "precompute": True,
            }
        else:
            model_class = ElasticNet
            model_args = {
                "alpha": alpha,
                "l1_ratio": l1_ratio,
                "fit_intercept": True,
                "tol": benchmark_convergence_tolerance,
                "precompute": True,
            }
    elif distribution == "binomial":
        # # sklearn uses sum(loss), glum uses mean(loss), so C = 1 / (alpha * n_samples)
        C_value = 1.0 / (alpha * n_samples) if alpha > 0 else 1e10

        # Newton-Cholesky is best choice for n_samples >> n_features
        # For saga to work, we need to scale the features
        solver = "newton-cholesky" if l1_ratio == 0.0 else "saga"

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

        # Use Newton-Cholesky as it is best choice for n_samples >> n_features and
        # one-hot encoded categorical features
        model_class = TweedieRegressor
        model_args = {
            "power": power,
            "alpha": alpha,
            "fit_intercept": True,
            "tol": benchmark_convergence_tolerance,
            "solver": "newton-cholesky",
        }

    fit_args = {"X": dat["X"], "y": dat["y"]}

    try:
        result["runtime"], m = runtime(
            _build_and_fit,
            iterations,
            model_class,
            model_args,
            fit_args,
            timeout=timeout,
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

    # Unstandardize coefficients if we standardized the features
    if standardize and scaler is not None:
        result["intercept"], result["coef"] = _unstandardize_coefficients(
            intercept, coef, scaler, scaled_indices
        )
    else:
        result["intercept"] = intercept
        result["coef"] = coef
    result["n_iter"] = n_iter
    # For convergence detection: get max_iter from model
    result["max_iter"] = getattr(m, "max_iter", None)

    return result
