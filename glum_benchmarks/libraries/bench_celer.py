import warnings
from typing import Any, Optional, Union

import numpy as np
from celer import ElasticNet, Lasso
from scipy import sparse as sps

from glum_benchmarks.util import (
    _standardize_features,
    _unstandardize_coefficients,
    benchmark_convergence_tolerance,
    runtime,
)


def _build_and_fit(model_class, model_args, fit_args):
    return model_class(**model_args).fit(**fit_args)


def celer_bench(
    dat: dict[str, Union[np.ndarray, sps.spmatrix]],
    distribution: str,
    alpha: float,
    l1_ratio: float,
    iterations: int,
    standardize: bool = True,
    timeout: Optional[float] = None,
    **kwargs,
):
    # Standardize features if requested
    scaler = None
    scaled_indices = None
    if standardize:
        dat = dat.copy()
        dat["X"], scaler, scaled_indices = _standardize_features(dat["X"])

    result: dict[str, Any] = {}
    fit_args = {"X": dat["X"], "y": dat["y"]}

    # LogisticRegression doesn't support fitting an intercept yet, so we also skip it
    if distribution not in ["gaussian"]:
        warnings.warn(f"Celer doesn't support {distribution}, skipping.")
        return {}

    if distribution == "gaussian":
        if l1_ratio == 0.0:
            warnings.warn("Celer doesn't support Ridge Regression, skipping.")
            return {}
        elif l1_ratio < 1.0:
            model_class = ElasticNet
        else:
            model_class = Lasso

    model_args = {
        "tol": benchmark_convergence_tolerance,
        "fit_intercept": True,
        "alpha": alpha,
    }

    if model_class == ElasticNet:
        model_args["l1_ratio"] = l1_ratio

    try:
        result["runtime"], m = runtime(
            _build_and_fit,
            iterations,
            model_class,
            model_args,
            fit_args,
            timeout=timeout,
        )
    except TimeoutError:
        # Re-raise TimeoutError to allow proper timeout handling at higher level
        raise
    except Exception as e:
        warnings.warn(f"Celer failed: {e}")
        return {}

    intercept = np.array(m.intercept_).ravel()[0]
    coef = np.array(m.coef_).ravel()

    if standardize:
        # Unstandardize coefficients to match original data scale
        result["intercept"], result["coef"] = _unstandardize_coefficients(
            intercept, coef, scaler, scaled_indices
        )
    else:
        result["intercept"] = intercept
        result["coef"] = coef

    result["n_iter"] = getattr(m, "n_iter_", None)
    result["max_iter"] = m.max_iter  # For convergence detection

    return result
