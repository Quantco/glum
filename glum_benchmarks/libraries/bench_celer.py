import warnings
from typing import Any, Optional, Union

import numpy as np
from celer import ElasticNet, Lasso
from scipy import sparse as sps

from glum_benchmarks.util import benchmark_convergence_tolerance, runtime


def _build_and_fit(model_class, model_args, fit_args):
    return model_class(**model_args).fit(**fit_args)


def celer_bench(
    dat: dict[str, Union[np.ndarray, sps.spmatrix]],
    distribution: str,
    alpha: float,
    l1_ratio: float,
    iterations: int,
    reg_multiplier: Optional[float] = None,
    **kwargs,
):
    result: dict[str, Any] = {}
    fit_args = {"X": dat["X"], "y": dat["y"]}
    reg_strength = alpha if reg_multiplier is None else alpha * reg_multiplier

    # LogisticRegression doesnt support fitting an intercept yet, so we also skip it
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
        "max_iter": 1000,
        "alpha": reg_strength,
    }

    if model_class == ElasticNet:
        model_args["l1_ratio"] = l1_ratio

    try:
        result["runtime"], m = runtime(
            _build_and_fit, iterations, model_class, model_args, fit_args
        )
    except Exception as e:
        warnings.warn(f"Celer failed: {e}")
        return {}

    result["intercept"] = np.array(m.intercept_).ravel()[0]
    result["coef"] = np.array(m.coef_).ravel()
    result["n_iter"] = getattr(m, "n_iter_", None)

    return result
