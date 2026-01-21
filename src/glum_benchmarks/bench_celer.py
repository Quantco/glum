import warnings
from typing import Any, Optional, Union

import numpy as np
from celer import ElasticNet, ElasticNetCV, Lasso, LassoCV
from scipy import sparse as sps

from .util import benchmark_convergence_tolerance, runtime

# TODO: Add weights
# TODO: ADD CV
# TODO: Different results (due to objective function?)


def _build_and_fit(model_args, fit_args, cv: bool):
    model_class = model_args.pop("_model_class")
    reg = model_class(**model_args)
    return reg.fit(**fit_args)


def celer_bench(
    dat: dict[str, Union[np.ndarray, sps.spmatrix]],
    distribution: str,
    alpha: float,
    l1_ratio: float,
    iterations: int,
    cv: bool,
    reg_multiplier: Optional[float] = None,
    **kwargs,
):
    result: dict[str, Any] = {}
    reg_strength = alpha if reg_multiplier is None else alpha * reg_multiplier

    if distribution not in ["gaussian"]:
        warnings.warn(f"Celer doesn't support {distribution}, skipping.")
        return {}

    if "offset" in dat:
        warnings.warn("Celer doesn't support offsets, skipping.")
        return {}

    if "sample_weight" in dat:
        warnings.warn("Celer sample_weight not implemented in benchmark, skipping.")
        return {}

    if distribution == "gaussian":
        if l1_ratio == 0.0:
            warnings.warn("Celer doesn't support Ridge (l1_ratio=0), skipping.")
            return {}
        elif l1_ratio < 1.0:
            model_class = ElasticNetCV if cv else ElasticNet
        else:
            model_class = LassoCV if cv else Lasso

    model_args = {
        "_model_class": model_class,
        "tol": benchmark_convergence_tolerance,
        "fit_intercept": True,
        "max_iter": 1000,
    }

    if model_class in (ElasticNet, ElasticNetCV):
        model_args["l1_ratio"] = l1_ratio

    if not cv:
        model_args["alpha"] = reg_strength

    fit_args = {"X": dat["X"], "y": dat["y"]}

    try:
        result["runtime"], m = runtime(
            _build_and_fit, iterations, model_args, fit_args, cv
        )
    except Exception as e:
        warnings.warn(f"Celer failed: {e}")
        return {}

    result["intercept"] = np.array(m.intercept_).ravel()[0]
    result["coef"] = np.array(m.coef_).ravel()
    result["n_iter"] = getattr(m, "n_iter_", None)

    if cv:
        result["best_alpha"] = m.alpha_
        result["n_alphas"] = len(m.alphas_)
        result["max_alpha"] = m.alphas_.max()
        result["min_alpha"] = m.alphas_.min()

    return result
