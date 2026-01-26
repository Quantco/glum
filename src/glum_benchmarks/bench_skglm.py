import warnings
from typing import Any, Optional, Union, cast

import numpy as np
from scipy import sparse as sps
from skglm import GeneralizedLinearEstimator, GeneralizedLinearEstimatorCV
from skglm.datafits import Gamma, Logistic, Poisson, Quadratic, WeightedQuadratic
from skglm.penalties import L1, L1_plus_L2
from skglm.solvers import AndersonCD, ProxNewton

from .util import benchmark_convergence_tolerance, runtime

# TODO: For CV the found alpha values differ significantly from glum
# TODO: Keep the data conversion? Runtime/2 for binomial


def _build_and_fit(model_args, fit_args, cv: bool):
    if cv:
        return GeneralizedLinearEstimatorCV(**model_args).fit(**fit_args)
    return GeneralizedLinearEstimator(**model_args).fit(**fit_args)


def skglm_bench(
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

    if "offset" in dat:
        warnings.warn("skglm doesn't support offsets, skipping.")
        return result

    if "tweedie" in distribution:
        warnings.warn("skglm doesn't support Tweedie, skipping.")
        return result

    DATAFITS = {
        "gaussian": Quadratic(),
        "poisson": Poisson(),
        "binomial": Logistic(),
        "gamma": Gamma(),
    }

    if "sample_weight" in dat.keys():
        if distribution == "gaussian":
            weights = np.asarray(dat["sample_weight"], dtype=np.float64)
            datafit = WeightedQuadratic(weights)
        else:
            warnings.warn(f"skglm doesn't support Weighted {distribution}, skipping.")
            return result
    else:
        datafit = DATAFITS[distribution]

    # For CV, alpha is determined internally, for non-CV, use reg_strength
    p_alpha = 1.0 if cv else reg_strength
    if l1_ratio == 1:
        penalty = L1(alpha=p_alpha)
    else:
        # We use L1_plus_L2(l1_ratio=0) for pure L2 to ensure prox_1d is available
        penalty = L1_plus_L2(alpha=p_alpha, l1_ratio=l1_ratio)

    # ProxNewton is required for Poisson/Gamma or L2 problems
    solver_tol = 1.0 if cv else benchmark_convergence_tolerance
    if distribution in ["poisson", "gamma"] or l1_ratio == 0:
        solver = ProxNewton(tol=solver_tol, fit_intercept=True, max_iter=1000)
    else:
        solver = AndersonCD(tol=solver_tol, fit_intercept=True, max_iter=1000)

    model_args = {
        "datafit": datafit,
        "penalty": penalty,
        "solver": solver,
    }

    if cv:
        # Same CV parameters as glum
        model_args.update({"cv": 5, "n_alphas": 100, "eps": 1e-6, "n_jobs": 1})

    # Data Conversion optimized for Coordinate Descent
    X_raw = dat["X"]
    if sps.issparse(X_raw):
        X = cast(sps.spmatrix, X_raw).tocsc()
    else:
        X = np.asfortranarray(X_raw, dtype=np.float64)

    y = np.asarray(dat["y"], dtype=np.float64).ravel()
    fit_args = {"X": X, "y": y}

    try:
        result["runtime"], m = runtime(
            _build_and_fit, iterations, model_args, fit_args, cv
        )
    except Exception as e:
        warnings.warn(f"skglm failed: {e}")
        return {}

    result["intercept"] = np.array(m.intercept_).ravel()[0]
    result["coef"] = np.array(m.coef_).ravel()
    if cv:
        n_iter_data = getattr(m, "n_iter_", None)
        result["n_iter"] = np.sum(n_iter_data) if n_iter_data is not None else None
        result["best_alpha"] = getattr(m, "alpha_", None)
        if hasattr(m, "alphas_"):
            result["n_alphas"] = len(m.alphas_)
            result["max_alpha"] = m.alphas_.max()
            result["min_alpha"] = m.alphas_.min()
    else:
        result["n_iter"] = getattr(m, "n_iter_", None)

    return result
