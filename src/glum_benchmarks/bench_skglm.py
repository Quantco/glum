import warnings
from typing import Any, Optional, Union

import numpy as np
from scipy import sparse as sps
from skglm import GeneralizedLinearEstimator, GeneralizedLinearEstimatorCV
from skglm.datafits import Gamma, Logistic, Poisson, Quadratic, WeightedQuadratic
from skglm.penalties import L1, L1_plus_L2
from skglm.solvers import AndersonCD, ProxNewton

from .util import benchmark_convergence_tolerance, runtime

# TODO: For Tweedie (p = 1.5), add custom datafit
# TODO: For CV the found alpha values differ significantly from glum


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

    DATAFITS = {
        "gaussian": Quadratic(),
        "poisson": Poisson(),
        "binomial": Logistic(),
        "gamma": Gamma(),
    }

    if distribution not in DATAFITS:
        warnings.warn(f"skglm doesn't support {distribution}, skipping.")
        return {}

    # For CV, alpha is determined internally, for non-CV, use reg_strength
    # We use L1_plus_L2(l1_ratio=0) for pure L2 to ensure prox_1d is available
    p_alpha = 1.0 if cv else reg_strength
    if l1_ratio == 1:
        penalty = L1(alpha=p_alpha)
    else:
        penalty = L1_plus_L2(alpha=p_alpha, l1_ratio=l1_ratio)

    # ProxNewton is required for Poisson/Gamma or smooth L2 problems.
    if distribution in ["poisson", "gamma"] or l1_ratio == 0:
        solver = ProxNewton(
            tol=benchmark_convergence_tolerance, fit_intercept=True, max_iter=1000
        )
    else:
        solver = AndersonCD(
            tol=benchmark_convergence_tolerance, fit_intercept=True, max_iter=1000
        )

    # Sample Weights currently only supported for Gaussian
    if "sample_weight" in dat:
        if distribution == "gaussian":
            weights = np.asarray(dat["sample_weight"], dtype=np.float64)
            datafit = WeightedQuadratic(weights)
        else:
            warnings.warn(f"Weighted {distribution} not natively supported, skipping.")
            return {}
    else:
        datafit = DATAFITS[distribution]

    model_args = {
        "datafit": datafit,
        "penalty": penalty,
        "solver": solver,
    }

    if cv:
        # Same CV parameters as glum
        model_args.update({"cv": 5, "n_alphas": 100, "n_jobs": 1})

    # Data Conversion
    X = (
        dat["X"]
        if sps.issparse(dat["X"])
        else np.ascontiguousarray(dat["X"], dtype=np.float64)
    )
    y = np.asarray(dat["y"], dtype=np.float64).ravel()
    fit_args = {"X": X, "y": y}

    # Numba Warm-up: Fit on a tiny subset to trigger JIT compilation before measurement
    try:
        warmup_df = (
            WeightedQuadratic(weights[:10]) if "sample_weight" in dat else datafit
        )
        warmup_model = GeneralizedLinearEstimator(
            datafit=warmup_df, penalty=penalty, solver=solver
        )
        warmup_model.fit(X[:10], y[:10])
    except Exception:
        pass

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
