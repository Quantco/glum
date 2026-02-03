import warnings
from typing import Any, Optional, Union, cast

import numpy as np
from scipy import sparse as sps
from skglm import GeneralizedLinearEstimator
from skglm.datafits import Gamma, Logistic, Poisson, Quadratic
from skglm.penalties import L1, L1_plus_L2
from skglm.solvers import AndersonCD, ProxNewton

from glum_benchmarks.util import (
    _standardize_features,
    _unstandardize_coefficients,
    benchmark_convergence_tolerance,
    runtime,
)


def _build_and_fit(model_args, fit_args):
    return GeneralizedLinearEstimator(**model_args).fit(**fit_args)


def skglm_bench(
    dat: dict[str, Union[np.ndarray, sps.spmatrix]],
    distribution: str,
    alpha: float,
    l1_ratio: float,
    iterations: int,
    reg_multiplier: Optional[float] = None,
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
    reg_strength = alpha if reg_multiplier is None else alpha * reg_multiplier

    if "tweedie" in distribution:
        warnings.warn("skglm doesn't support Tweedie, skipping.")
        return result

    DATAFITS = {
        "gaussian": Quadratic(),
        "poisson": Poisson(),
        "binomial": Logistic(),
        "gamma": Gamma(),
    }

    datafit = DATAFITS[distribution]

    if l1_ratio == 1:
        penalty = L1(alpha=reg_strength)
    else:
        # We use L1_plus_L2(l1_ratio=0) for pure L2 to ensure prox_1d is available
        penalty = L1_plus_L2(alpha=reg_strength, l1_ratio=l1_ratio)

    # ProxNewton is faster for non-Gaussian distributions (Poisson, Gamma, Binomial)
    # and required for L2 problems. AndersonCD is better for Gaussian.
    if distribution in ["poisson", "gamma", "binomial"] or l1_ratio == 0:
        solver = ProxNewton(tol=benchmark_convergence_tolerance, fit_intercept=True)
    else:
        solver = AndersonCD(tol=benchmark_convergence_tolerance, fit_intercept=True)

    model_args = {
        "datafit": datafit,
        "penalty": penalty,
        "solver": solver,
    }

    # Data Conversion optimized for Coordinate Descent
    X_raw = dat["X"]
    if sps.issparse(X_raw):
        X = cast(sps.spmatrix, X_raw).tocsc()
    else:
        X = np.asfortranarray(X_raw, dtype=np.float64)

    y = np.asarray(dat["y"], dtype=np.float64).ravel()

    # skglm's Logistic datafit expects labels in {-1, 1}, not {0, 1}
    if distribution == "binomial":
        y = 2 * y - 1

    fit_args = {"X": X, "y": y}

    try:
        result["runtime"], m = runtime(
            _build_and_fit, iterations, model_args, fit_args, timeout=timeout
        )
    except TimeoutError:
        # Re-raise TimeoutError to allow proper timeout handling at higher level
        raise
    except Exception as e:
        warnings.warn(f"skglm failed: {e}")
        return {}

    intercept = np.array(m.intercept_).ravel()[0]
    coef = np.array(m.coef_).ravel()

    # Unstandardize coefficients to match original data scale
    result["intercept"], result["coef"] = _unstandardize_coefficients(
        intercept, coef, scaler, scaled_indices
    )
    result["n_iter"] = getattr(m, "n_iter_", None)
    result["max_iter"] = solver.max_iter  # For convergence detection

    return result
