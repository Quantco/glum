import warnings
from typing import Any, Optional, Union

import numpy as np
import statsmodels.api as sm
from scipy import sparse as sps

from .util import benchmark_convergence_tolerance, runtime

# TODO: Add scaling/centering to prevent numerical overflow


def _fit_statsmodels(model, alpha, l1_ratio, fit_args):
    """
    Use fit_regularized if alpha is provided, else standard fit.

    alpha can be a scalar or vector. When vector, position 0 (intercept) should be 0.
    """
    if alpha is not None:
        return model.fit_regularized(alpha=alpha, L1_wt=l1_ratio, **fit_args)
    return model.fit(**fit_args)


def statsmodels_bench(
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
    Benchmark statsmodels GLM implementation.

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
    dict

    """
    if cv:
        warnings.warn(
            "statsmodels benchmarks do not support cross-validation, skipping."
        )
        return {}

    result: dict[str, Any] = {}
    X = dat["X"]
    reg_strength = alpha if reg_multiplier is None else alpha * reg_multiplier

    # Skip regularized problems for distributions where statsmodels.fit_regularized()
    # produces poor results despite our fixes (alpha vector + target scaling)
    if reg_strength > 0:
        if distribution == "gaussian":
            warnings.warn(
                "statsmodels.fit_regularized() produces poor results for Gaussian "
                "distributions (objective values thousands of times worse). Skipping."
            )
            return {}
        if distribution == "gamma":
            warnings.warn(
                "statsmodels.fit_regularized() is unreliable for Gamma distributions "
                "(numerical issues, convergence failures). Skipping."
            )
            return {}

    # Map family distributions with explicit canonical links for stability
    fam_map = {
        "gaussian": sm.families.Gaussian(link=sm.families.links.Identity()),
        "poisson": sm.families.Poisson(link=sm.families.links.Log()),
        "gamma": sm.families.Gamma(link=sm.families.links.Log()),
        "binomial": sm.families.Binomial(link=sm.families.links.Logit()),
    }

    if "tweedie" in distribution:
        p = float(distribution.split("-p=")[1])
        family = sm.families.Tweedie(var_power=p, link=sm.families.links.Log())
    else:
        family = fam_map.get(distribution)

    if family is None:
        warnings.warn(
            f"Distribution {distribution} not supported by statsmodels, skipping."
        )
        return {}

    # Convert to numpy/sparse and add intercept manually
    if sps.issparse(X):
        # Handle sparse matrices by adding intercept column manually
        n_samples = X.shape[0]
        intercept_col = sps.csr_matrix(np.ones((n_samples, 1)))
        X = sps.hstack([intercept_col, X], format="csr")
    else:
        if hasattr(X, "values"):
            X = X.values
        elif not isinstance(X, np.ndarray):
            X = np.asarray(X)

        # Ensure proper dtype and add intercept
        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]
        intercept_col = np.ones((n_samples, 1))
        X = np.hstack([intercept_col, X])

    # Scale target for numerical stability (but not for binomial)
    y = dat["y"]
    y_mean = np.mean(y)

    # Only scale for continuous distributions, not binomial
    if distribution == "binomial":
        y_scaled = y
        scale_factor = 1.0
    else:
        y_scaled = y / y_mean if y_mean > 0 else y
        scale_factor = float(y_mean)

    model = sm.GLM(
        y_scaled,  # Use scaled y for better numerical stability
        X,
        family=family,
        offset=dat.get("offset"),
        freq_weights=dat.get("sample_weight"),
    )

    # Initialize start_params to help with convergence
    # Initialize start_params at 0
    start_params = np.zeros(X.shape[1])

    # Set the intercept (index 0) to a reasonable guess based on scaled y
    if distribution in ["poisson", "gamma"] or "tweedie" in distribution:
        # For Log link with scaled y, intercept near 0 (y_scaled mean ~1)
        start_params[0] = 0.0
    elif distribution == "binomial":
        # For Logit link, intercept approx logit(mean(y))
        p_mean = np.clip(y_mean, 1e-6, 1 - 1e-6)
        start_params[0] = np.log(p_mean / (1 - p_mean))
    else:
        # For Gaussian / Identity link with scaled y, intercept should be near 1
        start_params[0] = 1.0

    # fit_regularized has different arg names than fit
    if reg_strength > 0:
        # Create alpha vector: 0 for intercept (position 0), reg_strength for others
        # This prevents statsmodels from shrinking the intercept
        alpha_vector = np.full(X.shape[1], reg_strength)
        alpha_vector[0] = 0.0
        fit_args = {"maxiter": 10000, "start_params": start_params}
        alpha_param = alpha_vector
    else:
        fit_args = {
            "maxiter": 10000,
            "tol": benchmark_convergence_tolerance,
            "start_params": start_params,
        }
        alpha_param = None

    try:
        result["runtime"], m = runtime(
            _fit_statsmodels, iterations, model, alpha_param, l1_ratio, fit_args
        )
    except Exception as e:
        warnings.warn(f"statsmodels failed with error: {e}")
        return {}

    # Adjust coefficients back to original scale
    if distribution in ["poisson", "gamma"] or "tweedie" in distribution:
        # For Log link: log(y/c) = X*beta', so log(y) = X*beta' + log(c)
        # Therefore: intercept_true = intercept_scaled + log(scale_factor)
        result["intercept"] = m.params[0] + np.log(scale_factor)
        result["coef"] = m.params[1:]  # Other coefficients unchanged
    elif distribution == "binomial":
        # Binomial y is not scaled, so no adjustment needed
        result["intercept"] = m.params[0]
        result["coef"] = m.params[1:]
    else:
        # For Gaussian / Identity link: y/c = X*beta', so y = X*(c*beta')
        # Therefore: all coefficients scale by scale_factor
        result["intercept"] = m.params[0] * scale_factor
        result["coef"] = m.params[1:] * scale_factor

    # Check for non-finite values (NaN or Inf) which can occur with numerical issues
    if not np.isfinite(result["intercept"]) or not np.all(np.isfinite(result["coef"])):
        warnings.warn(
            "statsmodels returned non-finite values (NaN/Inf), skipping this problem."
        )
        return {}

    # fit_regularized doesn't expose iteration count, use None for regularized
    if reg_strength > 0:
        result["n_iter"] = None
    else:
        result["n_iter"] = getattr(m, "iteration", None)

    return result
