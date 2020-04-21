import time
from typing import Dict, Tuple, Union

import numpy as np
from scipy import sparse as sps


def runtime(f, *args, **kwargs):
    start = time.time()
    out = f(*args, **kwargs)
    end = time.time()
    return end - start, out


def _get_poisson_ll(
    dat: Dict[str, Union[np.ndarray, sps.spmatrix]], intercept: float, coefs: np.ndarray
) -> float:
    """
    Only up to a constant!
    """
    ln_e_y = _get_linear_prediction_part(dat["X"], coefs, intercept)
    w_ln_e_y = dat["weights"] * ln_e_y if "weights" in dat.keys() else ln_e_y
    ll = w_ln_e_y.dot(dat["y"]) - np.exp(w_ln_e_y).sum()
    return ll


def _get_minus_ssr(
    dat: Dict[str, Union[np.ndarray, sps.spmatrix]], intercept: float, coefs: np.ndarray
) -> float:
    """
    The normal log-likelihood, up to a constant.
    """
    resids = dat["y"] - _get_linear_prediction_part(dat["X"], coefs, intercept)
    squared_resids = resids ** 2

    if "weights" in dat.keys():
        return -dat["weights"].dot(squared_resids)
    else:
        return -squared_resids.sum()


def _get_linear_prediction_part(
    x: Union[np.ndarray, sps.spmatrix], coefs: np.ndarray, intercept: float
) -> np.ndarray:
    return x.dot(coefs) + intercept


def _get_penalty(alpha: float, l1_ratio: float, coefs: np.ndarray) -> float:
    l1 = np.sum(np.abs(coefs))
    l2 = np.sum(coefs ** 2)
    penalty = alpha * (l1_ratio * l1 + (1 - l1_ratio) * l2)
    return penalty


def get_obj_val(
    dat: Dict[str, Union[np.ndarray, sps.spmatrix]],
    distribution: str,
    alpha: float,
    l1_ratio: float,
    intercept: float,
    coefs: np.ndarray,
    one_over_n_likelihood: bool = False,
) -> float:
    if distribution == "poisson":
        log_like = _get_poisson_ll(dat, intercept, coefs)
    elif distribution == "gaussian":
        log_like = _get_minus_ssr(dat, intercept, coefs)
    else:
        raise NotImplementedError

    penalty = _get_penalty(alpha, l1_ratio, coefs)
    if one_over_n_likelihood:
        log_like /= len(dat["y"])

    return -log_like + penalty


def exposure_correction(
    power: float,
    y: np.ndarray,
    exposure: np.ndarray = None,
    sample_weight: np.ndarray = None,
    offset: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Adjust outcomes and weights for exposure and offsets.

    This works for any Tweedie distributions with log-link. This is equivalence can be
    verified by checking the first order condition of the Tweedie log-likelihood.

    Parameters
    ----------
    power : float
        Power parameter of the Tweedie distribution.
    y : array-like
        Array with outcomes.
    exposure : array-like, optional, default=None
        Array with exposure.
    offset : array-like, optional, default=None
        Array with additive offsets.
    sample_weight : array-like, optional, default=None
        Array with sampling weights.

    Returns
    -------
    np.array
        Array with adjusted outcomes.
    np.array
        Estimation weights.
    """
    y = np.asanyarray(y)
    sample_weight = None if sample_weight is None else np.asanyarray(sample_weight)

    if offset is not None:
        offset = np.exp(np.asanyarray(offset))
        y = y / offset
        sample_weight = (
            offset ** (2 - power)
            if sample_weight is None
            else sample_weight * offset ** (2 - power)
        )
    if exposure is not None:
        exposure = np.asanyarray(exposure)
        y = y / exposure
        sample_weight = exposure if sample_weight is None else sample_weight * exposure

    return y, sample_weight
