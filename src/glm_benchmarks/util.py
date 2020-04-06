import time
from typing import Dict, Union

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
        return dat["weights"].dot(squared_resids)
    else:
        return squared_resids.sum()


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
) -> float:
    if distribution == "poisson":
        log_like = _get_poisson_ll(dat, intercept, coefs)
    elif distribution == "gaussian":
        log_like = _get_minus_ssr(dat, intercept, coefs)
    else:
        raise NotImplementedError

    penalty = _get_penalty(alpha, l1_ratio, coefs)

    return -log_like + penalty
