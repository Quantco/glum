from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy import sparse as sps

from glum_benchmarks.util import runtime

_CAT_COLS = ["VehPower", "VehAge", "DrivAge", "VehBrand", "VehGas", "Region", "Area"]
_NUM_COLS = ["Density_log"]

_PYGAM_DISTRIBUTIONS = {
    "poisson": "poisson",
    "gamma": "gamma",
}


def _prepare_pygam_data(df):
    """Build numpy X: BonusMalus (col 0), then Density_log, then coded categoricals."""
    parts = [df["BonusMalus"].values.reshape(-1, 1)]
    for col in _NUM_COLS:
        parts.append(df[col].values.reshape(-1, 1))
    for col in _CAT_COLS:
        codes = pd.Categorical(df[col]).codes.astype(float)
        parts.append(codes.reshape(-1, 1))
    return np.hstack(parts)


def _build_pygam_terms(n_splines=10):
    from pygam import f, l, s

    terms = s(0, n_splines=n_splines, constraints="monotonic_inc")
    col = 1
    for _ in _NUM_COLS:
        terms = terms + l(col)
        col += 1
    for _ in _CAT_COLS:
        terms = terms + f(col)
        col += 1
    return terms


def _build_and_fit(X, y, terms, distribution):
    from pygam import GAM

    m = GAM(terms, distribution=distribution, link="log")
    m.fit(X, y)
    return m


def pygam_bench(
    dat: dict[str, Union[np.ndarray, sps.spmatrix]],
    distribution: str,
    alpha: float,
    l1_ratio: float,
    iterations: int,
    timeout: Optional[float] = None,
    **kwargs,
):
    if "df" not in dat:
        return {}

    pygam_dist = _PYGAM_DISTRIBUTIONS.get(distribution)
    if pygam_dist is None:
        return {}

    X = _prepare_pygam_data(dat["df"])
    y = np.asarray(dat["y"], dtype=float)
    terms = _build_pygam_terms()

    result = {}
    result["runtime"], m = runtime(
        _build_and_fit, iterations, X, y, terms, pygam_dist, timeout=timeout
    )
    result["intercept"] = 0.0
    result["coef"] = m.coef_
    result["n_iter"] = None
    result["max_iter"] = None
    return result
