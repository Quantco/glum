import warnings
from typing import Any, Dict

import numpy as np
from pyglmnet import GLM

from .util import runtime


def setup_and_fit(
    dat: Dict[str, np.ndarray], distribution: str, alpha: float, l1_ratio: float,
):
    model = GLM(distr=distribution, alpha=l1_ratio, reg_lambda=alpha, solver="cdfast")
    return model.fit(dat["X"], dat["y"])


def pyglmnet_bench(
    dat: Dict[str, np.ndarray], distribution: str, alpha: float, l1_ratio: float,
) -> Dict[str, Any]:
    results: Dict[str, Any] = dict()
    if "weights" in dat.keys():
        warnings.warn("pyglmnet does not support weights")
        return results
    if not isinstance(dat["X"], np.ndarray):
        warnings.warn("pyglmnet only supports numpy array inputs")
        return results

    results["runtime"], model = runtime(
        setup_and_fit, dat, distribution, alpha, l1_ratio
    )
    results["model_obj"] = model
    results["intercept"] = model.beta0_
    results["coef"] = model.beta_
    results["n_iter"] = model.n_iter_
    return results
