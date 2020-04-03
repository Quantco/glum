import warnings
from typing import Any, Dict

import numpy as np

from .glmnet_qc.glmnet_qc import fit_glmnet
from .util import runtime


def glmnet_qc_bench(
    dat: Dict[str, np.ndarray], distribution: str, alpha: float, l1_ratio: float,
) -> Dict[str, Any]:
    result: Dict[str, Any] = dict()
    if distribution != "gaussian":
        warnings.warn("only gaussian is supported")
        return result
    if not isinstance(dat["X"], np.ndarray):
        warnings.warn("only dense arrays are supported")
        return result
    if "weights" in dat.keys():
        warnings.warn("weights are not supported")
        return result

    x = np.hstack((np.ones((len(dat["y"]), 1)), dat["X"]))

    result["runtime"], coef = runtime(fit_glmnet, dat["y"], x, alpha, l1_ratio)
    result["model_obj"] = None
    result["intercept"] = coef[0]
    result["coef"] = coef[1:]
    result["n_iter"] = 10
    return result
