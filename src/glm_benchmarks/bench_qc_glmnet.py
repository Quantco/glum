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
    if "weights" in dat.keys():
        warnings.warn("weights are not supported")
        return result

    n_iters = 10
    try:
        result["runtime"], model = runtime(
            fit_glmnet, dat["y"], dat["X"], alpha, l1_ratio, n_iters=n_iters,
        )
    except RuntimeError:
        warnings.warn("glmnet_qc did not converge.")
        return result
    result["model_obj"] = model
    result["intercept"] = model.intercept
    assert np.isfinite(model.params).all()
    result["coef"] = model.params
    result["n_iter"] = n_iters
    return result
