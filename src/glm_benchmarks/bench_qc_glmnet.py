import warnings
from typing import Any, Dict

import numpy as np
from scipy import sparse as sps

from .glmnet_qc.glmnet_qc import fit_glmnet
from .util import runtime


def glmnet_qc_bench(
    dat: Dict[str, np.ndarray],
    distribution: str,
    alpha: float,
    l1_ratio: float,
    cv: bool,
) -> Dict[str, Any]:
    result: Dict[str, Any] = dict()
    if distribution not in ["gaussian", "poisson", "binomial"]:
        warnings.warn(f"Distribution {distribution} is not supported.")
        return result
    if cv:
        warnings.warn("Cross-validation is not supported.")
        return result

    n_iters = 10
    if sps.issparse(dat["X"]):
        x = sps.hstack((np.ones((dat["X"].shape[0], 1)), dat["X"])).tocsc()
    else:
        x = np.hstack((np.ones((dat["X"].shape[0], 1)), dat["X"]))

    result["runtime"], model = runtime(
        fit_glmnet,
        dat["y"],
        x,
        alpha,
        l1_ratio,
        n_iters=n_iters,
        distribution=distribution,
        weights=dat["weights"] if "weights" in dat.keys() else None,
    )
    result["intercept"] = model.params[0]
    assert np.isfinite(model.params).all()
    result["coef"] = model.params[1:]
    result["n_iter"] = n_iters
    return result
