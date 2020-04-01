import warnings
from typing import Any, Dict, Union

import numpy as np
from glmnet_python import glmnet
from scipy import sparse as sps

from .util import runtime


def glmnet_python_bench(
    dat: Dict[str, Union[sps.spmatrix, np.ndarray]],
    distribution: str,
    alpha: float,
    l1_ratio: float,
) -> Dict[str, Any]:
    result: Dict = dict()
    if isinstance(dat["X"], sps.spmatrix):
        warnings.warn("glmnet_python does not support sparse matrices")
        return result

    if len(dat["y"]) <= 650:
        warnings.warn("glmnet_python does not work with too few rows")
        return result

    glmnet_kws = dict(
        x=dat["X"].copy(),
        y=dat["y"].copy(),
        family=distribution,
        alpha=l1_ratio,
        lambdau=np.array([alpha]),
        standardize=False,
        thresh=1e-7,
    )
    if "weights" in dat.keys():
        glmnet_kws.update({"weights": dat["weights"]})

    result["runtime"], m = runtime(glmnet, **glmnet_kws)

    result["model_obj"] = m
    result["intercept"] = m["a0"]
    result["coef"] = m["beta"][:, 0]
    result["n_iter"] = m["npasses"]
    return result
