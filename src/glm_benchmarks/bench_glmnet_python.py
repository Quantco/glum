import warnings
from typing import Any, Dict, Union

import numpy as np
from glmnet_python import glmnet
from scipy import sparse as sps

from .util import benchmark_convergence_tolerance, runtime


def glmnet_python_bench(
    dat: Dict[str, Union[sps.spmatrix, np.ndarray]],
    distribution: str,
    alpha: float,
    l1_ratio: float,
    iterations: int,
    print_diagnostics: bool = True,  # ineffective here
) -> Dict[str, Any]:
    result: Dict = dict()

    X = dat["X"]
    if isinstance(X, sps.spmatrix):
        if not isinstance(X, sps.csc.csc_matrix):
            warnings.warn("sparse matrix will be converted to csc format")
            X = X.tocsc()
    elif not isinstance(X, np.ndarray):
        warnings.warn(
            "glmnet_python requires data as scipy.sparse matrix or numpy array."
        )
        return result

    if len(dat["y"]) <= 650:
        warnings.warn("glmnet_python does not work with too few rows")
        return result
    if distribution == "gamma" or "tweedie" in distribution:
        warnings.warn("glmnet_python does not support gamma")
        return result
    if distribution == "gaussian" or (
        distribution == "poisson" and len(dat["y"]) == 1000 and dat["X"].shape[1] > 200
    ):
        warnings.warn("This problem causes a mysterious crash. Skipping.")
        return result

    glmnet_kws = dict(
        x=X,
        y=dat["y"].copy(),
        family=distribution,
        alpha=l1_ratio,
        lambdau=np.array([alpha]),
        standardize=False,
        thresh=benchmark_convergence_tolerance,
    )
    if "weights" in dat.keys():
        glmnet_kws.update({"weights": dat["weights"]})
    if "offset" in dat.keys():
        glmnet_kws.update({"offset": dat["offset"]})

    result["runtime"], m = runtime(glmnet, iterations, **glmnet_kws)
    result["intercept"] = m["a0"][0]
    result["coef"] = m["beta"][:, 0]
    result["n_iter"] = m["npasses"]
    return result
