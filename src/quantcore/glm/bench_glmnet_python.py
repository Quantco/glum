import warnings
from typing import Any, Dict, Optional, Union

import numpy as np
from glmnet_python import cvglmnet, glmnet
from scipy import sparse as sps

from .util import benchmark_convergence_tolerance, runtime


def glmnet_python_bench(
    dat: Dict[str, Union[sps.spmatrix, np.ndarray]],
    distribution: str,
    alpha: float,
    l1_ratio: float,
    iterations: int,
    cv: bool,
    print_diagnostics: bool = True,  # ineffective here
    reg_multiplier: Optional[float] = None,
    **kwargs,
) -> Dict[str, Any]:
    result: Dict = dict()

    X = dat["X"]
    if isinstance(X, sps.spmatrix):
        if not isinstance(X, sps.csc.csc_matrix):
            warnings.warn("sparse matrix will be converted to csc format")
            X = X.tocsc()
    elif not isinstance(X, np.ndarray):
        warnings.warn(
            "glmnet_python requires data as scipy.sparse matrix or numpy array. Skipping."
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
        y=dat["y"].astype(np.float64).copy(),
        family=distribution,
        alpha=l1_ratio,
        standardize=False,
        thresh=benchmark_convergence_tolerance,
    )
    if "weights" in dat.keys():
        glmnet_kws.update({"weights": dat["weights"][:, None]})
    if "offset" in dat.keys():
        glmnet_kws.update({"offset": dat["offset"][:, None]})

    if cv:
        result["runtime"], m = runtime(cvglmnet, iterations, **glmnet_kws)
        fit_model = m["glmnet_fit"]
        result["n_alphas"] = len(m["lambdau"])
        result["max_alpha"] = m["lambdau"].max()
        result["min_alpha"] = m["lambdau"].min()
        result["best_alpha"] = m["lambda_min"][0]
    else:
        glmnet_kws["lambdau"] = np.array([alpha])
        if reg_multiplier is not None:
            glmnet_kws["lambdau"] *= reg_multiplier
        result["runtime"], m = runtime(glmnet, iterations, **glmnet_kws)
        fit_model = m

    result["intercept"] = fit_model["a0"][0]
    result["coef"] = fit_model["beta"][:, 0]
    result["n_iter"] = fit_model["npasses"]

    return result
