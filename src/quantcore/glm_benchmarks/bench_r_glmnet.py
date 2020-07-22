import warnings
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri as n2r
from rpy2.robjects.packages import importr
from scipy import sparse as sps

from .util import benchmark_convergence_tolerance, runtime

is_initialized = False


def r_install_if_not_available(pkg_name):
    utils = importr("utils")
    if pkg_name not in utils.installed_packages():
        utils.install_packages(pkg_name, repos="https://cloud.r-project.org")


def setup_r_glmnet():
    global is_initialized
    if is_initialized:
        return
    n2r.activate()

    ro.r.library("utils")
    for pkg in ["glmnet", "statmod", "tweedie"]:
        r_install_if_not_available(pkg)
        ro.r.library(pkg)

    is_initialized = True


def r_glmnet_bench(
    dat: Dict[str, Union[sps.spmatrix, np.ndarray]],
    distribution: str,
    alpha: float,
    l1_ratio: float,
    iterations: int,
    cv: bool,
    reg_multiplier: Optional[float] = None,
    **kwargs,
) -> Dict[str, Any]:
    result: Dict = dict()

    setup_r_glmnet()

    X = dat["X"]
    if isinstance(X, sps.spmatrix):
        if not isinstance(X, sps.csc.csc_matrix):
            warnings.warn("sparse matrix will be converted to csc format")
            X = X.tocsc()
    elif isinstance(X, pd.DataFrame):
        X = X.to_numpy()
    elif not isinstance(X, np.ndarray):
        warnings.warn(
            "glmnet_python requires data as scipy.sparse matrix, pandas dataframe, or numpy array. Skipping."
        )
        return result

    if distribution == "gamma":
        distribution = ro.r["Gamma"](link="log")
    elif distribution.startswith("tweedie"):
        p = float(distribution.split("tweedie-p=")[1])
        distribution = ro.r["tweedie"](link_power=0, var_power=p)
    elif distribution == "binomial":
        warnings.warn("r-glmnet fails for binomial")
        return result

    r = ro.r
    glmnet_kws = dict(
        x=X,
        y=dat["y"],
        family=distribution,
        alpha=l1_ratio,
        standardize=False,
        thresh=benchmark_convergence_tolerance,
    )
    if "weights" in dat.keys():
        glmnet_kws.update({"weights": ro.FloatVector(dat["weights"])})
    if "offset" in dat.keys():
        glmnet_kws.update({"offset": ro.FloatVector(dat["offset"])})

    glmnet_kws["lambda"] = alpha
    # TODO: make sure that the runtime of converting array types to R is not included in here.
    result["runtime"], m = runtime(r["glmnet"], iterations, **glmnet_kws)
    result["intercept"] = m.rx2("a0")[0]
    result["coef"] = np.squeeze(np.asanyarray(r["as.matrix"](m.rx2("beta"))))
    result["n_iter"] = m.rx2("npasses")[0]
    return result
