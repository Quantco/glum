import warnings
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri as n2r
from rpy2.robjects.packages import importr
from scipy import sparse as sps

from .util import benchmark_convergence_tolerance, runtime

is_initialized = False


def _r_install_if_not_available(pkg_name):
    utils = importr("utils")
    if pkg_name not in utils.installed_packages():
        utils.install_packages(pkg_name, repos="https://cloud.r-project.org")


def _setup_r_glmnet():
    global is_initialized
    if is_initialized:
        return
    n2r.activate()

    ro.r.library("utils")
    for pkg in ["glmnet", "statmod", "tweedie"]:
        _r_install_if_not_available(pkg)
        ro.r.library(pkg)

    is_initialized = True


def _to_r_obj(X, R_name):
    nr = X.shape[0]
    nc = 1 if len(X.shape) == 1 else X.shape[1]
    if sps.issparse(X):
        r_Matrix = importr("Matrix")
        X_coo = X.tocoo()
        ro.r.assign(
            R_name,
            r_Matrix.sparseMatrix(
                i=ro.IntVector(X_coo.row + 1),
                j=ro.IntVector(X_coo.col + 1),
                x=ro.FloatVector(X_coo.data),
                dims=ro.IntVector(X_coo.shape),
            ),
        )
    else:  # if dense
        ro.r.assign(R_name, ro.r.matrix(X, nrow=nr, ncol=nc))


def r_glmnet_bench(
    dat: dict[str, Union[sps.spmatrix, np.ndarray]],
    distribution: str,
    alpha: float,
    l1_ratio: float,
    iterations: int,
    cv: bool,
    reg_multiplier: Optional[float] = None,
    **kwargs,
) -> dict[str, Any]:
    """
    Run glmnet benchmark in R then port the results back to Python.

    Parameters
    ----------
    dat
    distribution
    alpha
    l1_ratio
    iterations
    cv
    reg_multiplier
    kwargs

    Returns
    -------
    Dict storing info about this model run.
    """
    result: dict = {}

    _setup_r_glmnet()

    X = dat["X"]
    if isinstance(X, sps.spmatrix):
        if not isinstance(X, sps.csc.csc_matrix):
            warnings.warn("sparse matrix will be converted to csc format")
            X = X.tocsc()
    elif isinstance(X, pd.DataFrame):
        X = X.to_numpy()
    elif not isinstance(X, np.ndarray):
        warnings.warn(
            "glmnet requires data as scipy.sparse matrix, pandas dataframe, or "
            "numpy array. Skipping."
        )
        return result

    if distribution == "gamma":
        distribution = ro.r["Gamma"](link="log")
    elif distribution.startswith("tweedie"):
        p = float(distribution.split("tweedie-p=")[1])
        distribution = ro.r["tweedie"](link_power=0, var_power=p)

    r = ro.r

    # Do this before fitting so we're not including python to R conversion
    # times
    _to_r_obj(X, "X_in_R")
    _to_r_obj(dat["y"], "y_in_R")

    glmnet_kws = dict(
        x=r["X_in_R"],
        y=r["y_in_R"],
        family=distribution,
        alpha=l1_ratio,
        standardize=False,
        thresh=benchmark_convergence_tolerance,
    )
    if "sample_weight" in dat.keys():
        glmnet_kws.update({"weights": ro.FloatVector(dat["sample_weight"])})
    if "offset" in dat.keys():
        glmnet_kws.update({"offset": ro.FloatVector(dat["offset"])})

    # By default, glmnet runs for 100 different values of regularization strength.
    # For a fair comparison of runtime, we'd like to run for just a single value.
    # These parameters ensure that is the case.
    glmnet_kws["lambda"] = alpha
    glmnet_kws["nlambda"] = 1

    # NOTE: We checked thoroughly and this runtime measurement only includes
    # The cost of the glmnet function in R. We checked this by running the same
    # problem directly from R.
    result["runtime"], m = runtime(r["glmnet"], iterations, **glmnet_kws)
    result["intercept"] = m.rx2("a0")[0]
    result["coef"] = np.squeeze(np.asanyarray(r["as.matrix"](m.rx2("beta"))))
    result["n_iter"] = m.rx2("npasses")[0]
    return result
