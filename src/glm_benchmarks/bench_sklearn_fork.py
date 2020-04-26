import time
from typing import Dict, Union

import numpy as np
from scipy import sparse as sps

from .sklearn_fork import GeneralizedLinearRegressor, TweedieDistribution
from .util import runtime

random_seed = 110


def build_and_fit(model_args, fit_args):
    return GeneralizedLinearRegressor(**model_args).fit(**fit_args)


def sklearn_fork_bench(
    dat: Dict[str, Union[np.ndarray, sps.spmatrix]],
    distribution: str,
    alpha: float,
    l1_ratio: float,
):
    result = dict()

    X = dat["X"]
    fit_args = dict(X=X, y=dat["y"])
    if "weights" in dat.keys():
        fit_args.update({"sample_weight": dat["weights"]})

    family = distribution
    if family == "gaussian":
        family = "normal"
    elif "tweedie" in family:
        tweedie_p = float(family.split("_p=")[1])
        family = TweedieDistribution(tweedie_p)  # type: ignore

    model_args = dict(
        family=family,
        alpha=alpha,
        l1_ratio=l1_ratio,
        max_iter=100,
        random_state=random_seed,
        copy_X=False,
        selection="random",
        tol=1e-7,
    )

    result["runtime"], m = runtime(build_and_fit, model_args, fit_args)
    result["model_obj"] = m
    result["intercept"] = m.intercept_
    result["coef"] = m.coef_
    result["n_iter"] = m.n_iter_

    m.report_diagnostics()
    return result


def compute_path(niters, model_args, fit_args):
    step_model_args = model_args.copy()
    step_model_args["max_iter"] = 1
    step_model_args["warm_start"] = True
    m = GeneralizedLinearRegressor(**step_model_args)
    path = []
    for i in range(niters):
        start = time.time()
        m.fit(**fit_args)
        print(time.time() - start)
        path.append(m.coef_.copy())
    return path
