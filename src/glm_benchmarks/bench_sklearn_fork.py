import time
from typing import Dict, Union

import numpy as np
from scipy import sparse as sps

from .sklearn_fork import GeneralizedLinearRegressor
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

    fit_args = dict(X=dat["X"], y=dat["y"])
    if "weights" in dat.keys():
        fit_args.update({"sample_weight": dat["weights"]})

    model_args = dict(
        family="normal" if distribution == "gaussian" else distribution,
        alpha=alpha,
        l1_ratio=l1_ratio,
        max_iter=10000,
        random_state=random_seed,
        copy_X=True,
        selection="random",
        tol=1e-7,
    )

    result["runtime"], m = runtime(build_and_fit, model_args, fit_args)
    result["model_obj"] = m
    result["intercept"] = m.intercept_
    result["coef"] = m.coef_
    result["n_iter"] = m.n_iter_

    # import numpy as np
    # result["path"] = compute_path(m.n_iter_, model_args, fit_args)
    # np.testing.assert_almost_equal(
    #     result["path"][-1], result["coef"], -np.log10(model_args["tol"]) - 1
    # )
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
