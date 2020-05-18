from typing import Dict, Union

import numpy as np
from scipy import sparse as sps

from .orig_sklearn_fork import GeneralizedLinearRegressor, TweedieDistribution
from .util import benchmark_convergence_tolerance, runtime

random_seed = 110


def build_and_fit(model_args, fit_args):
    return GeneralizedLinearRegressor(**model_args).fit(**fit_args)


def orig_sklearn_fork_bench(
    dat: Dict[str, Union[np.ndarray, sps.spmatrix]],
    distribution: str,
    alpha: float,
    l1_ratio: float,
    iterations: int,
    cv: bool,
    print_diagnostics: bool = True,
    **kwargs,
):
    if cv:
        raise ValueError("original sklearn fork does not support cross-validation")
    result = dict()  # type: ignore

    X = dat["X"]
    fit_args = dict(X=X, y=dat["y"])
    if "weights" in dat.keys():
        fit_args.update({"sample_weight": dat["weights"]})
    if "offset" in dat.keys():
        print("Original sklearn_fork does not support offsets.")
        return result

    family = distribution
    if family == "gaussian":
        family = "normal"
    elif "tweedie" in family:
        tweedie_p = float(family.split("-p=")[1])
        family = TweedieDistribution(tweedie_p)  # type: ignore

    model_args = dict(
        family=family,
        alpha=alpha,
        l1_ratio=l1_ratio,
        max_iter=150,
        random_state=random_seed,
        copy_X=False,
        selection="cyclic",
        tol=benchmark_convergence_tolerance,
    )
    model_args.update(kwargs)

    try:
        result["runtime"], m = runtime(build_and_fit, iterations, model_args, fit_args)
    except ValueError as e:
        print(f"Problem failed with this error: {e}")
        return result

    result["intercept"] = m.intercept_
    result["coef"] = m.coef_
    result["n_iter"] = m.n_iter_

    return result
