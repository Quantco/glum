import time
from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy import sparse as sps

from glum import GeneralizedLinearRegressor
from glum_benchmarks.util import (
    benchmark_convergence_tolerance,
    get_sklearn_family,
    runtime,
)

random_seed = 110


def _build_and_fit(model_args, fit_args):
    return GeneralizedLinearRegressor(**model_args).fit(**fit_args)


def glum_bench(
    dat: dict[str, Union[np.ndarray, sps.spmatrix]],
    distribution: str,
    alpha: float,
    l1_ratio: float,
    iterations: int,
    diagnostics_level: str = "basic",
    hessian_approx: float = 0.0,
    standardize: bool = True,
    timeout: Optional[float] = None,
    **kwargs,
):
    """
    Run the glum.GeneralizedLinearRegressor benchmark.

    Parameters
    ----------
    dat
    distribution
    alpha
    l1_ratio
    iterations
    diagnostics_level
    hessian_approx
    standardize
    kwargs

    Returns
    -------
    dict

    """
    result = {}

    X = dat["X"]
    fit_args = dict(X=X, y=dat["y"])
    if "sample_weight" in dat.keys():
        fit_args.update({"sample_weight": dat["sample_weight"]})
    if "offset" in dat.keys():
        fit_args.update({"offset": dat["offset"]})

    model_args = dict(
        family=get_sklearn_family(distribution),
        l1_ratio=l1_ratio,
        random_state=random_seed,
        copy_X=False,
        selection="cyclic",
        gradient_tol=benchmark_convergence_tolerance,
        step_size_tol=0.01 * benchmark_convergence_tolerance,
        force_all_finite=False,
        hessian_approx=hessian_approx,
        scale_predictors=standardize,
        verbose=False,
    )

    model_args["alpha"] = alpha

    result["runtime"], m = runtime(
        _build_and_fit, iterations, model_args, fit_args, timeout=timeout
    )

    # Just check that predict works here... This doesn't take very long.
    m.predict(**{k: v for k, v in fit_args.items() if k != "y"})

    result["intercept"] = m.intercept_
    result["coef"] = m.coef_
    result["n_iter"] = m.n_iter_
    result["max_iter"] = m.max_iter  # For convergence detection

    with pd.option_context(
        "display.expand_frame_repr",
        False,
        "display.max_columns",
        None,
        "display.max_rows",
        None,
    ):
        if diagnostics_level == "basic":
            m.report_diagnostics()
        elif diagnostics_level == "full":
            m.report_diagnostics(full_report=True)
    return result


def _compute_path(niters, model_args, fit_args):
    step_model_args = model_args.copy()
    step_model_args["max_iter"] = 1
    step_model_args["warm_start"] = True
    m = GeneralizedLinearRegressor(**step_model_args)
    path = []
    for _ in range(niters):
        start = time.time()
        m.fit(**fit_args)
        print(time.time() - start)
        path.append(m.coef_.copy())
    return path
