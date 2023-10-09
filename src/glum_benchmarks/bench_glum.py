import time
from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy import sparse as sps

from glum import GeneralizedLinearRegressor, GeneralizedLinearRegressorCV

from .util import benchmark_convergence_tolerance, get_sklearn_family, runtime

random_seed = 110


def _build_and_fit(model_args, fit_args, cv: bool):
    if cv:
        return GeneralizedLinearRegressorCV(**model_args).fit(**fit_args)
    return GeneralizedLinearRegressor(**model_args).fit(**fit_args)


def glum_bench(
    dat: dict[str, Union[np.ndarray, sps.spmatrix]],
    distribution: str,
    alpha: float,
    l1_ratio: float,
    iterations: int,
    cv: bool,
    diagnostics_level: str = "basic",
    reg_multiplier: Optional[float] = None,
    hessian_approx: float = 0.0,
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
    cv
    diagnostics_level
    reg_multiplier
    hessian_approx
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
        max_iter=1000,
        random_state=random_seed,
        copy_X=False,
        selection="cyclic",
        gradient_tol=1 if cv else benchmark_convergence_tolerance,
        step_size_tol=0.01 * benchmark_convergence_tolerance,
        force_all_finite=False,
        hessian_approx=hessian_approx,
        verbose=False,
    )

    if not cv:
        model_args["alpha"] = (
            alpha if reg_multiplier is None else alpha * reg_multiplier
        )

    result["runtime"], m = runtime(_build_and_fit, iterations, model_args, fit_args, cv)
    if not cv:
        # Just check that predict works here... This doesn't take very long.
        m.predict(**{k: v for k, v in fit_args.items() if k != "y"})

    result["intercept"] = m.intercept_
    result["coef"] = m.coef_
    result["n_iter"] = m.n_iter_
    if cv:
        alphas: np.ndarray = m.alphas_
        result["n_alphas"] = len(alphas)
        result["max_alpha"] = alphas.max()
        result["min_alpha"] = alphas.min()
        result["best_alpha"] = m.alpha_

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
