import os
import signal
import statistics
import time
from typing import Optional, Union

import numpy as np
import pandas as pd
import tabmat as tm
from scipy import sparse as sps

from glum import GeneralizedLinearRegressor, TweedieDistribution
from glum._solvers import eta_mu_objective

benchmark_convergence_tolerance = 1e-4
cache_location = os.environ.get("GLM_BENCHMARKS_CACHE", None)


def runtime(f, iterations, *args, timeout=None, **kwargs):
    """
    Measure how long it takes to run function f.

    When iterations >= 2, the first iteration is treated as warmup and
    discarded. The median runtime of the remaining iterations is reported.
    This avoids JIT/cache warmup effects and is more robust to outliers
    than the minimum.

    When iterations == 1 (e.g. in tests), the single run is returned
    directly with no warmup discard.

    Parameters
    ----------
    f: function
    iterations: int
        Total number of times to run the function. Use >= 2 for
        benchmarking (1 warmup + measured runs). Use 1 for tests.
    args: Passed to f
    timeout: float, optional
        Timeout in seconds for each iteration.
    kwargs: Passed to f

    Returns
    -------
    Tuple: (Median runtime after warmup, output from the run closest to
        the median runtime)
        If all iterations timeout, raises TimeoutError.

    """

    successful_runs = []  # (runtime, output, iteration_index) tuples

    for i in range(iterations):
        # Set up timeout for this iteration if requested
        if timeout is not None:

            def _iter_timeout_handler(signum, frame):
                raise TimeoutError(f"Iteration {i + 1} exceeded {timeout}s timeout")

            old_handler = signal.signal(signal.SIGALRM, _iter_timeout_handler)
            signal.alarm(int(timeout))

        try:
            start = time.time()
            out = f(*args, **kwargs)
            end = time.time()
            runtime_val = end - start
            successful_runs.append((runtime_val, out, i))
        except TimeoutError:
            # This iteration timed out, continue to next iteration
            pass
        finally:
            # Cancel alarm and restore handler
            if timeout is not None:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

    if not successful_runs:
        # All iterations timed out
        raise TimeoutError(f"All {iterations} iterations exceeded {timeout}s timeout")

    # Discard the first successful iteration (warmup)
    if len(successful_runs) > 1:
        measured_runs = successful_runs[1:]
    else:
        # If only the warmup succeeded use it as a fallback
        measured_runs = successful_runs

    # Return the run closest to the median runtime
    runtimes = [r[0] for r in measured_runs]
    median_runtime = statistics.median(runtimes)
    closest_run = min(measured_runs, key=lambda x: abs(x[0] - median_runtime))
    return closest_run[0], closest_run[1]


def get_sklearn_family(distribution):
    """
    Translate statistical family to its equivalent in sklearn jargon.

    Parameters
    ----------
    distribution
    """
    family = distribution
    if family == "gaussian":
        family = "normal"
    elif "tweedie" in family:
        tweedie_p = float(family.split("-p=")[1])
        family = TweedieDistribution(tweedie_p)  # type: ignore
    return family


def get_obj_val(
    dat: dict[str, Union[np.ndarray, sps.spmatrix, tm.MatrixBase, pd.DataFrame]],
    distribution: str,
    alpha: float,
    l1_ratio: float,
    intercept: float,
    coefs: np.ndarray,
) -> float:
    """
    Return objective value.

    Parameters
    ----------
    dat
    distribution
    alpha
    l1_ratio
    intercept
    coefs

    Returns
    -------
    float
        Objective value

    """
    model = GeneralizedLinearRegressor(
        alpha=alpha,
        l1_ratio=l1_ratio,
        family=get_sklearn_family(distribution),
    )
    model._set_up_for_fit(dat["y"])

    full_coefs: np.ndarray = np.concatenate([[intercept], coefs])
    offset = dat.get("offset")
    if isinstance(dat["X"], tm.MatrixBase):
        X_dot_coef = dat["X"].matvec(coefs)
    elif isinstance(dat["X"], pd.DataFrame):
        X_dot_coef = dat["X"].to_numpy(dtype=float).dot(coefs)
    else:
        X_dot_coef = dat["X"].dot(coefs)
    X_dot_coef += intercept
    if isinstance(X_dot_coef, pd.Series):
        X_dot_coef = X_dot_coef.values
    if offset is not None:
        X_dot_coef += offset

    zeros = np.zeros(dat["X"].shape[0])
    y = dat["y"].astype(coefs.dtype)  # type: ignore
    weights = dat.get("sample_weight", dat.get("weights", np.ones_like(y)))
    weights = weights.astype(coefs.dtype)
    weights /= weights.sum()
    P1 = l1_ratio * alpha * np.ones_like(coefs)
    P2 = (1 - l1_ratio) * alpha * np.ones_like(coefs)

    _, _, obj_val, _ = eta_mu_objective(
        model._family_instance,
        model._link_instance,
        X_dot_coef,
        1.0,
        full_coefs,
        zeros,
        y,
        weights,
        P1,
        P2,
        1,
    )
    return obj_val


def exposure_and_offset_to_weights(
    power: float,
    y: np.ndarray,
    exposure: np.ndarray = None,
    sample_weight: np.ndarray = None,
    offset: np.ndarray = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Adjust outcomes and weights for exposure and offsets.

    This works for any Tweedie distributions with log-link. This is equivalence can be
    verified by checking the first order condition of the Tweedie log-likelihood.

    Parameters
    ----------
    power : float
        Power parameter of the Tweedie distribution.
    y : array-like
        Array with outcomes.
    exposure : array-like, optional, default=None
        Array with exposure.
    offset : array-like, optional, default=None
        Array with additive offsets.
    sample_weight : array-like, optional, default=None
        Array with sampling weights.

    Returns
    -------
    np.array
        Array with adjusted outcomes.
    np.array
        Estimation weights.
    """
    y = np.asanyarray(y)
    sample_weight = None if sample_weight is None else np.asanyarray(sample_weight)

    if offset is not None:
        exposure = np.exp(np.asanyarray(offset))
    elif exposure is not None:
        exposure = np.asarray(exposure)
    else:
        raise ValueError("Need offset or exposure.")
    y = y / exposure
    sample_weight = (
        exposure ** (2 - power)  # type: ignore
        if sample_weight is None
        else sample_weight * exposure ** (2 - power)  # type: ignore
    )
    return y, sample_weight  # type: ignore


class BenchmarkParams:
    """
    Store metadata about the problem we want to solve and how.

    Attributes reflect exactly what the user passed in, only modified by type
    conversions. Any additional processing should be downstream.
    """

    def __init__(
        self,
        problem_name: Optional[str] = None,
        library_name: Optional[str] = None,
        num_rows: Optional[int] = None,
        k_over_n_ratio: Optional[float] = None,
        storage: Optional[str] = None,
        threads: Optional[int] = None,
        alpha: Optional[float] = None,
        hessian_approx: Optional[float] = None,
        diagnostics_level: Optional[str] = None,
    ):
        self.problem_name = problem_name
        self.library_name = library_name
        self.num_rows = num_rows
        self.k_over_n_ratio = k_over_n_ratio
        self.storage = storage
        self.threads = threads
        self.alpha = alpha
        self.hessian_approx = hessian_approx
        self.diagnostics_level = diagnostics_level

    param_names = [
        "problem_name",
        "library_name",
        "num_rows",
        "k_over_n_ratio",
        "storage",
        "threads",
        "alpha",
        "hessian_approx",
        "diagnostics_level",
    ]

    def update_params(self, **kwargs):
        """
        Update attributes to those found in kwargs.

        Parameters
        ----------
        kwargs: dict

        Returns
        -------
        self with updated attributes

        """
        for k, v in kwargs.items():
            assert k in self.param_names
            setattr(self, k, v)
        return self

    def get_result_fname(self) -> str:
        """
        Get file name to which results will be written.

        Returns
        -------
        str
            File name within benchmark result directory.

        """
        return "_".join(str(getattr(self, k)) for k in self.param_names)


defaults = dict(
    threads=os.environ.get("OMP_NUM_THREADS", os.cpu_count()),
    problem_name=None,
    library_name=None,
    num_rows=None,
    k_over_n_ratio=1.0,
    alpha=None,
    storage="dense",
    hessian_approx=0.0,
    diagnostics_level="basic",
)


def get_params_from_fname(fname: str) -> BenchmarkParams:
    """
    Map file name to a BenchmarkParams instance.

    File names are formatted as:
    problem_library_numrows_storage_threads_alpha_hessian_diag.pkl

    Parameters
    ----------
    fname: file name

    Returns
    -------
    BenchmarkParams
    """
    parts = fname.replace(".pkl", "").split("_")

    # Parse each part, converting "None" strings to actual None
    def parse_value(value: str, dtype=str):
        if value == "None":
            return None
        if dtype is int:
            return int(value)
        if dtype is float:
            return float(value)
        return value

    # Map parts to parameter names with appropriate types
    # Order matches BenchmarkParams.param_names
    param_types = [str, str, int, float, str, int, float, float, str]

    kwargs = {}
    for i, (name, dtype) in enumerate(zip(BenchmarkParams.param_names, param_types)):
        if i < len(parts):
            kwargs[name] = parse_value(parts[i], dtype)

    return BenchmarkParams(**kwargs)


def get_tweedie_p(distribution: str) -> float:
    """
    Extract the "p" parameter of the Tweedie distribution from the string name of the \
    distribution.

    Examples
    --------
    >>> get_tweedie_p("tweedie_p=1.5")
    1.5

    """
    tweedie = "tweedie" in distribution
    if tweedie:
        return float(distribution.split("=")[-1])
    if "poisson" == distribution:
        return 1
    if "gamma" == distribution:
        return 2
    if "gaussian" in distribution:
        return 0
    else:
        raise ValueError("Not a Tweedie distribution.")


def get_all_libraries() -> dict:
    """
    Get the names of all available libraries and the functions to benchmark them.

    Libraries with missing dependencies are excluded from the result.

    Returns
    -------
    dict
        Mapping of library name to benchmark function.
    """
    from glum_benchmarks.libraries import (
        celer_bench,
        glmnet_bench,
        glum_bench,
        h2o_bench,
        skglm_bench,
        sklearn_bench,
        zeros_bench,
    )

    all_libraries = {
        "glum": glum_bench,
        "zeros": zeros_bench,
        "celer": celer_bench,
        "h2o": h2o_bench,
        "glmnet": glmnet_bench,
        "skglm": skglm_bench,
        "sklearn": sklearn_bench,
    }

    # Filter out libraries that aren't available (None due to missing deps)
    return {k: v for k, v in all_libraries.items() if v is not None}


def execute_problem_library(
    params: BenchmarkParams,
    iterations: int = 1,
    diagnostics_level: Optional[str] = "basic",
    standardize: bool = True,
    timeout: Optional[float] = None,
    **kwargs,
):
    """
    Run the benchmark problem specified by 'params', 'iterations' times.

    By default, continuous features are pre-standardized in the data loader
    before OHE and format conversion. Pass ``standardize=False`` to skip
    (e.g. for golden master tests).

    Parameters
    ----------
    params
    iterations
    diagnostics_level
    standardize
        Whether to pre-standardize continuous features in the data loader.
    kwargs

    Returns
    -------
    Tuple: Result data on this run, and the alpha applied
    """
    from glum_benchmarks.problems import get_all_problems

    assert params.problem_name is not None
    assert params.library_name is not None
    P = get_all_problems()[params.problem_name]
    L = get_all_libraries()[params.library_name]

    for k in params.param_names:
        if getattr(params, k) is None:
            params.update_params(**{k: defaults[k]})

    dat = P.data_loader(
        num_rows=params.num_rows,
        k_over_n_ratio=params.k_over_n_ratio,
        storage=params.storage,
        standardize=standardize,
    )

    os.environ["OMP_NUM_THREADS"] = str(params.threads)

    if params.alpha is None:
        params.alpha = P.alpha

    alpha = (
        params.alpha / np.asarray(dat["sample_weight"]).mean()
        if "sample_weight" in dat
        else params.alpha
    )

    result = L(
        dat,
        distribution=P.distribution,
        alpha=alpha,
        l1_ratio=P.l1_ratio,
        iterations=iterations,
        diagnostics_level=diagnostics_level,
        hessian_approx=params.hessian_approx,
        timeout=timeout,
        **kwargs,
    )

    if len(result) > 0:
        # Use best_alpha from CV if available, otherwise use base alpha
        alpha_for_obj = result.get("best_alpha", P.alpha)
        alpha_for_obj = (
            alpha_for_obj / np.asarray(dat["sample_weight"]).mean()
            if "sample_weight" in dat
            else alpha_for_obj
        )
        obj_val = get_obj_val(
            dat,
            P.distribution,
            alpha_for_obj,
            P.l1_ratio,
            result["intercept"],
            result["coef"],
        )

        result["obj_val"] = obj_val
        result["num_rows"] = dat["y"].shape[0]

    return result, params.alpha
