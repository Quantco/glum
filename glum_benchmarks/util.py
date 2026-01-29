import os
import time
from typing import Optional, Union

import numpy as np
import pandas as pd
import tabmat as tm
from scipy import sparse as sps
from scipy.sparse import csc_matrix

from glum import GeneralizedLinearRegressor, TweedieDistribution
from glum._solvers import eta_mu_objective

benchmark_convergence_tolerance = 1e-4
cache_location = os.environ.get("GLM_BENCHMARKS_CACHE", None)


def runtime(f, iterations, *args, **kwargs):
    """
    Measure how long it tales to run function f.

    Parameters
    ----------
    f: function
    iterations
    args: Passed to f
    kwargs: Passed to f

    Returns
    -------
    Tuple: (Minimimum runtime across iterations, output of f)

    """
    rs = []
    for _ in range(iterations):
        start = time.time()
        out = f(*args, **kwargs)
        end = time.time()
        rs.append(end - start)
    return np.min(rs), out


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
        storage: Optional[str] = None,
        threads: Optional[int] = None,
        regularization_strength: Optional[float] = None,
        hessian_approx: Optional[float] = None,
        diagnostics_level: Optional[str] = None,
    ):
        self.problem_name = problem_name
        self.library_name = library_name
        self.num_rows = num_rows
        self.storage = storage
        self.threads = threads
        self.regularization_strength = regularization_strength
        self.hessian_approx = hessian_approx
        self.diagnostics_level = diagnostics_level

    param_names = [
        "problem_name",
        "library_name",
        "num_rows",
        "storage",
        "threads",
        "regularization_strength",
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
    regularization_strength=None,
    storage="dense",
    hessian_approx=0.0,
    diagnostics_level="basic",
)


def get_params_from_fname(fname: str) -> BenchmarkParams:
    """
    Map file name to a BenchmarkParams instance.

    File names are formatted as:
    problem_library_numrows_storage_threads_reg_hessian_diag.pkl

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
    param_types = [str, str, int, str, int, float, float, str]

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


def _standardize_features(
    X: Union[np.ndarray, pd.DataFrame, csc_matrix, tm.MatrixBase],
) -> Union[np.ndarray, pd.DataFrame, csc_matrix, tm.MatrixBase]:
    """
    Standardize features by scaling to unit L2 norm per column.

    For consistency across sparse/dense, we use L2 norm scaling (no centering).
    This ensures all benchmark libraries start with the same pre-processed data.
    """
    dtype = np.float64

    if isinstance(X, pd.DataFrame):
        # Preserve DataFrame type
        X_arr = X.values.astype(dtype)
        col_norms = np.linalg.norm(X_arr, axis=0)
        col_norms[col_norms == 0] = 1.0
        return pd.DataFrame(X_arr / col_norms, columns=X.columns, index=X.index)

    if isinstance(X, np.ndarray):
        # Dense: scale columns to unit L2 norm
        X = np.asarray(X, dtype=dtype)
        col_norms = np.linalg.norm(X, axis=0)
        col_norms[col_norms == 0] = 1.0
        return X / col_norms

    elif isinstance(X, csc_matrix):
        # Sparse: scale columns to unit L2 norm
        X = X.astype(dtype)
        col_norms = np.sqrt(X.power(2).sum(axis=0)).A1
        col_norms[col_norms == 0] = 1.0
        from scipy.sparse import diags

        return X @ diags(1.0 / col_norms)

    elif isinstance(X, tm.MatrixBase):
        # tabmat matrices: skip standardization
        # glum is the only library that can use tabmat directly, and it handles
        # standardization internally. Other libraries would need conversion anyway.
        return X

    else:
        # Unknown type, return as-is
        return X


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
        glum_bench,
        h2o_bench,
        liblinear_bench,
        skglm_bench,
        sklearn_bench,
        zeros_bench,
    )

    all_libraries = {
        "glum": glum_bench,
        "zeros": zeros_bench,
        "celer": celer_bench,
        "h2o": h2o_bench,
        "liblinear": liblinear_bench,
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
    **kwargs,
):
    """
    Run the benchmark problem specified by 'params', 'iterations' times.

    Parameters
    ----------
    params
    iterations
    diagnostics_level
    standardize
        Whether to standardize features before fitting. Default True for benchmarks.
    kwargs

    Returns
    -------
    Tuple: Result data on this run, and the regularization applied
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
        storage=params.storage,
    )

    # Standardize features for better convergence across all libraries
    if standardize:
        dat["X"] = _standardize_features(dat["X"])

    os.environ["OMP_NUM_THREADS"] = str(params.threads)

    if params.regularization_strength is None:
        params.regularization_strength = P.regularization_strength

    # Weights have been multiplied by exposure. The new sum of weights
    # should influence the objective function (in order to keep everything comparable
    # to the "weights instead of offset" setup), but this will get undone by weight
    # normalization. So instead divide the penalty by the new weight sum divided by
    # the old weight sum
    reg_multiplier = (
        1 / dat["sample_weight"].mean() if "sample_weight" in dat.keys() else None
    )

    result = L(
        dat,
        distribution=P.distribution,
        alpha=params.regularization_strength,
        l1_ratio=P.l1_ratio,
        iterations=iterations,
        diagnostics_level=diagnostics_level,
        reg_multiplier=reg_multiplier,
        hessian_approx=params.hessian_approx,
        **kwargs,
    )

    if len(result) > 0:
        # Use best_alpha from CV if available, otherwise use regularization_strength
        alpha_for_obj = result.get("best_alpha", P.regularization_strength)
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

    return result, params.regularization_strength
