import glob
import os
import shutil
import time
from functools import reduce
from typing import Callable, Dict, List, Optional, Tuple, Union

import click
import numpy as np
from scipy import sparse as sps

from .sklearn_fork import GeneralizedLinearRegressor, TweedieDistribution
from .sklearn_fork._solvers import eta_mu_objective

benchmark_convergence_tolerance = 1e-4
cache_location = os.environ.get("GLM_BENCHMARKS_CACHE", None)


def runtime(f, iterations, *args, **kwargs):
    rs = []
    for i in range(iterations):
        start = time.time()
        out = f(*args, **kwargs)
        end = time.time()
        rs.append(end - start)
    return np.min(rs), out


def get_sklearn_family(distribution):
    family = distribution
    if family == "gaussian":
        family = "normal"
    elif "tweedie" in family:
        tweedie_p = float(family.split("-p=")[1])
        family = TweedieDistribution(tweedie_p)  # type: ignore
    return family


def get_obj_val(
    dat: Dict[str, Union[np.ndarray, sps.spmatrix]],
    distribution: str,
    alpha: float,
    l1_ratio: float,
    intercept: float,
    coefs: np.ndarray,
    tweedie_p: float = None,
) -> float:

    model = GeneralizedLinearRegressor(
        alpha=alpha, l1_ratio=l1_ratio, family=get_sklearn_family(distribution),
    )
    model.set_up_for_fit(dat["y"])

    full_coefs = np.concatenate(([intercept], coefs))
    offset = dat.get("offset")
    X_dot_coef = dat["X"].to_numpy().dot(coefs) + intercept
    if offset is not None:
        X_dot_coef += offset

    zeros = np.zeros(dat["X"].shape[0])
    y = dat["y"].astype(coefs.dtype)
    weights = dat.get("weights", np.ones_like(y)).astype(coefs.dtype)
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
) -> Tuple[np.ndarray, np.ndarray]:
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
        exposure ** (2 - power)
        if sample_weight is None
        else sample_weight * exposure ** (2 - power)
    )
    return y, sample_weight


class BenchmarkParams:
    """
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
        single_precision: Optional[bool] = None,
        regularization_strength: Optional[float] = None,
        cv: Optional[bool] = None,
        hessian_approx: Optional[float] = None,
    ):

        self.problem_name = problem_name
        self.library_name = library_name
        self.num_rows = num_rows
        self.storage = storage
        self.threads = threads
        self.single_precision = single_precision
        self.regularization_strength = regularization_strength
        self.cv = cv
        self.hessian_approx = hessian_approx

    param_names = [
        "problem_name",
        "library_name",
        "num_rows",
        "storage",
        "threads",
        "single_precision",
        "regularization_strength",
        "cv",
        "hessian_approx",
    ]

    def update_params(self, **kwargs):
        for k, v in kwargs.items():
            assert k in self.param_names
            setattr(self, k, v)
        return self

    def get_result_fname(self):
        return "_".join(str(getattr(self, k)) for k in self.param_names)


defaults = dict(
    threads=os.environ.get("OMP_NUM_THREADS", os.cpu_count()),
    problem_name=None,
    library_name=None,
    num_rows=None,
    regularization_strength=None,
    storage="dense",
    cv=False,
    single_precision=False,
    hessian_approx=0.0,
)


def benchmark_params_cli(func: Callable) -> Callable:
    @click.option(
        "--problem_name",
        type=str,
        help="Specify a comma-separated list of benchmark problems you want to run. Leaving this blank will default to running all problems.",
    )
    @click.option(
        "--library_name",
        help="Specify a comma-separated list of libaries to benchmark. Leaving this blank will default to running all problems.",
    )
    @click.option(
        "--num_rows",
        type=int,
        help="Pass an integer number of rows. This is useful for testing and development. The default is to use the full dataset.",
    )
    @click.option(
        "--storage",
        type=str,
        help="Specify the storage format. Currently supported: dense, sparse. Leaving this black will default to dense.",
    )
    @click.option(
        "--threads",
        type=int,
        help="Specify the number of threads. If not set, it will use OMP_NUM_THREADS. If that's not set either, it will default to os.cpu_count().",
    )
    @click.option("--cv", type=bool, help="Cross-validation")
    @click.option(
        "--single_precision", type=bool, help="Whether to use 32-bit data",
    )
    @click.option(
        "--regularization_strength",
        type=float,
        help="Regularization strength. Set to None to use the default value of the problem.",
    )
    @click.option(
        "--hessian_approx",
        type=float,
        help="Threshold for dropping rows in the IRLS approximate Hessian update.",
    )
    def wrapped_func(
        problem_name: Optional[str],
        library_name: Optional[str],
        num_rows: Optional[int],
        storage: Optional[str],
        threads: Optional[int],
        cv: Optional[bool],
        single_precision: Optional[bool],
        regularization_strength: Optional[float],
        hessian_approx: Optional[float],
        *args,
        **kwargs,
    ):
        params = BenchmarkParams(
            problem_name,
            library_name,
            num_rows,
            storage,
            threads,
            single_precision,
            regularization_strength,
            cv,
            hessian_approx,
        )
        return func(params, *args, **kwargs)

    return wrapped_func


@click.command()
@benchmark_params_cli
def get_params(params):
    get_params.out = params


def get_params_from_fname(fname: str) -> BenchmarkParams:
    cli_list = reduce(
        lambda x, y: x + y,
        [
            ["--" + elt[0], elt[1]]
            for elt in zip(BenchmarkParams.param_names, fname.strip(".pkl").split("_"))
            if elt[1] != "None"
        ],
    )
    get_params(cli_list, standalone_mode=False)
    return get_params.out  # type: ignore


def _get_size_of_cache_directory():
    return sum(
        os.path.getsize(x) for x in glob.glob(f"{cache_location}/**", recursive=True)
    )


def clear_cache(force=False):
    """Clear the cache directory if its size exceeds a threshold."""

    if cache_location is None:
        return

    cache_size_limit = float(
        os.environ.get("GLM_BENCHMARKS_CACHE_SIZE_LIMIT", 1024 ** 3)
    )

    if force or _get_size_of_cache_directory() > cache_size_limit:
        shutil.rmtree(cache_location)


def get_comma_sep_names(xs: str) -> List[str]:
    return [x.strip() for x in xs.split(",")]


def get_tweedie_p(distribution):
    tweedie = "tweedie" in distribution
    if tweedie:
        return float(distribution.split("=")[-1])
    if "poisson" == distribution:
        return 1
    if "gamma" == distribution:
        return 2
    if "gaussian" in distribution:
        return 0
    return None
