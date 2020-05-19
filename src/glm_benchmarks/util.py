import glob
import os
import shutil
import time
from functools import reduce
from typing import Callable, Dict, Optional, Tuple, Union

import click
import numpy as np
from scipy import sparse as sps

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


def _get_minus_tweedie_ll_by_obs(eta: np.ndarray, y: np.ndarray, p: float):
    if p == 0:
        expected_y = eta
    else:
        expected_y = np.exp(eta)

    def _f(exp: float):
        if exp == 0:
            # equal to log expected y; limit as exp goes to 1 of below func
            return eta
        return expected_y ** exp / exp

    return _f(2 - p) - y * _f(1 - p)


def _get_minus_gamma_ll_by_obs(eta: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Only up to a constant! From h2o documentation.
    """
    return _get_minus_tweedie_ll_by_obs(eta, y, 2)


def _get_poisson_ll_by_obs(eta: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Only up to a constant!
    """
    return eta * y - np.exp(eta)


def _get_minus_gaussian_ll_by_obs(eta: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    The normal log-likelihood, up to a constant.
    """
    return (y - eta) ** 2 / 2


def _get_linear_prediction_part(
    x: Union[np.ndarray, sps.spmatrix], coefs: np.ndarray, intercept: float
) -> np.ndarray:
    return x.dot(coefs) + intercept


def _get_penalty(alpha: float, l1_ratio: float, coefs: np.ndarray) -> float:
    l1 = np.sum(np.abs(coefs))
    l2 = np.sum(coefs ** 2)
    penalty = alpha * (l1_ratio * l1 + (1 - l1_ratio) * l2)
    return penalty


def get_obj_val(
    dat: Dict[str, Union[np.ndarray, sps.spmatrix]],
    distribution: str,
    alpha: float,
    l1_ratio: float,
    intercept: float,
    coefs: np.ndarray,
    tweedie_p: float = None,
) -> float:
    assert "offset" not in dat.keys()
    weights = dat.get("weights", np.ones_like(dat["y"])).astype(np.float64)
    weights /= weights.sum()

    eta = _get_linear_prediction_part(dat["X"], coefs, intercept)

    if distribution == "poisson":
        minus_log_like_by_ob = -_get_poisson_ll_by_obs(eta, dat["y"])
    elif distribution == "gaussian":
        minus_log_like_by_ob = _get_minus_gaussian_ll_by_obs(eta, dat["y"])
    elif distribution == "gamma":
        minus_log_like_by_ob = _get_minus_gamma_ll_by_obs(eta, dat["y"])
    elif "tweedie" in distribution:
        assert tweedie_p is not None
        minus_log_like_by_ob = _get_minus_tweedie_ll_by_obs(eta, dat["y"], tweedie_p)
    else:
        raise NotImplementedError

    penalty = _get_penalty(alpha, l1_ratio, coefs)

    return minus_log_like_by_ob.dot(weights) + penalty


def exposure_correction(
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
        offset = np.exp(np.asanyarray(offset))
        y = y / offset
        sample_weight = (
            offset ** (2 - power)
            if sample_weight is None
            else sample_weight * offset ** (2 - power)
        )
    if exposure is not None:
        exposure = np.asanyarray(exposure)
        y = y / exposure
        sample_weight = exposure if sample_weight is None else sample_weight * exposure

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
    ):

        self.problem_name = problem_name
        self.library_name = library_name
        self.num_rows = num_rows
        self.storage = storage
        self.threads = threads
        self.single_precision = single_precision
        self.regularization_strength = regularization_strength
        self.cv = cv

    param_names = [
        "problem_name",
        "library_name",
        "num_rows",
        "storage",
        "threads",
        "single_precision",
        "regularization_strength",
        "cv",
    ]

    def update_params(self, **kwargs):
        for k, v in kwargs.items():
            assert k in self.param_names
            setattr(self, k, v)
        return self

    def get_result_fname(self):
        return "_".join(str(getattr(self, k)) for k in self.param_names)


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
    def wrapped_func(
        problem_name: Optional[str],
        library_name: Optional[str],
        num_rows: Optional[int],
        storage: Optional[str],
        threads: Optional[int],
        cv: Optional[bool],
        single_precision: Optional[bool],
        regularization_strength: Optional[float],
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
