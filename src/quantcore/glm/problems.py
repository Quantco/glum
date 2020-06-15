import os
from functools import partial
from typing import Callable, Dict, Optional, Tuple

import attr
import numpy as np
from git_root import git_root
from joblib import Memory
from scipy.sparse import csc_matrix

import quantcore.glm.matrix as mx

from .data import (
    generate_intermediate_insurance_dataset,
    generate_narrow_insurance_dataset,
    generate_real_insurance_dataset,
    generate_wide_insurance_dataset,
)
from .util import cache_location, exposure_and_offset_to_weights, get_tweedie_p

joblib_memory = Memory(cache_location, verbose=0)


@attr.s
class Problem:
    data_loader = attr.ib(type=Callable)
    distribution = attr.ib(type=str)
    regularization_strength = attr.ib(type=float)
    l1_ratio = attr.ib(type=float)


@joblib_memory.cache
def load_data(
    loader_func: Callable[
        [Optional[int], Optional[float], Optional[str]],
        Tuple[np.ndarray, np.ndarray, np.ndarray],
    ],
    num_rows: int = None,
    storage: str = "dense",
    single_precision: bool = False,
    noise: float = None,
    distribution: str = "poisson",
    data_setup: str = "weights",
) -> Dict[str, np.ndarray]:
    """
    Due to the way we have set up this problem, by rescaling the target variable, it
    is appropriate to pass what is modeled as an 'exposure' as a weight. Everywhere else,
    exposures will be referred to as weights.
    """
    # TODO: add a weights_and_offset option
    if data_setup not in ["weights", "offset", "no-weights"]:
        raise NotImplementedError
    X, y, exposure = loader_func(num_rows, noise, distribution)

    if single_precision:
        X = X.astype(np.float32)
        y = y.astype(np.float32)
        if exposure is not None:
            exposure = exposure.astype(np.float32)

    if storage == "sparse":
        X = csc_matrix(X)
    elif storage.startswith("split"):
        threshold = float(storage.split("split")[1])
        X = mx.csc_to_split(csc_matrix(X), threshold)
    if data_setup == "weights":
        # The exposure correction doesn't make sense for these distributions since
        # they don't use a log link (plus binomial isn't in the tweedie family),
        # but this is what we were doing before.
        if distribution in ["gaussian", "binomial"]:
            return dict(X=X, y=y, weights=exposure)
        # when poisson, should be y=y, weights=exposure
        # instead have y = y / exposure, weight = exposure
        y, sample_weight = exposure_and_offset_to_weights(
            get_tweedie_p(distribution), y, exposure
        )
        return dict(X=X, y=y * exposure, weights=sample_weight)
    if data_setup == "offset":
        log_exposure = np.log(exposure)
        assert np.all(np.isfinite(log_exposure))
        # y has already been divided by exposure loader_func, so undo it here
        return dict(X=X, y=y * exposure, offset=log_exposure)
    # data_setup = "no_weights"
    return dict(X=X, y=y)


def get_all_problems() -> Dict[str, Problem]:
    regularization_strength = 0.001
    distributions = ["gaussian", "poisson", "gamma", "tweedie-p=1.5", "binomial"]
    load_funcs = {
        "intermediate-insurance": generate_intermediate_insurance_dataset,
        "narrow-insurance": generate_narrow_insurance_dataset,
        "wide-insurance": generate_wide_insurance_dataset,
    }
    if os.path.isfile(git_root("data", "X.parquet")):
        load_funcs["real-insurance"] = generate_real_insurance_dataset

    problems = dict()
    for penalty_str, l1_ratio in [("l2", 0.0), ("net", 0.5), ("lasso", 1.0)]:
        for distribution in distributions:
            suffix = penalty_str + "-" + distribution
            dist = distribution

            for problem_name, load_fn in load_funcs.items():
                for data_setup in ["weights", "no-weights", "offset"]:
                    problems["-".join((problem_name, data_setup, suffix))] = Problem(
                        data_loader=partial(
                            load_data, load_fn, distribution=dist, data_setup=data_setup
                        ),
                        distribution=distribution,
                        regularization_strength=regularization_strength,
                        l1_ratio=l1_ratio,
                    )

    return problems
