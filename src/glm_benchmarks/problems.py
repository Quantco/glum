from functools import partial
from os.path import isfile
from typing import Callable, Dict, Optional

import attr
import numpy as np
from git_root import git_root

from .data import (
    generate_narrow_insurance_dataset,
    generate_real_insurance_dataset,
    generate_wide_insurance_dataset,
)


@attr.s
class Problem:
    data_loader = attr.ib(type=Callable[[Optional[int]], Dict[str, np.ndarray]])
    distribution = attr.ib(type=str)
    regularization_strength = attr.ib(type=float)
    l1_ratio = attr.ib(type=float)


def load_narrow_insurance_data(
    num_rows: int = None, noise: float = None, distribution: str = "poisson",
) -> Dict[str, np.ndarray]:
    """
    Due to the way we have set up this problem, by rescaling the target variable, it
    is appropriate to pass what is modeled as an 'exposure' as a weight. Everywhere else,
    exposures will be referred to as weights.
    """
    X, y, exposure = generate_narrow_insurance_dataset(num_rows, noise, distribution)
    return dict(X=X, y=y, weights=exposure)


def load_narrow_insurance_data_no_weights(
    num_rows: int = None, noise: float = None, distribution: str = "poisson",
) -> Dict[str, np.ndarray]:
    X, y, _ = generate_narrow_insurance_dataset(num_rows, noise, distribution)
    return dict(X=X, y=y)


def load_wide_insurance_data(
    num_rows: int = None, noise: float = None, distribution: str = "poisson",
) -> Dict[str, np.ndarray]:
    """
    Due to the way we have set up this problem, by rescaling the target variable, it
    is appropriate to pass what is modeled as an 'exposure' as a weight. Everywhere else,
    exposures will be referred to as weights.
    """
    X, y, exposure = generate_wide_insurance_dataset(num_rows, noise, distribution)
    return dict(X=X, y=y, weights=exposure)


def load_wide_insurance_data_no_weights(
    num_rows: int = None, noise: float = None, distribution: str = "poisson",
) -> Dict[str, np.ndarray]:
    X, y, _ = generate_wide_insurance_dataset(num_rows, noise, distribution)
    return dict(X=X, y=y)


def load_real_insurance_data(
    num_rows: int = None, noise: float = None, distribution: str = "poisson",
) -> Dict[str, np.ndarray]:
    """
    Due to the way we have set up this problem, by rescaling the target variable, it
    is appropriate to pass what is modeled as an 'exposure' as a weight. Everywhere else,
    exposures will be referred to as weights.
    """
    X, y, exposure = generate_real_insurance_dataset(num_rows, noise, distribution)
    return dict(X=X, y=y, weights=exposure)


def get_all_problems() -> Dict[str, Problem]:
    regularization_strength = 0.1
    distributions = ["gaussian", "poisson", "gamma", "tweedie_p=1.5"]

    problems = dict()
    for penalty_str, l1_ratio in [("l2", 0.0), ("net", 0.5), ("lasso", 1.0)]:
        for distribution in distributions:
            suffix = penalty_str + "_" + distribution
            data_version = distribution
            if "tweedie" in data_version:
                data_version = "tweedie"

            problems["narrow_insurance_" + suffix] = Problem(
                data_loader=partial(
                    load_narrow_insurance_data, distribution=data_version
                ),
                distribution=distribution,
                regularization_strength=regularization_strength,
                l1_ratio=l1_ratio,
            )

            problems["narrow_insurance_no_weights_" + suffix] = Problem(
                data_loader=partial(
                    load_narrow_insurance_data_no_weights, distribution=data_version
                ),
                distribution=distribution,
                regularization_strength=regularization_strength,
                l1_ratio=l1_ratio,
            )

            problems["wide_insurance_" + suffix] = Problem(
                data_loader=partial(
                    load_wide_insurance_data, distribution=data_version
                ),
                distribution=distribution,
                regularization_strength=regularization_strength,
                l1_ratio=l1_ratio,
            )

            problems["wide_insurance_no_weights_" + suffix] = Problem(
                data_loader=partial(
                    load_wide_insurance_data_no_weights, distribution=data_version
                ),
                distribution=distribution,
                regularization_strength=regularization_strength,
                l1_ratio=l1_ratio,
            )

            # only add these problems if you have the data
            if isfile(git_root("data", "X.parquet")):
                problems["real_insurance_" + suffix] = Problem(
                    data_loader=partial(
                        load_real_insurance_data, distribution=data_version
                    ),
                    distribution=distribution,
                    regularization_strength=regularization_strength,
                    l1_ratio=l1_ratio,
                )

    return problems
