from functools import partial
from os.path import isfile
from typing import Callable, Dict, Optional, Tuple

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


def load_data(
    loader_func: Callable[
        [Optional[int], Optional[float], str], Tuple[np.ndarray, np.ndarray, np.ndarray]
    ],
    num_rows: int = None,
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
    x, y, exposure = loader_func(num_rows, noise, distribution)
    if data_setup == "weights":
        return dict(X=x, y=y, weights=exposure)
    if data_setup == "offset":
        offset = np.log(exposure)
        assert np.all(np.isfinite(offset))
        # y has already been divided by exposure loader_func, so undo it here
        return dict(X=x, y=y * exposure, offset=offset)
    # data_setup = "no-weights"
    return dict(X=x, y=y)


def load_narrow_insurance_data(
    num_rows: int = None, noise: float = None, distribution: str = "poisson",
) -> Dict[str, np.ndarray]:
    return load_data(generate_narrow_insurance_dataset, num_rows, noise, distribution,)


def get_all_problems() -> Dict[str, Problem]:
    regularization_strength = 0.001
    distributions = ["gaussian", "poisson", "gamma", "tweedie_p=1.5"]
    load_funcs = {
        "narrow-insurance": generate_narrow_insurance_dataset,
        "wide-insurance": generate_wide_insurance_dataset,
    }
    if isfile(git_root("data", "X.parquet")):
        load_funcs["real-insurance"] = generate_real_insurance_dataset

    problems = dict()
    for penalty_str, l1_ratio in [("l2", 0.0), ("net", 0.5), ("lasso", 1.0)]:
        for distribution in distributions:
            suffix = penalty_str + "-" + distribution
            dist = distribution
            if "tweedie" in dist:
                dist = "tweedie"

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
