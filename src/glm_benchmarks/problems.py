from typing import Callable, Dict, Optional, Union

import attr
import numpy as np
from scipy import sparse as sps

from .data import generate_simple_insurance_dataset, generate_sparse_insurance_dataset


@attr.s
class Problem:
    data_loader = attr.ib(
        type=Callable[[Optional[int]], Dict[str, Union[np.ndarray, sps.spmatrix]]]
    )
    distribution = attr.ib(type=str)
    regularization_strength = attr.ib(type=float)
    l1_ratio = attr.ib(type=float)


def load_simple_insurance_data(num_rows=None,) -> Dict[str, np.ndarray]:
    """
    Due to the way we have set up this problem, by rescaling the target variable, it
    is appropriate to pass what is modeled as an 'exposure' as a weight. Everywhere else,
    exposures will be referred to as weights.
    """
    X, y, exposure = generate_simple_insurance_dataset(num_rows)
    return dict(X=X, y=y, weights=exposure)


def load_simple_insurance_data_no_weights(
    num_rows: int = None,
) -> Dict[str, np.ndarray]:
    X, y, _ = generate_simple_insurance_dataset(num_rows)
    return dict(X=X, y=y)


def load_sparse_insurance_data(
    num_rows=None,
) -> Dict[str, Union[np.ndarray, sps.spmatrix]]:
    """
    Due to the way we have set up this problem, by rescaling the target variable, it
    is appropriate to pass what is modeled as an 'exposure' as a weight. Everywhere else,
    exposures will be referred to as weights.
    """
    X, y, exposure = generate_sparse_insurance_dataset(num_rows)
    return dict(X=X, y=y, weights=exposure)


def load_sparse_insurance_data_no_weights(num_rows=None,) -> Dict[str, np.ndarray]:
    X, y, _ = generate_sparse_insurance_dataset(num_rows)
    return dict(X=X, y=y)


def get_all_problems() -> Dict[str, Problem]:
    regularization_strength = 0.001
    distributions = ["gaussian", "poisson"]

    problems = dict()
    for penalty_str, l1_ratio in [("l2", 0.0), ("net", 0.5), ("lasso", 1.0)]:
        for distribution in distributions:
            suffix = penalty_str + "_" + distribution

            problems["simple_insurance_" + suffix] = Problem(
                data_loader=load_simple_insurance_data,
                distribution=distribution,
                regularization_strength=regularization_strength,
                l1_ratio=l1_ratio,
            )

            problems["simple_insurance_no_weights_" + suffix] = Problem(
                data_loader=load_simple_insurance_data_no_weights,
                distribution=distribution,
                regularization_strength=regularization_strength,
                l1_ratio=l1_ratio,
            )

            problems["sparse_insurance_" + suffix] = Problem(
                data_loader=load_simple_insurance_data,
                distribution=distribution,
                regularization_strength=regularization_strength,
                l1_ratio=l1_ratio,
            )

            problems["sparse_insurance_no_weights_" + suffix] = Problem(
                data_loader=load_sparse_insurance_data,
                distribution=distribution,
                regularization_strength=regularization_strength,
                l1_ratio=l1_ratio,
            )

    return problems
