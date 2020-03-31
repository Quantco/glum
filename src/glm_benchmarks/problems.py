from typing import Callable

import attr

from .data import generate_simple_insurance_dataset, generate_sparse_insurance_dataset


@attr.s
class Problem:
    data_loader = attr.ib(type=Callable)
    distribution = attr.ib(type=str)
    regularization_strength = attr.ib(type=float)
    l1_ratio = attr.ib(type=float)


def load_simple_insurance_data(num_rows=None):
    X, y, exposure = generate_simple_insurance_dataset(num_rows)
    return dict(X=X, y=y, exposure=exposure)


def load_sparse_insurance_data(num_rows=None):
    X, y, exposure = generate_sparse_insurance_dataset(num_rows)
    return dict(X=X, y=y, exposure=exposure)


def get_all_problems():
    problems = dict()
    for suffix, l1_ratio in [("l2", 0.0), ("net", 0.5), ("lasso", 1.0)]:
        problems["simple_insurance_" + suffix] = Problem(
            data_loader=load_simple_insurance_data,
            distribution="poisson",
            regularization_strength=0.001,
            l1_ratio=l1_ratio,
        )

        problems["sparse_insurance_" + suffix] = Problem(
            data_loader=load_simple_insurance_data,
            distribution="poisson",
            regularization_strength=0.001,
            l1_ratio=l1_ratio,
        )
    return problems
