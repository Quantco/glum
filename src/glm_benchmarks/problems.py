from typing import Callable, Dict

import attr
import git_root
import pandas as pd


@attr.s
class Problem:
    data_loader = attr.ib(type=Callable)
    distribution = attr.ib(type=str)
    regularization_strength = attr.ib(type=float)
    l1_ratio = attr.ib(type=float)


def load_simple_insurance_data(num_rows=None):
    df = pd.read_parquet(git_root.git_root("data/data.parquet"))
    if num_rows is not None:
        df = df.iloc[:num_rows]
    X = df[[col for col in df.columns if col not in ["y", "exposure"]]]
    y = df["y"]
    exposure = df["exposure"]
    return dict(X=X, y=y, exposure=exposure)


def get_all_problems() -> Dict[str, Problem]:
    problems = dict()
    for suffix, l1_ratio in [("l2", 0.0), ("net", 0.5), ("lasso", 1.0)]:
        problems["simple_insurance_" + suffix] = Problem(
            data_loader=load_simple_insurance_data,
            distribution="poisson",
            regularization_strength=0.001,
            l1_ratio=l1_ratio,
        )
    return problems
