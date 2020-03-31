from typing import Callable, Dict, Union

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
    """
    Due to the way we have set up this problem, by rescaling the target variable, it
    is appropriate to pass what is modeled as an 'exposure' as a weight. Everywhere else,
    exposures will be referred to as weights.
    """
    df = pd.read_parquet(git_root.git_root("data/data.parquet"))
    if num_rows is not None:
        df = df.iloc[:num_rows]
    X = df[[col for col in df.columns if col not in ["y", "exposure"]]]
    y = df["y"]
    exposure = df["exposure"]
    return dict(X=X, y=y, weights=exposure)


def load_simple_insurance_data_no_weights(
    num_rows: int = None,
) -> Dict[str, Union[pd.Series, pd.DataFrame]]:
    dat = load_simple_insurance_data(num_rows)
    return dict(X=dat["X"], y=dat["y"])


def get_all_problems() -> Dict[str, Problem]:
    regularization_strength = 0.001
    distribution = "poisson"

    problems = dict()
    for suffix, l1_ratio in [("l2", 0.0), ("net", 0.5), ("lasso", 1.0)]:
        problems["simple_insurance_" + suffix] = Problem(
            data_loader=load_simple_insurance_data,
            distribution=distribution,
            regularization_strength=regularization_strength,
            l1_ratio=l1_ratio,
        )
        problems["simple_insurance_no_weights" + suffix] = Problem(
            data_loader=load_simple_insurance_data_no_weights,
            distribution=distribution,
            regularization_strength=regularization_strength,
            l1_ratio=l1_ratio,
        )
    return problems
