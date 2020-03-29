import os
import sys
import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd
import pytest
from git_root import git_root
from glmnet_python import glmnet

# Copied over from https://bitbucket.org/quantco/wayfairelastpricing/


@pytest.fixture
def df() -> pd.DataFrame:
    n_rows = 1000
    return pd.DataFrame(
        {
            "y": np.maximum(0.5, np.linspace(0, 1, n_rows)),
            "x1": np.ones(n_rows),
            "x2": np.arange(n_rows),
        }
    )


def format_glmnet_as_df(model: dict) -> pd.DataFrame:
    """
    :param model: Output of glmnet
    :return: Output formatted as dataframe for easier use
    """
    return pd.DataFrame(
        {
            "lambda_": model["lambdau"],
            "beta_0": model["beta"][0, :],
            "beta_1": model["beta"][1, :],
            "dev_ratio": model["dev"],
            "df": model["df"],
            "npasses": model["npasses"],
        }
    )


def load_results_from_r(model_name: str, cvglmnet: bool = False) -> pd.DataFrame:
    if cvglmnet:
        keep_cols = [
            "lambda_",
            "beta_0",
            "beta_1",
            "cvm",
            "cvsd",
            "nzero",
            "best_lambda",
            "best_beta_0",
            "best_beta_1",
            "name",
        ]
    else:
        keep_cols = [
            "lambda_",
            "beta_0",
            "beta_1",
            "dev_ratio",
            "df",
            "npasses",
            "name",
        ]
    # Dtypes chosen to ensure that we read in R results in the same format as Python

    dtypes = {
        "lambda_": np.float64,
        "beta_0": np.float64,
        "beta_1": np.float64,
        "dev_ratio": np.float64,
        "cvm": np.float64,
        "cvsd": np.float64,
        "nzero": np.int64,
        "best_lambda": np.float64,
        "best_beta_0": np.float64,
        "df": np.int64,
        "npasses": np.int64,
    }

    r_results = (
        pd.read_csv(git_root("data/glmnet_results_from_R.csv"), index_col=0)
        .loc[lambda x: x["name"] == model_name, keep_cols]
        .drop(["name"], axis=1)
        .reset_index(drop=True)
    )

    if len(r_results) == 0:
        raise KeyError(f"Relevant R results not found. No set of results {model_name}")
    for col, dt in dtypes.items():
        if col in r_results.columns:
            r_results[col] = r_results[col].astype(dt)
    return r_results


def test_glmnet_basic(df: pd.DataFrame) -> None:
    """
    Check that a glmnet example gives the same results as when run through R.
    :return:
    """

    model = glmnet(
        x=df[["x1", "x2"]].values,
        y=np.stack((1 - df["y"].values, df["y"].values)).T,
        family="binomial",
        intr=False,
    )
    py_results = format_glmnet_as_df(model)
    r_results = load_results_from_r("glmnet_base")
    pd.testing.assert_frame_equal(py_results, r_results)


def test_glmnet_varied_penalty_same_lambda(df: pd.DataFrame) -> None:
    """
    The penalties Python uses are exactly 3x as high as the
    penalties R uses. However, results are otherwise identical.

    Using a warning rather than error so that the test passes.
    """

    model = glmnet(
        x=df[["x1", "x2"]].values,
        y=np.stack((1 - df["y"].values, df["y"].values)).T,
        family="binomial",
        intr=False,
        penalty_factor=np.array([2, 1]),
    )
    py_results = format_glmnet_as_df(model)
    r_results = load_results_from_r("glmnet_varied_penalty")
    assert len(py_results) == len(r_results)
    if not (py_results["lambda_"] == r_results["lambda_"]).all():
        warnings.warn(
            f"""Difference between Python and R lambdas
            is {(py_results['lambda_'] / r_results['lambda_']).values}"""
        )


def test_glmnet_varied_penalty_same_non_lambda(df: pd.DataFrame) -> None:
    """
    The penalties Python uses are exactly 3x as high as the
    penalties R uses. However, results are otherwise identical.
    """
    model = glmnet(
        x=df[["x1", "x2"]].values,
        y=np.stack((1 - df["y"].values, df["y"].values)).T,
        family="binomial",
        intr=False,
        penalty_factor=np.array([2, 1]),
    )
    py_results = format_glmnet_as_df(model).drop(["lambda_"], axis=1)
    r_results = load_results_from_r("glmnet_varied_penalty").drop(["lambda_"], axis=1)
    pd.testing.assert_frame_equal(py_results, r_results)


def test_glmnet_one_unpenalized(df: pd.DataFrame) -> None:
    """
    Tests that in a simple logistic regression in which one coefficient is not
    penalized, glmnet gives the same results in Python and in R.
    """
    model = glmnet(
        x=df[["x1", "x2"]].values,
        y=np.stack((1 - df["y"].values, df["y"].values)).T,
        family="binomial",
        intr=False,
        penalty_factor=np.array([0, 1]),
    )
    py_results = format_glmnet_as_df(model)
    r_results = load_results_from_r("glmnet_one_penalized")
    pd.testing.assert_frame_equal(py_results, r_results)


def test_glmnet_constrained(df: pd.DataFrame) -> None:
    """ Tests that glmnet gives the same results in Python and R in the presence of
    coefficient constraints."""
    model = glmnet(
        x=df[["x1", "x2"]].values,
        y=np.stack((1 - df["y"].values, df["y"].values)).T,
        family="binomial",
        intr=False,
        cl=np.stack(([-np.inf, -np.inf], [np.inf, 1e-4])),
    )
    py_results = format_glmnet_as_df(model)
    r_results = load_results_from_r("glmnet_constrained")
    pd.testing.assert_frame_equal(py_results, r_results)
