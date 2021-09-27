import numpy as np
import pandas as pd
import pytest

from quantcore.glm._util import _align_df_dtypes


@pytest.fixture()
def df():
    return pd.DataFrame(
        {
            "x1": np.array([0, 1], dtype="int64"),
            "x2": np.array([0, 1], dtype="bool"),
            "x3": np.array([0, 1], dtype="float64"),
            "x4": ["0", "1"],
            "x5": ["a", "b"],
            "x6": pd.Categorical(["a", "b"]),
            "x7": pd.Categorical(["a", "b"], categories=["b", "a"]),
        }
    )


def test_align_df_dtypes_numeric(df):

    dtypes = {column: np.float64 for column in df}

    expected = pd.DataFrame(
        {
            "x1": np.array([0, 1], dtype="int64"),
            "x2": np.array([0, 1], dtype="bool"),
            "x3": np.array([0, 1], dtype="float64"),
            "x4": np.array([0, 1], dtype="int64"),
            "x5": np.array([np.nan, np.nan], dtype="float64"),
            "x6": np.array([np.nan, np.nan], dtype="float64"),
            "x7": np.array([np.nan, np.nan], dtype="float64"),
        }
    )

    pd.testing.assert_frame_equal(_align_df_dtypes(df, dtypes), expected)


def test_align_df_dtypes_categorical(df):

    dtypes = {column: pd.CategoricalDtype(["a", "b"]) for column in df}

    expected = pd.DataFrame(
        {
            "x1": [np.nan, np.nan],
            "x2": [np.nan, np.nan],
            "x3": [np.nan, np.nan],
            "x4": [np.nan, np.nan],
            "x5": ["a", "b"],
            "x6": ["a", "b"],
            "x7": ["a", "b"],
        },
        dtype=pd.CategoricalDtype(["a", "b"]),
    )

    pd.testing.assert_frame_equal(_align_df_dtypes(df, dtypes), expected)


def test_align_df_dtypes_excess_columns(df):
    dtypes = {"x1": np.float64}
    expected = pd.DataFrame({"x1": np.array([0, 1], dtype="int64")})
    pd.testing.assert_frame_equal(_align_df_dtypes(df, dtypes), expected)


def test_align_df_dtypes_missing_columns(df):
    with pytest.raises(KeyError):
        _align_df_dtypes(df, {"x0": np.float64})


def test_align_df_dtypes_not_df():
    with pytest.raises(TypeError):
        _align_df_dtypes(np.array([[0], [1]]), {"x0": np.float64})
