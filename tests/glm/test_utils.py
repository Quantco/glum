import numpy as np
import pandas as pd
import pytest

from glum._util import _align_df_categories


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


def test_align_df_categories_numeric(df):

    dtypes = {column: np.float64 for column in df}

    expected = pd.DataFrame(
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

    pd.testing.assert_frame_equal(_align_df_categories(df, dtypes), expected)


def test_align_df_categories_categorical(df):

    dtypes = {column: pd.CategoricalDtype(["a", "b"]) for column in df}

    expected = pd.DataFrame(
        {
            "x1": [np.nan, np.nan],
            "x2": [np.nan, np.nan],
            "x3": [np.nan, np.nan],
            "x4": [np.nan, np.nan],
            "x5": pd.Categorical(["a", "b"]),
            "x6": pd.Categorical(["a", "b"]),
            "x7": pd.Categorical(["a", "b"]),
        },
        dtype=pd.CategoricalDtype(["a", "b"]),
    )

    pd.testing.assert_frame_equal(_align_df_categories(df, dtypes), expected)


def test_align_df_categories_excess_columns(df):

    dtypes = {"x1": np.float64}

    expected = pd.DataFrame(
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

    pd.testing.assert_frame_equal(_align_df_categories(df, dtypes), expected)


def test_align_df_categories_missing_columns(df):

    dtypes = {"x0": np.float64}

    expected = pd.DataFrame(
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

    pd.testing.assert_frame_equal(_align_df_categories(df, dtypes), expected)


def test_align_df_categories_not_df():
    with pytest.raises(TypeError):
        _align_df_categories(np.array([[0], [1]]), {"x0": np.float64})
