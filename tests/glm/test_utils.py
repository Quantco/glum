import numpy as np
import pandas as pd
import pytest

from glum._util import _add_missing_categories, _align_df_categories


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
            "x8": pd.Categorical(["a", pd.NA], categories=["b", "a"]),
        }
    )


def test_align_df_categories_numeric(df):
    dtypes = {column: np.float64 for column in df}
    has_missing_category = {column: False for column in df}
    missing_method = "fail"

    expected = pd.DataFrame(
        {
            "x1": np.array([0, 1], dtype="int64"),
            "x2": np.array([0, 1], dtype="bool"),
            "x3": np.array([0, 1], dtype="float64"),
            "x4": ["0", "1"],
            "x5": ["a", "b"],
            "x6": pd.Categorical(["a", "b"]),
            "x7": pd.Categorical(["a", "b"], categories=["b", "a"]),
            "x8": pd.Categorical(["a", pd.NA], categories=["b", "a"]),
        }
    )

    pd.testing.assert_frame_equal(
        _align_df_categories(df, dtypes, has_missing_category, missing_method), expected
    )


def test_align_df_categories_categorical(df):
    df = df[["x5", "x6", "x7", "x8"]]
    dtypes = {column: pd.CategoricalDtype(["a", "b"]) for column in df}
    has_missing_category = {column: False for column in df}
    missing_method = "fail"

    expected = pd.DataFrame(
        {
            "x5": pd.Categorical(["a", "b"]),
            "x6": pd.Categorical(["a", "b"]),
            "x7": pd.Categorical(["a", "b"]),
            "x8": pd.Categorical(["a", pd.NA], categories=["b", "a"]),
        },
        dtype=pd.CategoricalDtype(["a", "b"]),
    )

    pd.testing.assert_frame_equal(
        _align_df_categories(df, dtypes, has_missing_category, missing_method),
        expected,
    )


def test_align_df_categories_excess_columns(df):
    dtypes = {"x1": np.float64}
    has_missing_category = {column: False for column in df}
    missing_method = "fail"

    expected = pd.DataFrame(
        {
            "x1": np.array([0, 1], dtype="int64"),
            "x2": np.array([0, 1], dtype="bool"),
            "x3": np.array([0, 1], dtype="float64"),
            "x4": ["0", "1"],
            "x5": ["a", "b"],
            "x6": pd.Categorical(["a", "b"]),
            "x7": pd.Categorical(["a", "b"], categories=["b", "a"]),
            "x8": pd.Categorical(["a", pd.NA], categories=["b", "a"]),
        }
    )

    pd.testing.assert_frame_equal(
        _align_df_categories(df, dtypes, has_missing_category, missing_method), expected
    )


def test_align_df_categories_missing_columns(df):
    dtypes = {"x0": np.float64}
    has_missing_category = {column: False for column in df}
    missing_method = "fail"

    expected = pd.DataFrame(
        {
            "x1": np.array([0, 1], dtype="int64"),
            "x2": np.array([0, 1], dtype="bool"),
            "x3": np.array([0, 1], dtype="float64"),
            "x4": ["0", "1"],
            "x5": ["a", "b"],
            "x6": pd.Categorical(["a", "b"]),
            "x7": pd.Categorical(["a", "b"], categories=["b", "a"]),
            "x8": pd.Categorical(["a", pd.NA], categories=["b", "a"]),
        }
    )

    pd.testing.assert_frame_equal(
        _align_df_categories(df, dtypes, has_missing_category, missing_method), expected
    )


@pytest.mark.parametrize("has_missings", [False, True])
def test_align_df_categories_convert(df, has_missings):
    df = df[["x5", "x6", "x7", "x8"]]
    dtypes = {column: pd.CategoricalDtype(["a", "b"]) for column in df}
    has_missing_category = {column: has_missings for column in df}
    missing_method = "convert"

    expected = pd.DataFrame(
        {
            "x5": pd.Categorical(["a", "b"]),
            "x6": pd.Categorical(["a", "b"]),
            "x7": pd.Categorical(["a", "b"]),
            "x8": pd.Categorical(["a", pd.NA], categories=["b", "a"]),
        },
        dtype=pd.CategoricalDtype(["a", "b"]),
    )

    if has_missings:
        pd.testing.assert_frame_equal(
            _align_df_categories(
                df[["x5", "x6", "x7", "x8"]],
                dtypes,
                has_missing_category,
                missing_method,
            ),
            expected,
        )
    else:
        with pytest.raises(ValueError, match="contains unseen categories"):
            _align_df_categories(
                df[["x5", "x6", "x7", "x8"]],
                dtypes,
                has_missing_category,
                missing_method,
            )


def test_align_df_categories_raise_on_unseen(df):
    dtypes = {column: pd.CategoricalDtype(["a", "b"]) for column in df}
    has_missing_category = {column: False for column in df}
    missing_method = "fail"

    with pytest.raises(ValueError, match="contains unseen categories"):
        _align_df_categories(
            df,
            dtypes,
            has_missing_category,
            missing_method,
        )


def test_align_df_categories_not_df():
    with pytest.raises(TypeError):
        _align_df_categories(np.array([[0], [1]]), {"x0": np.float64}, {}, "fail")


@pytest.fixture()
def df_na():
    return pd.DataFrame(
        {
            "num": np.array([0, 1], dtype="float64"),
            "cat": pd.Categorical(["a", "b"]),
            "cat_na": pd.Categorical(["a", pd.NA]),
            "cat2": pd.Categorical(["a", "b"]),
        }
    )


def test_add_missing_categories(df_na):
    categorical_format = "{name}[{category}]"
    cat_missing_name = "(M)"
    dtypes = df_na.dtypes
    feature_names = [
        "num",
        "num[(M)]",
        "cat[a]",
        "cat[b]",
        "cat[(M)]",
        "cat_na[a]",
        "cat_na[(M)]",
        "cat2[a]",
        "cat2[b]",
    ]

    expected = pd.DataFrame(
        {
            "num": np.array([0, 1], dtype="float64"),
            "cat": pd.Categorical(["a", "b"], categories=["a", "b", "(M)"]),
            "cat_na": pd.Categorical(["a", "(M)"], categories=["a", "(M)"]),
            "cat2": pd.Categorical(["a", "b"], categories=["a", "b"]),
        }
    )

    pd.testing.assert_frame_equal(
        _add_missing_categories(
            df=df_na,
            dtypes=dtypes,
            feature_names=feature_names,
            categorical_format=categorical_format,
            cat_missing_name=cat_missing_name,
        ),
        expected,
    )


def test_raise_on_existing_missing(df_na):
    categorical_format = "{name}[{category}]"
    cat_missing_name = "(M)"
    dtypes = df_na.dtypes
    feature_names = [
        "num",
        "num[(M)]",
        "cat[a]",
        "cat[b]",
        "cat[(M)]",
        "cat_na[a]",
        "cat_na[(M)]",
        "cat2[a]",
        "cat2[b]",
    ]

    df = df_na
    df["cat_na"] = df["cat_na"].cat.add_categories("(M)")
    df.loc[df.cat_na.isna(), "cat_na"] = "(M)"

    with pytest.raises(ValueError):
        _add_missing_categories(
            df=df,
            dtypes=dtypes,
            feature_names=feature_names,
            categorical_format=categorical_format,
            cat_missing_name=cat_missing_name,
        )
