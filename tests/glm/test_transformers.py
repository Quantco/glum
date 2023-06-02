from typing import NamedTuple

import numpy as np
import pandas as pd
import pytest
import tabmat as tm

from glum import Decollinearizer


class PandasTestExpectation(NamedTuple):
    """Expectation for a Pandas test."""

    num_cols_to_drop: int
    num_cols_to_keep: int
    intercept_not_collinear: bool
    num_categories_replace: int
    design_matrix_rank_with_intercept: int


@pytest.fixture()
def df_independent_numeric():
    return (
        pd.DataFrame(
            {
                "a": [1.0, 0.0, 0.0],
                "b": [0.0, 1.0, 0.0],
            }
        ),
        PandasTestExpectation(
            num_cols_to_drop=0,
            num_cols_to_keep=2,
            intercept_not_collinear=True,
            num_categories_replace=0,
            design_matrix_rank_with_intercept=3,
        ),
    )


@pytest.fixture()
def df_dependent_numeric():
    return (
        pd.DataFrame(
            {
                "a": [1.0, 0.0, 0.0],
                "b": [2.0, 0.0, 0.0],
            }
        ),
        PandasTestExpectation(
            num_cols_to_drop=1,
            num_cols_to_keep=1,
            intercept_not_collinear=True,
            num_categories_replace=0,
            design_matrix_rank_with_intercept=2,
        ),
    )


@pytest.fixture()
def df_dependent_on_combination_numeric():
    return (
        pd.DataFrame(
            {
                "a": [1.0, 0.0, 0.0, 0.0],
                "b": [0.0, 1.0, 0.0, 0.0],
                "c": [2.0, 2.0, 0.0, 0.0],
            }
        ),
        PandasTestExpectation(
            num_cols_to_drop=1,
            num_cols_to_keep=2,
            intercept_not_collinear=True,
            num_categories_replace=0,
            design_matrix_rank_with_intercept=3,
        ),
    )


@pytest.fixture()
def df_dependent_on_intercept_numeric():
    return (
        pd.DataFrame(
            {
                "a": [1.0, 0.0, 0.0, 0.0],
                "b": [0.3, 0.3, 0.3, 0.3],
                "c": [0.0, 1.0, 0.0, 0.0],
            }
        ),
        PandasTestExpectation(
            num_cols_to_drop=1,
            num_cols_to_keep=2,
            intercept_not_collinear=False,
            num_categories_replace=0,
            design_matrix_rank_with_intercept=3,
        ),
    )


@pytest.fixture()
def df_independent_categorical():
    return (
        pd.DataFrame(
            {
                "a": pd.Categorical(["a", "b", "b"]),
                "b": pd.Categorical(["a", "a", "b"]),
            }
        ),
        PandasTestExpectation(
            num_cols_to_drop=0,
            num_cols_to_keep=2,
            intercept_not_collinear=True,
            num_categories_replace=0,
            design_matrix_rank_with_intercept=3,
        ),
    )


@pytest.fixture()
def df_dependent_categorical():
    return (
        pd.DataFrame(
            {
                "a": pd.Categorical(["a", "b", "c"]),
                "b": pd.Categorical(["a", "a", "c"]),
            }
        ),
        PandasTestExpectation(
            num_cols_to_drop=0,
            num_cols_to_keep=2,
            intercept_not_collinear=True,
            num_categories_replace=1,
            design_matrix_rank_with_intercept=3,
        ),
    )


def test_decollinearizer_independent_numeric(df_independent_numeric):
    df_independent, expectation = df_independent_numeric
    decollinearizer = Decollinearizer(fit_intercept=True)
    df_result = decollinearizer.fit_transform(df_independent)
    assert isinstance(df_result, pd.DataFrame)
    assert decollinearizer.intercept_safe == expectation.intercept_not_collinear
    assert len(decollinearizer.drop_columns) == expectation.num_cols_to_drop
    assert len(decollinearizer.keep_columns) == expectation.num_cols_to_keep
    assert len(df_result.columns) == expectation.num_cols_to_keep
    assert len(decollinearizer.replace_categories) == expectation.num_categories_replace


def test_np_decollinearizer_independent_numeric(df_independent_numeric):
    df_independent, expectation = df_independent_numeric
    X = df_independent.values
    decollinearizer = Decollinearizer(fit_intercept=True)
    X_result = decollinearizer.fit_transform(X)
    assert isinstance(X_result, np.ndarray)
    assert decollinearizer.intercept_safe == expectation.intercept_not_collinear
    assert len(decollinearizer.drop_columns) == expectation.num_cols_to_drop
    assert len(decollinearizer.keep_columns) == expectation.num_cols_to_keep
    assert X_result.shape[1] == expectation.num_cols_to_keep
    assert len(decollinearizer.replace_categories) == expectation.num_categories_replace


def test_decollinearizer_dependent_numeric(df_dependent_numeric):
    df, expectation = df_dependent_numeric
    decollinearizer = Decollinearizer(fit_intercept=True)
    df_result = decollinearizer.fit_transform(df)
    assert isinstance(df_result, pd.DataFrame)
    assert decollinearizer.intercept_safe == expectation.intercept_not_collinear
    assert len(decollinearizer.drop_columns) == expectation.num_cols_to_drop
    assert len(decollinearizer.keep_columns) == expectation.num_cols_to_keep
    assert len(df_result.columns) == expectation.num_cols_to_keep
    assert len(decollinearizer.replace_categories) == expectation.num_categories_replace


def test_np_decollinearizer_dependent_numeric(df_dependent_numeric):
    df_independent, expectation = df_dependent_numeric
    X = df_independent.values
    decollinearizer = Decollinearizer(fit_intercept=True)
    X_result = decollinearizer.fit_transform(X)
    assert isinstance(X_result, np.ndarray)
    assert decollinearizer.intercept_safe == expectation.intercept_not_collinear
    assert len(decollinearizer.drop_columns) == expectation.num_cols_to_drop
    assert len(decollinearizer.keep_columns) == expectation.num_cols_to_keep
    assert X_result.shape[1] == expectation.num_cols_to_keep
    assert len(decollinearizer.replace_categories) == expectation.num_categories_replace


def test_decollinearizer_dependent_on_combination_numeric(
    df_dependent_on_combination_numeric,
):
    df, expectation = df_dependent_on_combination_numeric
    decollinearizer = Decollinearizer(fit_intercept=True)
    df_result = decollinearizer.fit_transform(df)
    assert decollinearizer.intercept_safe == expectation.intercept_not_collinear
    assert len(decollinearizer.drop_columns) == expectation.num_cols_to_drop
    assert len(decollinearizer.keep_columns) == expectation.num_cols_to_keep
    assert len(df_result.columns) == expectation.num_cols_to_keep
    assert len(decollinearizer.replace_categories) == expectation.num_categories_replace


def test_np_decollinearizer_dependent_on_combination_numeric(
    df_dependent_on_combination_numeric,
):
    df_independent, expectation = df_dependent_on_combination_numeric
    X = df_independent.values
    decollinearizer = Decollinearizer(fit_intercept=True)
    X_result = decollinearizer.fit_transform(X)
    assert isinstance(X_result, np.ndarray)
    assert decollinearizer.intercept_safe == expectation.intercept_not_collinear
    assert len(decollinearizer.drop_columns) == expectation.num_cols_to_drop
    assert len(decollinearizer.keep_columns) == expectation.num_cols_to_keep
    assert X_result.shape[1] == expectation.num_cols_to_keep
    assert len(decollinearizer.replace_categories) == expectation.num_categories_replace


def test_decollinearizer_dependent_on_intercept_numeric(
    df_dependent_on_intercept_numeric,
):
    df, expectation = df_dependent_on_intercept_numeric
    decollinearizer = Decollinearizer(fit_intercept=True)
    df_result = decollinearizer.fit_transform(df)
    assert isinstance(df_result, pd.DataFrame)
    if decollinearizer.intercept_safe:
        assert len(decollinearizer.drop_columns) == expectation.num_cols_to_drop
        assert len(decollinearizer.keep_columns) == expectation.num_cols_to_keep
        assert len(df_result.columns) == expectation.num_cols_to_keep
    else:
        assert len(decollinearizer.drop_columns) == expectation.num_cols_to_drop - 1
        assert len(decollinearizer.keep_columns) == expectation.num_cols_to_keep + 1
        assert len(df_result.columns) == expectation.num_cols_to_keep + 1


def test_np_decollinearizer_dependent_on_intercept_numeric(
    df_dependent_on_intercept_numeric,
):
    df_independent, expectation = df_dependent_on_intercept_numeric
    X = df_independent.values
    decollinearizer = Decollinearizer(fit_intercept=True)
    X_result = decollinearizer.fit_transform(X)
    assert isinstance(X_result, np.ndarray)
    if decollinearizer.intercept_safe:
        assert len(decollinearizer.drop_columns) == expectation.num_cols_to_drop
        assert len(decollinearizer.keep_columns) == expectation.num_cols_to_keep
        assert X_result.shape[1] == expectation.num_cols_to_keep
    else:
        assert len(decollinearizer.drop_columns) == expectation.num_cols_to_drop - 1
        assert len(decollinearizer.keep_columns) == expectation.num_cols_to_keep + 1
        assert X_result.shape[1] == expectation.num_cols_to_keep + 1


def test_decollinearizer_no_intercept_independent(df_dependent_on_intercept_numeric):
    df, expectation = df_dependent_on_intercept_numeric
    decollinearizer = Decollinearizer(fit_intercept=False)
    df_result = decollinearizer.fit_transform(df)
    correction = 1 - expectation.intercept_not_collinear
    assert isinstance(df_result, pd.DataFrame)
    assert not decollinearizer.intercept_safe
    assert (
        len(decollinearizer.drop_columns) == expectation.num_cols_to_drop - correction
    )
    assert (
        len(decollinearizer.keep_columns) == expectation.num_cols_to_keep + correction
    )
    assert len(df_result.columns) == expectation.num_cols_to_keep + correction


def test_np_decollinearizer_no_intercept_independent(
    df_dependent_on_intercept_numeric,
):
    df_independent, expectation = df_dependent_on_intercept_numeric
    X = df_independent.values
    decollinearizer = Decollinearizer(fit_intercept=False)
    X_result = decollinearizer.fit_transform(X)
    correction = 1 - expectation.intercept_not_collinear
    assert isinstance(X_result, np.ndarray)
    assert not decollinearizer.intercept_safe
    assert (
        len(decollinearizer.drop_columns) == expectation.num_cols_to_drop - correction
    )
    assert (
        len(decollinearizer.keep_columns) == expectation.num_cols_to_keep + correction
    )
    assert X_result.shape[1] == expectation.num_cols_to_keep + correction


def test_decollinearizer_no_intercept_dependent(df_dependent_numeric):
    df, expectation = df_dependent_numeric
    decollinearizer = Decollinearizer(fit_intercept=False)
    df_result = decollinearizer.fit_transform(df)
    correction = 1 - expectation.intercept_not_collinear
    assert isinstance(df_result, pd.DataFrame)
    assert not decollinearizer.intercept_safe
    assert (
        len(decollinearizer.drop_columns) == expectation.num_cols_to_drop - correction
    )
    assert (
        len(decollinearizer.keep_columns) == expectation.num_cols_to_keep + correction
    )
    assert len(df_result.columns) == expectation.num_cols_to_keep + correction


def test_np_decollinearizer_no_intercept_dependent(
    df_dependent_numeric,
):
    df_independent, expectation = df_dependent_numeric
    X = df_independent.values
    decollinearizer = Decollinearizer(fit_intercept=False)
    X_result = decollinearizer.fit_transform(X)
    correction = 1 - expectation.intercept_not_collinear
    assert isinstance(X_result, np.ndarray)
    assert not decollinearizer.intercept_safe
    assert (
        len(decollinearizer.drop_columns) == expectation.num_cols_to_drop - correction
    )
    assert (
        len(decollinearizer.keep_columns) == expectation.num_cols_to_keep + correction
    )
    assert X_result.shape[1] == expectation.num_cols_to_keep + correction


def test_decollinearizer_wrong_type(df_dependent_numeric):
    df, _ = df_dependent_numeric
    decollinearizer = Decollinearizer(fit_intercept=False)
    decollinearizer.fit(df)
    with pytest.raises((ValueError, NotImplementedError)):
        decollinearizer.transform(df.to_numpy())
    with pytest.raises((ValueError, NotImplementedError)):
        decollinearizer.transform(tm.from_pandas(df))


def test_np_decollinearizer_wrong_type(df_dependent_numeric):
    df, _ = df_dependent_numeric
    decollinearizer = Decollinearizer(fit_intercept=False)
    X = df.to_numpy()
    decollinearizer.fit(X)
    with pytest.raises((ValueError, NotImplementedError)):
        decollinearizer.transform(df)
    with pytest.raises((ValueError, NotImplementedError)):
        decollinearizer.transform(tm.SplitMatrix(tm.from_pandas(df)))
    # note that DenseMatrix is also a numpy.ndarray:
    tm_dense_result = decollinearizer.transform(tm.from_pandas(df))
    assert isinstance(tm_dense_result, tm.DenseMatrix)


def test_decollinearizer_independent_categorical(df_independent_categorical):
    df, expectation = df_independent_categorical
    decollinearizer = Decollinearizer(fit_intercept=True)
    df_result = decollinearizer.fit_transform(df)
    assert isinstance(df_result, pd.DataFrame)
    assert decollinearizer.intercept_safe == expectation.intercept_not_collinear
    assert len(decollinearizer.drop_columns) == expectation.num_cols_to_drop
    assert len(decollinearizer.keep_columns) == expectation.num_cols_to_keep
    assert len(df_result.columns) == expectation.num_cols_to_keep
    assert len(decollinearizer.replace_categories) == expectation.num_categories_replace
    assert (
        len(pd.get_dummies(df_result, drop_first=True).columns)
        == expectation.design_matrix_rank_with_intercept - 1
    )


def test_np_decollinearizer_independent_categorical(df_independent_categorical):
    df, expectation = df_independent_categorical
    X = pd.get_dummies(df, drop_first=True).values
    decollinearizer = Decollinearizer(fit_intercept=True)
    X_result = decollinearizer.fit_transform(X)
    assert isinstance(X_result, np.ndarray)
    assert decollinearizer.intercept_safe == expectation.intercept_not_collinear
    assert (
        len(decollinearizer.drop_columns)
        == expectation.num_cols_to_drop + expectation.num_categories_replace
    )
    assert (
        len(decollinearizer.keep_columns)
        == expectation.design_matrix_rank_with_intercept - 1
    )
    assert X_result.shape[1] == expectation.design_matrix_rank_with_intercept - 1


def test_decollinearizer_dependent_categorical(df_dependent_categorical):
    df, expectation = df_dependent_categorical
    decollinearizer = Decollinearizer(fit_intercept=True)
    df_result = decollinearizer.fit_transform(df)
    assert isinstance(df_result, pd.DataFrame)
    assert decollinearizer.intercept_safe == expectation.intercept_not_collinear
    assert len(decollinearizer.drop_columns) == expectation.num_cols_to_drop
    assert len(decollinearizer.keep_columns) == expectation.num_cols_to_keep
    assert len(df_result.columns) == expectation.num_cols_to_keep
    assert len(decollinearizer.replace_categories) == expectation.num_categories_replace
    assert (
        len(pd.get_dummies(df_result, drop_first=True).columns)
        == expectation.design_matrix_rank_with_intercept - 1
    )


def test_np_decollinearizer_dependent_categorical(df_dependent_categorical):
    df, expectation = df_dependent_categorical
    X = pd.get_dummies(df, drop_first=True).values
    decollinearizer = Decollinearizer(fit_intercept=True)
    X_result = decollinearizer.fit_transform(X)
    assert isinstance(X_result, np.ndarray)
    assert decollinearizer.intercept_safe == expectation.intercept_not_collinear
    assert (
        len(decollinearizer.drop_columns)
        == expectation.num_cols_to_drop + expectation.num_categories_replace
    )
    assert (
        len(decollinearizer.keep_columns)
        == expectation.design_matrix_rank_with_intercept - 1
    )
    assert X_result.shape[1] == expectation.design_matrix_rank_with_intercept - 1


def test_intercept_not_dropped():
    X = pd.DataFrame({"a": [5, 5, 5, 5]}).to_numpy()
    decollinearizer = Decollinearizer(fit_intercept=True)
    decollinearizer.fit(X)
    assert decollinearizer.intercept_safe
