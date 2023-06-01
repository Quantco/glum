import pandas as pd
import pytest
import tabmat as tm

from glum import Decollinearizer


@pytest.fixture()
def df_independent_numeric():
    return (
        pd.DataFrame(
            {
                "a": [1.0, 0.0, 0.0],
                "b": [0.0, 1.0, 0.0],
            }
        ),
        2,
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
        1,
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
        2,
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
        2,
    )


@pytest.fixture()
def df_independent_categorical():
    return (
        pd.DataFrame(
            {
                "a": pd.CategoricalDtype(["a", "b", "b"]),
                "b": pd.Categorical(["a", "a", "b"]),
            }
        ),
        [],
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
        [],
    )


def test_decollinearizer_independent_numeric(df_independent_numeric):
    df_independent, rank = df_independent_numeric
    decollinearizer = Decollinearizer(fit_intercept=True)
    df_result = decollinearizer.fit_transform(df_independent)
    assert decollinearizer.intercept_safe
    assert len(decollinearizer.drop_columns) == 0
    assert len(df_result.columns) == rank


def test_decollinearizer_dependent_numeric(df_dependent_numeric):
    df, rank = df_dependent_numeric
    decollinearizer = Decollinearizer(fit_intercept=True)
    df_result = decollinearizer.fit_transform(df)
    assert decollinearizer.intercept_safe
    assert len(decollinearizer.drop_columns) == 1
    assert len(df_result.columns) == rank


def test_decollinearizer_dependent_on_combination_numeric(
    df_dependent_on_combination_numeric,
):
    df, rank = df_dependent_on_combination_numeric
    decollinearizer = Decollinearizer(fit_intercept=True)
    df_result = decollinearizer.fit_transform(df)
    assert decollinearizer.intercept_safe
    assert len(decollinearizer.drop_columns) == 1
    assert len(df_result.columns) == rank


def test_decollinearizer_dependent_on_intercept_numeric(
    df_dependent_on_intercept_numeric,
):
    df, rank = df_dependent_on_intercept_numeric
    decollinearizer = Decollinearizer(fit_intercept=True)
    df_result = decollinearizer.fit_transform(df)
    assert (
        decollinearizer.intercept_safe and len(decollinearizer.drop_columns) == 1
    ) or (not decollinearizer.intercept_safe and len(decollinearizer.drop_columns) == 0)
    total_rank = decollinearizer.intercept_safe + len(df_result.columns)
    assert total_rank == rank + 1


def test_decollinearizer_no_intercept_independent(df_dependent_on_intercept_numeric):
    df, rank = df_dependent_on_intercept_numeric
    decollinearizer = Decollinearizer(fit_intercept=False)
    df_result = decollinearizer.fit_transform(df)
    assert not decollinearizer.intercept_safe
    assert len(decollinearizer.drop_columns) == 0
    assert len(df_result.columns) == rank + 1


def test_decollinearizer_no_intercept_dependent(df_dependent_numeric):
    df, rank = df_dependent_numeric
    decollinearizer = Decollinearizer(fit_intercept=False)
    df_result = decollinearizer.fit_transform(df)
    assert not decollinearizer.intercept_safe
    assert len(decollinearizer.drop_columns) == 1
    assert len(df_result.columns) == rank


def test_decollinearizer_wrong_type(df_dependent_numeric):
    df, _ = df_dependent_numeric
    decollinearizer = Decollinearizer(fit_intercept=False)
    decollinearizer.fit(df)
    with pytest.raises((ValueError, NotImplementedError)):
        decollinearizer.transform(df.to_numpy())
    with pytest.raises((ValueError, NotImplementedError)):
        decollinearizer.transform(tm.from_pandas(df))
