from typing import NamedTuple, Union

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from glum import Decollinearizer


class UnitTestExpectation(NamedTuple):
    """Expectation for a Pandas test."""

    intercept_collinear: bool
    design_matrix_rank: int
    cols_to_drop: int


@pytest.fixture
def simple_test_data(case: str):
    if case == "independent_numeric":
        return (
            pd.DataFrame(
                {
                    "a": [1.0, 0.0, 0.0],
                    "b": [0.0, 1.0, 0.0],
                }
            ),
            UnitTestExpectation(
                intercept_collinear=False, design_matrix_rank=2, cols_to_drop=0
            ),
        )
    elif case == "dependent_numeric":
        return (
            pd.DataFrame(
                {
                    "a": [1.0, 0.0, 0.0],
                    "b": [2.0, 0.0, 0.0],
                }
            ),
            UnitTestExpectation(
                intercept_collinear=False, design_matrix_rank=1, cols_to_drop=1
            ),
        )
    elif case == "dependent_on_combination_numeric":
        return (
            pd.DataFrame(
                {
                    "a": [1.0, 0.0, 0.0, 0.0],
                    "b": [0.0, 1.0, 0.0, 0.0],
                    "c": [2.0, 2.0, 0.0, 0.0],
                }
            ),
            UnitTestExpectation(
                intercept_collinear=False, design_matrix_rank=2, cols_to_drop=1
            ),
        )
    elif case == "dependent_on_intercept_numeric":
        return (
            pd.DataFrame(
                {
                    "a": [1.0, 0.0, 0.0, 0.0],
                    "b": [0.3, 0.3, 0.3, 0.3],
                    "c": [0.0, 1.0, 0.0, 0.0],
                }
            ),
            UnitTestExpectation(
                intercept_collinear=True, design_matrix_rank=3, cols_to_drop=0
            ),
        )
    elif case == "independent_categorical":
        return (
            pd.DataFrame(
                {
                    "a": pd.Categorical(["a", "b", "b"]),
                    "b": pd.Categorical(["a", "a", "b"]),
                }
            ),
            UnitTestExpectation(
                intercept_collinear=False, design_matrix_rank=2, cols_to_drop=0
            ),
        )
    elif case == "dependent_categorical":
        return (
            pd.DataFrame(
                {
                    "a": pd.Categorical(["a", "b", "c"]),
                    "b": pd.Categorical(["a", "a", "c"]),
                }
            ),
            UnitTestExpectation(
                intercept_collinear=False, design_matrix_rank=2, cols_to_drop=1
            ),
        )
    else:
        raise ValueError(f"Invalid case: '{case}'")


def check_expectation_dataframe(
    decollinearizer: Decollinearizer,
    df_result: pd.DataFrame,
    expectation: UnitTestExpectation,
):
    num_collinear = len(decollinearizer.drop_columns) + len(
        decollinearizer.replace_categories
    )
    if decollinearizer.fit_intercept and expectation.intercept_collinear:
        expected_rank = expectation.design_matrix_rank - 1
        cols_to_drop = expectation.cols_to_drop + 1
    else:
        expected_rank = expectation.design_matrix_rank
        cols_to_drop = expectation.cols_to_drop

    assert isinstance(df_result, pd.DataFrame)
    assert decollinearizer.intercept_safe == decollinearizer.fit_intercept
    assert num_collinear == cols_to_drop
    assert len(pd.get_dummies(df_result, drop_first=True).columns) == expected_rank


def check_expectation_array(
    decollinearizer: Decollinearizer,
    X_result: Union[np.ndarray, sparse.csc_matrix],
    expectation: UnitTestExpectation,
    format: str,
):
    if decollinearizer.fit_intercept and expectation.intercept_collinear:
        expected_rank = expectation.design_matrix_rank - 1
        cols_to_drop = expectation.cols_to_drop + 1
    else:
        expected_rank = expectation.design_matrix_rank
        cols_to_drop = expectation.cols_to_drop
    if format == "numpy":
        assert isinstance(X_result, np.ndarray)
    elif format == "csc":
        assert isinstance(X_result, sparse.csc_matrix)
    assert decollinearizer.intercept_safe == decollinearizer.fit_intercept
    assert len(decollinearizer.drop_columns) == cols_to_drop
    assert decollinearizer.replace_categories == []
    assert X_result.shape[1] == expected_rank


@pytest.mark.parametrize(
    "case",
    [
        "independent_numeric",
        "dependent_numeric",
        "dependent_on_combination_numeric",
        "dependent_on_intercept_numeric",
        "independent_categorical",
        "dependent_categorical",
    ],
)
@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("use_tabmat", [True, False])
@pytest.mark.parametrize("format", ["pandas", "numpy", "csc"])
def test_against_expectation(
    simple_test_data, format: str, fit_intercept: bool, use_tabmat: bool
):
    df_input, expectation = simple_test_data

    decollinearizer_pd = Decollinearizer(fit_intercept=fit_intercept)
    if format == "pandas":
        pd_result = decollinearizer_pd.fit_transform(df_input, use_tabmat=use_tabmat)
        check_expectation_dataframe(decollinearizer_pd, pd_result, expectation)

    else:
        np_input = pd.get_dummies(df_input, drop_first=True).to_numpy(dtype=np.float64)
        if format == "numpy":
            decollinearizer_np = Decollinearizer(fit_intercept=True)
            if not use_tabmat:
                np_result = decollinearizer_np.fit_transform(
                    np_input, use_tabmat=use_tabmat
                )
                check_expectation_array(
                    decollinearizer_np, np_result, expectation, format
                )
            if use_tabmat:
                with pytest.raises(ValueError):
                    np_result = decollinearizer_np.fit_transform(
                        np_input, use_tabmat=use_tabmat
                    )
        else:
            csc_input = sparse.csc_matrix(np_input)
            decollinearizer_csc = Decollinearizer(fit_intercept=fit_intercept)
            csc_result = decollinearizer_csc.fit_transform(
                csc_input, use_tabmat=use_tabmat
            )
            check_expectation_array(
                decollinearizer_csc, csc_result, expectation, format
            )
