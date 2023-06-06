from typing import List, NamedTuple, Union

import numpy as np
import pandas as pd
import pytest
import tabmat as tm
from scipy import sparse

from glum import Decollinearizer
from glum._transformers import (
    CollinearityResults,
    _adjust_column_indices_for_intercept,
    _find_collinear_columns_from_gram,
    _find_intercept_alternative,
    _get_column_mapping,
    _get_gram_matrix_csc,
    _get_gram_matrix_numpy,
    _get_gram_matrix_tabmat,
    _safe_get_dummies,
)


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
    elif case == "wider_than_tall":
        return (
            pd.DataFrame(
                {
                    "a": [1.0, 0.0],
                    "b": [0.0, 1.0],
                    "c": [2.0, 3.0],
                }
            ),
            UnitTestExpectation(
                intercept_collinear=True, design_matrix_rank=2, cols_to_drop=1
            ),
        )
    elif case == "mixed_cat_num":
        return (
            pd.DataFrame(
                {
                    "a": [1.0, 2.0, 3.0, 4.0, 5.0],
                    "b": pd.Categorical(["a", "a", "a", "b", "c"]),
                    "c": [0, 0, 0, 2, 2],
                    "d": [1.0, 2.0, 4.0, 8.0, 16.0],
                }
            ),
            UnitTestExpectation(
                intercept_collinear=False, design_matrix_rank=4, cols_to_drop=1
            ),
        )
    else:
        raise ValueError(f"Invalid case: '{case}'")


SIMPLE_TEST_CASES = [
    "independent_numeric",
    "dependent_numeric",
    "dependent_on_combination_numeric",
    "dependent_on_intercept_numeric",
    "independent_categorical",
    "dependent_categorical",
    "wider_than_tall",
    "mixed_cat_num",
]


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
    assert len(_safe_get_dummies(df_result, drop_first=True).columns) == expected_rank


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
    SIMPLE_TEST_CASES,
)
@pytest.mark.parametrize(
    "fit_intercept", [True, False], ids=["intercept", "no_intercept"]
)
@pytest.mark.parametrize("use_tabmat", [True, False], ids=["tabmat", "no_tabmat"])
@pytest.mark.parametrize("format", ["pandas", "numpy", "csc"])
def test_against_expectation(
    simple_test_data, format: str, fit_intercept: bool, use_tabmat: bool
):
    df_input, expectation = simple_test_data

    if format == "pandas":
        decollinearizer_pd = Decollinearizer(fit_intercept=fit_intercept)
        pd_result = decollinearizer_pd.fit_transform(df_input, use_tabmat=use_tabmat)
        check_expectation_dataframe(decollinearizer_pd, pd_result, expectation)

    else:
        np_input = _safe_get_dummies(df_input, drop_first=True).to_numpy(
            dtype=np.float64
        )
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


@pytest.mark.parametrize(
    "case",
    SIMPLE_TEST_CASES,
)
@pytest.mark.parametrize(
    "fit_intercept", [True, False], ids=["intercept", "no_intercept"]
)
@pytest.mark.parametrize("format", ["pandas", "csc"])
def test_same_results_backend(simple_test_data, fit_intercept, format):
    df_input, _ = simple_test_data

    if format == "pandas":
        decollinearizer_tm = Decollinearizer(fit_intercept=fit_intercept)
        result_tm = decollinearizer_tm.fit_transform(df_input, use_tabmat=True)
        decollinearizer_np = Decollinearizer(fit_intercept=fit_intercept)
        result_np = decollinearizer_np.fit_transform(df_input, use_tabmat=False)

    elif format == "csc":
        np_input = _safe_get_dummies(df_input, drop_first=True).to_numpy(
            dtype=np.float64
        )
        csc_input = sparse.csc_matrix(np_input)
        decollinearizer_tm = Decollinearizer(fit_intercept=fit_intercept)
        result_tm = decollinearizer_tm.fit_transform(csc_input, use_tabmat=True)
        decollinearizer_np = Decollinearizer(fit_intercept=fit_intercept)
        result_np = decollinearizer_np.fit_transform(csc_input, use_tabmat=False)

    else:
        raise ValueError("This test is only defined for pandas and csc inputs")

    assert decollinearizer_tm.drop_columns == decollinearizer_np.drop_columns
    assert decollinearizer_tm.keep_columns == decollinearizer_np.keep_columns
    assert decollinearizer_tm.intercept_safe == decollinearizer_np.intercept_safe
    assert (
        decollinearizer_tm.replace_categories == decollinearizer_np.replace_categories
    )

    if format == "pandas":
        assert (result_tm == result_np).all(axis=None)
    elif format == "csc":
        # No .all() in scipy.sparse.csc_matrix
        assert not (result_tm != result_np).max()


@pytest.mark.parametrize(
    "case",
    SIMPLE_TEST_CASES,
)
@pytest.mark.parametrize(
    "fit_intercept", [True, False], ids=["intercept", "no_intercept"]
)
def test_same_results_format(simple_test_data, fit_intercept):
    df_input, _ = simple_test_data
    decollinearizer_pd = Decollinearizer(fit_intercept=fit_intercept)
    result_pd = decollinearizer_pd.fit_transform(df_input, use_tabmat=True)

    np_input = _safe_get_dummies(df_input, drop_first=True).to_numpy(dtype=np.float64)
    decollinearizer_np = Decollinearizer(fit_intercept=fit_intercept)
    result_np = decollinearizer_np.fit_transform(np_input, use_tabmat=False)

    csc_input = sparse.csc_matrix(np_input)
    decollinearizer_csc = Decollinearizer(fit_intercept=fit_intercept)
    result_csc = decollinearizer_csc.fit_transform(csc_input, use_tabmat=True)

    result_pd_matrix = _safe_get_dummies(result_pd, drop_first=True, dtype=np.float64)
    result_csc_matrix = result_csc.toarray()

    assert (result_np == result_pd_matrix).all(axis=None)
    assert (result_np == result_csc_matrix).all(axis=None)


@pytest.mark.parametrize(
    "gram, X1, results, valid",
    [
        (
            np.array([[2, 3], [3, 5]]),
            np.array([5, 8]),
            CollinearityResults(np.array([1, 2]), np.array([0])),
            [1],
        ),
        (
            np.array([[1, 0, 1], [0, 1, 2], [0, 1, 3]]),
            np.array([1, 2, 6]),
            CollinearityResults(np.array([1, 2, 3]), np.array([0])),
            [1, 2],
        ),
        (
            np.array([[2, 3], [3, 5]]),
            np.array([5, 8]),
            CollinearityResults(np.array([3, 5]), np.array([0])),
            [3, 5],
        ),
    ],
    ids=["one_choice", "multiple_choices", "weird_col_nums"],
)
def test_find_intercept_alternative(gram, X1, results, valid):
    new_results = _find_intercept_alternative(gram, X1, results)
    orig_keep_set = set(results.keep_idx)
    orig_drop_set = set(results.drop_idx)
    new_keep_set = set(new_results.keep_idx)
    new_drop_set = set(new_results.drop_idx)
    valid_set = set(valid)
    assert len(new_drop_set & valid_set) == 1
    assert len(new_keep_set & valid_set) == len(valid_set) - 1
    assert len(orig_keep_set & new_keep_set) == len(orig_keep_set) - 1
    assert len(orig_drop_set & new_drop_set) == len(orig_drop_set) - 1


def get_gram_matrix_reference(X: np.ndarray, fit_intercept: bool):
    if fit_intercept:
        X = np.hstack([np.ones((X.shape[0], 1)), X])
    return X.T @ X


@pytest.fixture
def random_matrix(m: int, n: int):
    np.random.seed(42)
    return np.random.randn(m, n)


@pytest.mark.parametrize(
    "m, n",
    [
        (10, 5),
        (10, 10),
        (1000, 5),
        (5, 20),
    ],
)
@pytest.mark.parametrize(
    "fit_intercept", [True, False], ids=["intercept", "no_intercept"]
)
@pytest.mark.parametrize("function", ["tabmat", "numpy", "csc"])
def test_gram_matrix(random_matrix, function, fit_intercept):
    X = random_matrix
    reference_gram = get_gram_matrix_reference(X, fit_intercept)
    if function == "tabmat":
        X_tm = tm.SplitMatrix([tm.DenseMatrix(X)])
        gram = _get_gram_matrix_tabmat(X_tm, fit_intercept)
    elif function == "numpy":
        gram = _get_gram_matrix_numpy(X, fit_intercept)
    elif function == "csc":
        csc_input = sparse.csc_matrix(X)
        gram = _get_gram_matrix_csc(csc_input, fit_intercept)
    assert np.allclose(gram, reference_gram)


@pytest.mark.parametrize(
    "case",
    SIMPLE_TEST_CASES,
)
@pytest.mark.parametrize(
    "fit_intercept", [True, False], ids=["intercept", "no_intercept"]
)
def test_find_collinear_columns_from_gram(simple_test_data, fit_intercept):
    df_input, expectation = simple_test_data
    X = _safe_get_dummies(df_input, drop_first=True).to_numpy(dtype=np.float64)
    gram = _get_gram_matrix_numpy(X, fit_intercept)
    results = _find_collinear_columns_from_gram(gram, fit_intercept)

    if fit_intercept:
        if not expectation.intercept_collinear:
            expected_rank = expectation.design_matrix_rank + 1
            cols_to_drop = expectation.cols_to_drop
        else:
            expected_rank = expectation.design_matrix_rank
            cols_to_drop = expectation.cols_to_drop + 1
    else:
        expected_rank = expectation.design_matrix_rank
        cols_to_drop = expectation.cols_to_drop

    assert len(results.keep_idx) == expected_rank
    assert len(results.drop_idx) == cols_to_drop


@pytest.mark.parametrize(
    "results",
    [
        CollinearityResults(np.array([0]), np.array([1, 2])),
        CollinearityResults(np.array([0, 1, 4]), np.array([2, 3])),
        CollinearityResults(np.array([0, 1, 4]), np.array([])),
    ],
    ids=["intercept_only", "usual_case", "no_drop"],
)
def test_adjust_column_indices(results):
    new_results = _adjust_column_indices_for_intercept(results)
    assert (new_results.drop_idx == results.drop_idx - 1).all()
    keep_idx_wo_intercept = results.keep_idx[results.keep_idx != 0]
    assert (new_results.keep_idx == keep_idx_wo_intercept - 1).all()


@pytest.mark.parametrize(
    "results",
    [
        CollinearityResults(np.array([1, 2]), np.array([0, 3])),
        CollinearityResults(np.array([1, 2]), np.array([3])),
    ],
    ids=["intercept_in_drop", "intercept_not_in_keep"],
)
def test_adjust_column_indices_wrong_input(results):
    with pytest.raises(ValueError):
        _adjust_column_indices_for_intercept(results)


@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame({"x": pd.Categorical([])}),
        pd.DataFrame(
            {
                "x": [1.0, 2.0, 3.0],
                "y": pd.Categorical(["a", "b", "c"]),
                "z": [1.0, 2.0, 3.0],
            }
        ),
        pd.DataFrame(
            {
                "x": [1.0, 2.0, 3.0],
                "y": pd.Categorical(["a", "b", "c"], categories=["a", "b", "c", "d"]),
                "z": [1.0, 2.0, 3.0],
            }
        ),
    ],
    ids=["empty", "mixed", "unused_category"],
)
def test_get_column_mapping(df: pd.DataFrame):
    column_mapping = _get_column_mapping(df)
    df_expanded = _safe_get_dummies(df, drop_first=True)
    assert len(column_mapping) == df_expanded.shape[1]
    for expanded_col_pos, (
        col_pos,
        col_name,
        is_categorical,
        category,
        base_category,
    ) in enumerate(column_mapping):
        assert col_name in df.columns
        assert df.iloc[:, col_pos].name == col_name
        assert is_categorical == pd.api.types.is_categorical_dtype(df[col_name])
        if is_categorical:
            assert category in df[col_name].cat.categories.tolist()
            assert base_category == df[col_name].cat.categories.tolist()[0]
            assert (
                df_expanded.iloc[:, expanded_col_pos].name == f"{col_name}_{category}"
            )


@pytest.mark.parametrize(
    "df, expected_columns",
    [
        (pd.DataFrame({"x": pd.Categorical([])}), []),
        (
            pd.DataFrame(
                {
                    "x": [1.0, 2.0, 3.0],
                    "y": pd.Categorical(["a", "b", "c"]),
                    "z": [1.0, 2.0, 3.0],
                },
            ),
            ["x", "y_b", "y_c", "z"],
        ),
        (
            pd.DataFrame(
                {
                    "x": [1.0, 2.0, 3.0],
                    "y": pd.Categorical(
                        ["a", "b", "c"], categories=["a", "b", "c", "d"]
                    ),
                    "z": [1.0, 2.0, 3.0],
                },
            ),
            ["x", "y_b", "y_c", "y_d", "z"],
        ),
    ],
    ids=["empty", "mixed", "unused_category"],
)
def test_safe_get_dummies(df: pd.DataFrame, expected_columns: List[str]):
    df_expanded = _safe_get_dummies(df, drop_first=True)
    df_expanded_pandas = pd.get_dummies(df, drop_first=True)
    assert len(df_expanded.columns) == len(df_expanded_pandas.columns)
    assert df_expanded.columns.tolist() == expected_columns
