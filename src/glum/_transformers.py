"""Some utilities for transforming data before fitting a model."""

from typing import Any, Hashable, List, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import tabmat as tm
from scipy import sparse
from scipy.linalg import qr, solve
from sklearn.base import BaseEstimator, TransformerMixin, check_is_fitted

DataLike = Union[np.ndarray, sparse.csc_matrix, pd.DataFrame]


def _safe_sandwich_dot(
    X: Union[tm.MatrixBase, tm.StandardizedMatrix],
    d: np.ndarray,
    rows: np.ndarray = None,
    cols: np.ndarray = None,
    intercept=False,
) -> np.ndarray:
    """
    Compute sandwich product ``X.T @ diag(d) @ X``.

    With ``intercept=True``, ``X`` is treated as if a column of 1 were appended
    as first column of ``X``. ``X`` can be sparse; ``d`` must be an ndarray.
    Always returns an ndarray.

    Source: glum/_util.py
    """
    result = X.sandwich(d, rows, cols)
    if isinstance(result, sparse.dia_matrix):
        result = result.A

    if intercept:
        dim = result.shape[0] + 1
        res_including_intercept = np.empty((dim, dim), dtype=X.dtype)
        res_including_intercept[0, 0] = d.sum()
        res_including_intercept[1:, 0] = X.transpose_matvec(d, rows, cols)
        res_including_intercept[0, 1:] = res_including_intercept[1:, 0]
        res_including_intercept[1:, 1:] = result
    else:
        res_including_intercept = result
    return res_including_intercept


def _safe_get_dummies(
    data: Union[pd.DataFrame, pd.Series], *args, **kwargs
) -> pd.DataFrame:
    """`pd.get_dummies`, but preserve column order of the original dataframe."""
    if isinstance(data, pd.DataFrame):
        dtypes_to_encode = ["object", "string", "category"]
        cols_to_encode = data.select_dtypes(include=dtypes_to_encode).columns
        new_data_chunks = []
        for col in data.columns:
            if col in cols_to_encode:
                new_data_chunks.append(pd.get_dummies(data[[col]], *args, **kwargs))
            else:
                new_data_chunks.append(data[col])
        return pd.concat(new_data_chunks, axis=1)
    else:
        return pd.get_dummies(data, *args, **kwargs)


class CollinearityResults(NamedTuple):
    """Results of collinearity analysis."""

    keep_idx: np.ndarray
    drop_idx: np.ndarray


def _get_gram_matrix_tabmat(X: tm.MatrixBase, fit_intercept: bool = True) -> np.ndarray:
    return _safe_sandwich_dot(X, np.ones(X.shape[0]), intercept=fit_intercept)


def _get_gram_matrix_numpy(X: np.ndarray, fit_intercept: bool = True) -> np.ndarray:
    gram = X.T @ X
    if fit_intercept:
        corner = X.shape[0]
        sides = X.sum(axis=0, keepdims=True)
        gram = np.block(
            [
                [corner, sides],
                [sides.T, gram],
            ]
        )
    return gram


def _get_gram_matrix_csc(X: sparse.csc_matrix, fit_intercept: bool = True):
    gram = (X.T @ X).todense()
    if fit_intercept:
        corner = X.shape[0]
        sides = X.sum(axis=0)
        gram = np.block(
            [
                [corner, sides],
                [sides.T, gram],
            ]
        )
    return gram


def _find_collinear_columns_from_gram(
    gram: np.ndarray, fit_intercept: bool = True, tolerance: float = 1e-6
) -> CollinearityResults:
    """Find the collinear columns in a numpy array.
    Return all information needed to decollinearize the array.

    Parameters
    ----------
    gram: np.ndarray
        X.T @ X, where X is the design matrix.
    fit_intercept : bool, optional
        Whether an intercept was added to the design matrix when computing the gram matrix.
        The intercept is assumed to be the first column.
    tolerance : float, optional

    Returns
    -------
    CollinearityResults
        Information about the columns to keep and the columns to drop
    """
    R, P = qr(gram, mode="r", pivoting=True)  # type: ignore
    permuted_keep_mask = np.abs(np.diag(R)) > tolerance
    # More columns than rows case:
    if R.shape[1] > R.shape[0]:
        permuted_keep_mask = np.concatenate(
            (permuted_keep_mask, np.zeros(R.shape[1] - R.shape[0], dtype=bool))
        )
    keep_mask = np.empty_like(permuted_keep_mask)
    keep_mask[P] = permuted_keep_mask

    keep_idx = np.where(keep_mask)[0]
    drop_idx = np.where(~keep_mask)[0]

    return CollinearityResults(keep_idx, drop_idx)


def _find_intercept_alternative(
    gram: np.ndarray, X1: np.ndarray, results: CollinearityResults
) -> CollinearityResults:
    """Assuming that the intercept is among the columns to drop, find an alternative

    Parameters
    ----------
    gram: np.ndarray
        X.T @ X where X are the independent columns of the design matrix
    X1: np.ndarray
        X'.T @ 1 where X are the independent columns of the design matrix
    results: CollinearityResults
        The results of the collinearity analysis

    Returns
    -------
    CollinearityResults
        Alternative collinearity results with the intercept not dropped
    """
    keep_idx = results.keep_idx
    drop_idx = results.drop_idx

    # We know that the restricted gram matrix is non-singular
    lin_comb_for_intercept = solve(gram, X1)
    # Find the column that is has a large weight in the linear combination
    drop_instead_of_intercept = keep_idx[np.argmax(np.abs(lin_comb_for_intercept))]
    # Update the keep and drop indices
    keep_idx = keep_idx[keep_idx != drop_instead_of_intercept]
    keep_idx = np.insert(keep_idx, 0, 0)
    drop_idx = drop_idx[drop_idx != 0]
    drop_insert_idx = np.searchsorted(drop_idx, drop_instead_of_intercept)
    drop_idx = np.insert(drop_idx, drop_insert_idx, drop_instead_of_intercept)

    return CollinearityResults(keep_idx, drop_idx)


def _adjust_column_indices_for_intercept(
    results: CollinearityResults,
) -> CollinearityResults:
    """Adjust the column indices such that index 0 corresponds to the first
    non-intercept column.
    """
    if 0 in results.drop_idx or 0 not in results.keep_idx:
        raise ValueError("The intercept should be among the columns to keep.")
    keep_idx = results.keep_idx - 1
    drop_idx = results.drop_idx - 1
    keep_idx = keep_idx[keep_idx != -1]

    return CollinearityResults(keep_idx, drop_idx)


class ColumnMap(NamedTuple):
    """Mapping from DataFrame to design matrix."""

    column_pos: int
    column_name: Hashable
    categorical: bool
    category: Optional[str] = None
    base_category: Optional[str] = None


def _get_column_mapping(X: pd.DataFrame) -> Sequence[ColumnMap]:
    column_mapping = []
    for column_pos, (column_name, dtype) in enumerate(X.dtypes.items()):
        if isinstance(dtype, pd.CategoricalDtype):
            if len(dtype.categories) > 1:
                base_category = dtype.categories[0]
                for category in dtype.categories[1:]:
                    column_mapping.append(
                        ColumnMap(
                            column_pos, column_name, True, category, base_category
                        )
                    )
        else:
            column_mapping.append(ColumnMap(column_pos, column_name, False))
    return column_mapping


class Decollinearizer(TransformerMixin, BaseEstimator):
    """Drop collinear columns from the design matrix implied by a dataset.

    The type of the output is the same as the input. For non-categorical
    columns, collinear ones are simply dropped. For categorical columns
    (e.g. in a pandas.DataFrame or a tabmat.SplitMatrix), values whose
    columns in the design matrix would be dropped are replaced with the
    first category. This supposes that the first category is the reference,
    and will be dropped in the subsequent model fitting step.
    """

    drop_columns: Union[Sequence[Hashable], Sequence[int]]
    keep_columns: Union[Sequence[Hashable], Sequence[int]]
    intercept_safe: bool
    input_type: str
    replace_categories: List[Tuple[Hashable, str, str]]

    def __init__(self, fit_intercept: bool = True, tolerance: float = 1e-6) -> None:
        self.fit_intercept = fit_intercept
        self.tolerance = tolerance

    def fit(
        self,
        X: DataLike,
        y: Optional[Any] = None,
        use_tabmat: Optional[bool] = None,
    ) -> "Decollinearizer":
        """Fit the transformer by finding a maximal set of linearly independent columns.

        Parameters
        ----------
        X : ArrayLike
            The data to fit. Can be a pandas.DataFrame
            a tabmat.SplitMatrix, or a numpy.ndarray.
        y: Optional[Any]
            Ignored. Present for API consistency.

        Returns
        -------
        Self
            The fitted transformer.
        """
        if isinstance(X, pd.DataFrame):
            self._fit_pandas(X, use_tabmat=use_tabmat)
        elif isinstance(X, np.ndarray):
            self._fit_numpy(X, use_tabmat=use_tabmat)
        elif isinstance(X, sparse.csc_matrix):
            self._fit_csc(X, use_tabmat=use_tabmat)
        else:
            raise ValueError(
                "X must be a pandas.DataFrame, a numpy.ndarray or a scipy.sparse.csc_matrix."
                f"Got {type(X)} instead."
            )
        return self

    def _fit_pandas(
        self,
        df: pd.DataFrame,
        use_tabmat: Optional[bool] = None,
    ) -> None:
        """Fit the transformer on a pandas.DataFrame."""
        if use_tabmat or use_tabmat is None:
            # TODO: checks before conversion
            X_tm = tm.from_pandas(df, drop_first=True)
            gram = _get_gram_matrix_tabmat(X_tm, fit_intercept=self.fit_intercept)
        else:
            # TODO: make sure that object columns are handled the same in all modes
            X_np = _safe_get_dummies(df, drop_first=True).to_numpy(dtype=np.float_)
            gram = _get_gram_matrix_numpy(X_np, fit_intercept=self.fit_intercept)

        results = _find_collinear_columns_from_gram(
            gram, self.fit_intercept, self.tolerance
        )
        if self.fit_intercept and 0 not in results.keep_idx:
            keep_idx_wo_intercept = results.keep_idx - 1
            if use_tabmat:
                X1 = X_tm.matvec(np.ones(X_tm.shape[1]), cols=keep_idx_wo_intercept)
            else:
                X1 = X_np[:, keep_idx_wo_intercept].sum(axis=0)
            results = _find_intercept_alternative(
                gram[np.ix_(results.keep_idx, results.keep_idx)], X1, results
            )

        if self.fit_intercept:
            results = _adjust_column_indices_for_intercept(results)

        # Convert design matrix indices to column names and category replacements
        self.column_mapping = _get_column_mapping(df)
        drop_columns = []
        replace_categories = []
        for col_idx in results.drop_idx:
            col_name = self.column_mapping[col_idx].column_name
            if not self.column_mapping[col_idx].categorical:
                drop_columns.append(col_name)
            else:
                column_map = self.column_mapping[col_idx]
                replace_categories.append(
                    (col_name, column_map.category, column_map.base_category)
                )

        self.drop_columns = list(drop_columns)
        self.keep_columns = list(df.columns.difference(drop_columns))
        self.replace_categories = replace_categories  # type: ignore
        self.intercept_safe = self.fit_intercept  # We never drop the intercept
        self.input_type = "pandas"

    def _fit_numpy(
        self,
        X: np.ndarray,
        use_tabmat: Optional[bool] = None,
    ) -> None:
        """Fit the transformer on a numpy.ndarray."""
        if use_tabmat:
            raise ValueError("use_tabmat=True is not supported for numpy arrays")
        if X.dtype != np.float_:
            raise ValueError("X must be a float array.")

        gram = _get_gram_matrix_numpy(X, fit_intercept=self.fit_intercept)
        results = _find_collinear_columns_from_gram(
            gram, self.fit_intercept, self.tolerance
        )
        if self.fit_intercept and 0 not in results.keep_idx:
            keep_idx_wo_intercept = results.keep_idx - 1
            X1 = X[:, keep_idx_wo_intercept].sum(axis=1)
            results = _find_intercept_alternative(
                gram[np.ix_(results.keep_idx, results.keep_idx)], X1, results
            )

        if self.fit_intercept:
            results = _adjust_column_indices_for_intercept(results)

        self.drop_columns = list(results.drop_idx)
        self.keep_columns = list(results.keep_idx)
        self.intercept_safe = self.fit_intercept
        self.replace_categories = []
        self.input_type = "numpy"

    def _fit_csc(
        self,
        X: sparse.csc_matrix,
        use_tabmat: Optional[bool] = None,
    ) -> None:
        """Fit the transformer on a scipy.sparse.csc_matrix."""
        if X.dtype != np.float_:
            raise ValueError("X must be a float array.")

        if use_tabmat or use_tabmat is None:
            # TODO: checks before conversion?
            X_tm = tm.from_csc(X)
            gram = _get_gram_matrix_tabmat(X_tm, fit_intercept=self.fit_intercept)
        else:
            gram = _get_gram_matrix_csc(X, fit_intercept=self.fit_intercept)

        results = _find_collinear_columns_from_gram(
            gram, self.fit_intercept, self.tolerance
        )
        if self.fit_intercept and 0 not in results.keep_idx:
            keep_idx_wo_intercept = results.keep_idx - 1
            if use_tabmat:
                X1 = X_tm.matvec(np.ones(X_tm.shape[1]), cols=keep_idx_wo_intercept)
            else:
                X1 = X[:, keep_idx_wo_intercept].sum(axis=1)
            results = _find_intercept_alternative(
                gram[np.ix_(results.keep_idx, results.keep_idx)], X1, results
            )

        if self.fit_intercept:
            results = _adjust_column_indices_for_intercept(results)

        self.drop_columns = list(results.drop_idx)
        self.keep_columns = list(results.keep_idx)
        self.replace_categories = []  # type: ignore
        self.intercept_safe = self.fit_intercept  # We never drop the intercept
        self.input_type = "csc"

    def transform(self, X: DataLike, y: Optional[Any] = None) -> DataLike:
        """Transform the data by dropping collinear columns.

        Parameters
        ----------
        X : ArrayLike
            The data to transform. Can be a pandas.DataFrame
            a tabmat.SplitMatrix, or a numpy.ndarray.
        y: Optional[Any]
            Ignored. Present for API consistency.

        Returns
        -------
        ArrayLike
            The transformed data, in the same format as the input.
        """
        if isinstance(X, pd.DataFrame):
            return self._transform_pandas(X)
        elif isinstance(X, np.ndarray):
            return self._transform_numpy(X)
        elif isinstance(X, sparse.csc_matrix):
            return self._transform_csc(X)
        else:
            raise ValueError(
                "X must be a pandas.DataFrame, numpy.ndarray or scipy.sparse.csc_matrix. "
                f"Got {type(X)}."
            )

    def _transform_pandas(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the transformer to a fitted pandas.DataFrame."""
        check_is_fitted(self, ["input_type"])
        if self.input_type != "pandas":
            raise ValueError(  # Should it be a TypeError?
                "The transformer was fitted on a pandas.DataFrame, "
                "but is being asked to transform a {}".format(type(X))
            )

        X = X.copy().drop(columns=self.drop_columns)
        for col_name, category, base_category in self.replace_categories:
            X.loc[
                lambda df: df.loc[:, col_name] == category,  # noqa: B023
                col_name,
            ] = base_category
        for column in X.select_dtypes(include="category"):
            X[column] = X[column].cat.remove_unused_categories()

        return X

    def _transform_numpy(self, X: np.ndarray) -> np.ndarray:
        """Apply the transformer to a fitted numpy.ndarray."""
        check_is_fitted(self, ["input_type"])
        if self.input_type != "numpy":
            raise ValueError(  # Should it be a TypeError?
                "The transformer was fitted on a numpy.ndarray, "
                "but is being asked to transform a {}".format(type(X))
            )

        return np.delete(X, self.drop_columns, axis=1)

    def _transform_csc(self, X: np.ndarray) -> np.ndarray:
        """Apply the transformer to a fitted scipy.sparse.csc_matrix."""
        check_is_fitted(self, ["input_type"])
        if self.input_type != "csc":
            raise ValueError(  # Should it be a TypeError?
                "The transformer was fitted on a scipy.sparse.csc_matrix, "
                "but is being asked to transform a {}".format(type(X))
            )

        col_mask = np.ones(X.shape[1], dtype="bool")
        col_mask[self.drop_columns] = False
        return X[:, col_mask]
