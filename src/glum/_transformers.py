"""Some utilities for transforming data before fitting a model."""

from typing import Hashable, List, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import tabmat as tm
from scipy import sparse
from scipy.linalg import qr, solve
from sklearn.base import BaseEstimator, TransformerMixin, check_is_fitted

from ._glm import ArrayLike, VectorLike
from ._util import _safe_sandwich_dot


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
    keep_idx: Sequence[int]
        The indices of the kept columns in the original design matrix
    drop_idx: Sequence[int]
        The indices of the dropped columns in the original design matrix
    """
    keep_idx = results.keep_idx
    drop_idx = results.drop_idx

    # We know that the restricted gram matrix is non-singular
    lin_comb_for_intercept = solve(gram, X1)
    # Find the column that is has a large weight in the linear combination
    drop_instead_of_intercept = keep_idx[np.argmax(np.abs(lin_comb_for_intercept))]
    # Update the keep and drop indices
    keep_idx = keep_idx[keep_idx != drop_instead_of_intercept]
    drop_insert_idx = np.searchsorted(drop_idx, drop_instead_of_intercept)
    drop_idx = np.insert(drop_idx, drop_insert_idx, drop_instead_of_intercept)

    return CollinearityResults(keep_idx, drop_idx)


def _adjust_column_indices_for_intercept(
    results: CollinearityResults,
) -> CollinearityResults:
    """Adjust the column indices such that index 0 corresponds to the first
    non-intercept column.
    """
    keep_idx = results.keep_idx - 1
    drop_idx = results.drop_idx - 1
    keep_idx = keep_idx[keep_idx != -1]
    drop_idx = drop_idx[drop_idx != -1]

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
        X: ArrayLike,
        y: Optional[VectorLike] = None,
        use_tabmat: Optional[bool] = None,
    ) -> "Decollinearizer":
        """Fit the transformer by finding a maximal set of linearly independent columns.

        Parameters
        ----------
        X : ArrayLike
            The data to fit. Can be a pandas.DataFrame
            a tabmat.SplitMatrix, or a numpy.ndarray.
        y: Optional[VectorLike]
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
            X_np = pd.get_dummies(df, drop_first=True).to_numpy()
            gram = _get_gram_matrix_numpy(X_np, fit_intercept=self.fit_intercept)

        results = _find_collinear_columns_from_gram(
            gram, self.fit_intercept, self.tolerance
        )
        if self.fit_intercept and 0 not in results.keep_idx:
            keep_idx_wo_intercept = results.keep_idx - 1
            if use_tabmat:
                X1 = X_tm.matvec(np.ones(df.shape[0]), cols=keep_idx_wo_intercept)
            else:
                X1 = X_np[:, keep_idx_wo_intercept].sum(axis=0)
            results = _find_intercept_alternative(
                gram[results.keep_idx, results.keep_idx], X1, results
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

        self.drop_columns = drop_columns
        self.keep_columns = df.columns.difference(drop_columns)
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

        gram = _get_gram_matrix_numpy(X, fit_intercept=self.fit_intercept)
        results = _find_collinear_columns_from_gram(
            gram, self.fit_intercept, self.tolerance
        )
        if self.fit_intercept and 0 not in results.keep_idx:
            keep_idx_wo_intercept = results.keep_idx - 1
            X1 = X[:, keep_idx_wo_intercept].sum(axis=0)
            results = _find_intercept_alternative(
                gram[results.keep_idx, results.keep_idx], X1, results
            )

        if self.fit_intercept:
            results = _adjust_column_indices_for_intercept(results)

        self.drop_columns = results.drop_idx
        self.keep_columns = results.keep_idx
        self.intercept_safe = self.fit_intercept
        self.replace_categories = []
        self.input_type = "numpy"

    def _fit_csc(
        self,
        X: sparse.csc_matrix,
        use_tabmat: Optional[bool] = None,
    ) -> None:
        """Fit the transformer on a scipy.sparse.csc_matrix."""
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
                X1 = X_tm.matvec(np.ones(X.shape[0]), cols=keep_idx_wo_intercept)
            else:
                X1 = X[:, keep_idx_wo_intercept].sum(axis=0)
            results = _find_intercept_alternative(
                gram[results.keep_idx, results.keep_idx], X1, results
            )

        if self.fit_intercept:
            results = _adjust_column_indices_for_intercept(results)

        self.drop_columns = results.drop_idx
        self.keep_columns = results.keep_idx
        self.replace_categories = []  # type: ignore
        self.intercept_safe = self.fit_intercept  # We never drop the intercept
        self.input_type = "csc"

    def transform(self, X: ArrayLike, y: Optional[VectorLike] = None) -> ArrayLike:
        """Transform the data by dropping collinear columns.

        Parameters
        ----------
        X : ArrayLike
            The data to transform. Can be a pandas.DataFrame
            a tabmat.SplitMatrix, or a numpy.ndarray.
        y: Optional[VectorLike]
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
        else:
            raise ValueError(
                f"X must be a pandas.DataFrame or a numpy.ndarray, got {type(X)}"
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
