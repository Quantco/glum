"""Some utilities for transforming data before fitting a model."""

from typing import Hashable, List, NamedTuple, Optional

import numpy as np
import pandas as pd
import tabmat as tm
from scipy.linalg import qr
from sklearn.base import BaseEstimator, TransformerMixin, check_is_fitted

from ._glm import ArrayLike, VectorLike
from ._util import _safe_sandwich_dot


class CollinearityResults(NamedTuple):
    """Results of collinearity analysis."""

    keep_idx: List[int]
    drop_idx: List[int]
    intercept_safe: bool


def _find_collinear_columns(
    X: tm.MatrixBase, fit_intercept: bool = False, tolerance: float = 1e-6
) -> CollinearityResults:
    """Determine the rank of X from the QR decomposition of X^T X.

    Parameters
    ----------
    X : tm.MatrixBase
        The design matrix.

    Returns
    -------
    CollinearityResults
        The indices of the columns to keep and the columns to drop.
    """
    gram = _safe_sandwich_dot(X, np.ones(X.shape[0]), intercept=fit_intercept)
    R, P = qr(gram, mode="r", pivoting=True)  # type: ignore

    permuted_keep_mask = np.abs(np.diag(R)) > tolerance
    keep_mask = np.empty_like(permuted_keep_mask)
    keep_mask[P] = permuted_keep_mask

    keep_idx = np.where(keep_mask)[0]
    drop_idx = np.where(~keep_mask)[0]

    intercept_safe = False
    if fit_intercept:
        if 0 not in drop_idx:
            intercept_safe = True
        keep_idx -= 1
        drop_idx -= 1

    return CollinearityResults(keep_idx, drop_idx, intercept_safe)


class ColumnMap(NamedTuple):
    """Mapping from DataFrame to design matrix."""

    column_pos: int
    column_name: Hashable
    categorical: bool
    category: Optional[str] = None
    base_category: Optional[str] = None


def _get_column_mapping(X: pd.DataFrame) -> List[ColumnMap]:
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

    def __init__(self, fit_intercept: bool = True, tolerance: float = 1e-6) -> None:
        self.fit_intercept = fit_intercept
        self.tolerance = tolerance

    def fit(self, X: ArrayLike, y: Optional[VectorLike] = None) -> "Decollinearizer":
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
            self._fit_pandas(X)
        else:
            raise NotImplementedError
        return self

    def _fit_pandas(self, X: pd.DataFrame) -> None:
        """Fit the transformer on a pandas.DataFrame."""
        X_tm = tm.from_pandas(X, drop_first=True)  # TODO: checks, like in ._glm
        results = _find_collinear_columns(
            X_tm, fit_intercept=self.fit_intercept, tolerance=self.tolerance
        )
        self.column_mapping = _get_column_mapping(X)
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
        self.replace_categories = replace_categories
        self.intercept_safe = results.intercept_safe
        self.input_type = "pandas"

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
        else:
            raise NotImplementedError
        return self

    def _transform_pandas(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit the transformer on a pandas.DataFrame."""
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

        return X
