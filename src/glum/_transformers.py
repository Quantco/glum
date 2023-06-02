"""Some utilities for transforming data before fitting a model."""

from typing import Hashable, List, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import tabmat as tm
from scipy.linalg import lstsq, qr, solve
from sklearn.base import BaseEstimator, TransformerMixin, check_is_fitted

from ._glm import ArrayLike, VectorLike
from ._util import _safe_sandwich_dot


class CollinearityResults(NamedTuple):
    """Results of collinearity analysis."""

    keep_idx: Sequence[int]
    drop_idx: Sequence[int]
    intercept_safe: bool


def _find_collinear_columns_pandas(
    df: pd.DataFrame,
    fit_intercept=True,
    mode: str = "gram",
    use_tabmat: bool = True,
    tolerance=1e-6,
) -> CollinearityResults:
    """Find the collinear columns and categories in a pandas DataFrame.
    Return all information needed to decollinearize the data.

    Parameters
    ----------
    df : pd.DataFrame
        The data to decompose.
    fit_intercept : bool, optional
        Whether to fit an intercept, by default True
    mode : str, optional
        Either 'gram' or 'qr', by default 'gram'.
        "gram" runs QR decomposition on X'X
        "direct" runs QR decomposition directly on X
    use_tabmat : bool, optional
        Whether to convert the data to a tabmat.SplitMatrix before decomposition.
        It is recommended for datasets with high-cardinality categorical columns.

    Returns
    -------
    CollinearityResults
        Information about the columns to keep and the columns to drop.
    """
    if mode == "gram":
        if use_tabmat:
            X = tm.from_pandas(df, drop_first=True)
            # TODO: add checks here
            gram = _safe_sandwich_dot(X, np.ones(X.shape[0]), intercept=fit_intercept)
        else:
            X = pd.get_dummies(df, drop_first=True)
            if fit_intercept:
                X.insert(0, "Intercept", 1)
            gram = X.to_numpy().T @ X.to_numpy()

        R, P = qr(gram, mode="r", pivoting=True)  # type: ignore

    elif mode == "direct":
        if use_tabmat:
            raise NotImplementedError(
                "Direct mode for tabmat matrices not implemented."
            )
        else:
            X = pd.get_dummies(df, drop_first=True)
            if fit_intercept:
                X.insert(0, "Intercept", 1)
            R, P = qr(X.to_numpy(), mode="r", pivoting=True)  # type: ignore

    else:
        raise ValueError(f"Mode must be 'gram' or 'direct', got {mode}")
        # Cannot happen

    results = _find_collinear_columns(R, P, fit_intercept, tolerance=tolerance)

    if fit_intercept and not results.intercept_safe:
        if mode == "gram":
            if use_tabmat:
                # rowsum in tabmat would be nice here
                lin_comb = solve(
                    gram[results.keep_idx, results.keep_idx],
                    X.transpose_matvec(np.ones(X.shape[0]), cols=results.keep_idx),
                )
            else:
                lin_comb = solve(
                    gram[results.keep_idx, results.keep_idx],
                    X[:, results.keep_idx].sum(axis=0),
                )
        elif mode == "direct":
            if use_tabmat:
                raise NotImplementedError(
                    "Direct mode for tabmat matrices not implemented."
                )
            else:
                lin_comb = lstsq(
                    X[:, results.keep_idx],
                    np.ones(X.shape[0]),
                )[0]

        drop_instead_of_intercept = results.keep_idx[np.argmax(np.abs(lin_comb))]
        keep_idx = results.keep_idx[results.keep_idx != drop_instead_of_intercept]
        drop_insert_idx = np.searchsorted(results.drop_idx, drop_instead_of_intercept)
        drop_idx = np.insert(
            results.drop_idx, drop_insert_idx, drop_instead_of_intercept
        )
        intercept_safe = True
    else:
        keep_idx = results.keep_idx
        drop_idx = results.drop_idx
        if fit_intercept:
            intercept_safe = True
        else:
            intercept_safe = False

    return CollinearityResults(keep_idx, drop_idx, intercept_safe)


def _find_collinear_columns_numpy(
    X: np.ndarray, fit_intercept=True, mode="gram", tolerance: float = 1e-6
) -> CollinearityResults:
    """Find the collinear columns in a numpy array.
    Return all information needed to decollinearize the array.

    Parameters
    ----------
    df : pd.DataFrame
        The data to decompose.
    fit_intercept : bool, optional
        Whether to fit an intercept, by default True
    mode : str, optional
        Either 'gram' or 'qr', by default 'gram'.
        "gram" runs QR decomposition on X'X
        "direct" runs QR decomposition directly on X

    Returns
    -------
    CollinearityResults
        Information about the columns to keep and the columns to drop.
    """
    if fit_intercept:
        X = np.hstack((np.ones((X.shape[0], 1)), X))
    if mode == "gram":
        gram = X.T @ X
        R, P = qr(gram, mode="r", pivoting=True)  # type: ignore
    elif mode == "direct":
        R, P = qr(X, mode="r", pivoting=True)  # type: ignore
    else:
        raise ValueError(f"Mode must be 'gram' or 'direct', got {mode}")
        # Cannot happen

    results = _find_collinear_columns(R, P, fit_intercept, tolerance=tolerance)

    if fit_intercept and not results.intercept_safe:
        if mode == "gram":
            lin_comb = solve(
                gram[results.keep_idx, results.keep_idx],
                X[:, results.keep_idx].sum(axis=0),
            )
        elif mode == "direct":
            lin_comb = lstsq(
                X[:, results.keep_idx],
                np.ones(X.shape[0]),
            )[0]

        drop_instead_of_intercept = results.keep_idx[np.argmax(np.abs(lin_comb))]
        keep_idx = results.keep_idx[results.keep_idx != drop_instead_of_intercept]
        drop_insert_idx = np.searchsorted(results.drop_idx, drop_instead_of_intercept)
        drop_idx = np.insert(
            results.drop_idx, drop_insert_idx, drop_instead_of_intercept
        )
        intercept_safe = True
    else:
        keep_idx = results.keep_idx
        drop_idx = results.drop_idx
        if fit_intercept:
            intercept_safe = True
        else:
            intercept_safe = False

    return CollinearityResults(keep_idx, drop_idx, intercept_safe)


def _find_collinear_columns(
    R: np.ndarray, P: np.ndarray, fit_intercept, tolerance: float = 1e-6
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

    intercept_safe = False
    if fit_intercept:
        if 0 not in drop_idx:
            intercept_safe = True
        keep_idx -= 1
        drop_idx -= 1

    keep_idx = keep_idx[keep_idx != -1]
    drop_idx = drop_idx[drop_idx != -1]

    return CollinearityResults(keep_idx, drop_idx, intercept_safe)


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
        mode: Optional[str] = None,
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
        if mode not in ["gram", "direct", None]:
            raise ValueError(f"Mode must be 'gram' or 'direct', got {mode}")

        if isinstance(X, pd.DataFrame):
            self._fit_pandas(X, mode=mode, use_tabmat=use_tabmat)
        elif isinstance(X, np.ndarray):
            self._fit_numpy(X, mode=mode, use_tabmat=use_tabmat)
        else:
            raise ValueError(
                f"X must be a pandas.DataFrame or a numpy.ndarray, got {type(X)}"
            )
        return self

    def _fit_pandas(
        self,
        X: pd.DataFrame,
        mode: Optional[str] = None,
        use_tabmat: Optional[bool] = None,
    ) -> None:
        """Fit the transformer on a pandas.DataFrame."""
        # TODO: make sure that object columns are handled the same in all modes
        if mode is None:
            mode = "gram"
        if use_tabmat is None:
            use_tabmat = True

        results = _find_collinear_columns_pandas(
            X,
            fit_intercept=self.fit_intercept,
            mode=mode,
            use_tabmat=use_tabmat,
            tolerance=self.tolerance,
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
        self.keep_columns = X.columns.difference(drop_columns)
        self.replace_categories = replace_categories  # type: ignore
        self.intercept_safe = results.intercept_safe
        self.input_type = "pandas"

    def _fit_numpy(
        self,
        X: np.ndarray,
        mode: Optional[str] = None,
        use_tabmat: Optional[bool] = None,
    ) -> None:
        """Fit the transformer on a numpy.ndarray."""
        if mode is None:
            mode = "direct"
        if use_tabmat is None:
            use_tabmat = False

        if use_tabmat:
            raise ValueError("use_tabmat=True is not supported for numpy arrays")
        results = _find_collinear_columns_numpy(
            X, fit_intercept=self.fit_intercept, mode=mode, tolerance=self.tolerance
        )
        self.drop_columns = results.drop_idx
        self.keep_columns = results.keep_idx
        self.intercept_safe = results.intercept_safe
        self.replace_categories = []
        self.input_type = "numpy"

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
