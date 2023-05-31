"""Some utilities for transforming data before fitting a model."""

from collections import namedtuple
from typing import Optional

import numpy as np
import tabmat as tm
from scipy.linalg import qr
from sklearn.base import BaseEstimator, TransformerMixin

from ._glm import ArrayLike, VectorLike
from ._util import _safe_sandwich_dot

CollinearityResults = namedtuple(
    "CollinearityResults", ["keep_idx", "drop_idx", "intercept_safe"]
)


def _find_collinear_columns(
    X: tm.MatrixBase, fit_intercept: bool = False, tol: float = 1e-6
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

    permuted_keep_mask = np.abs(np.diag(R)) > tol
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


class Decollinearizer(TransformerMixin, BaseEstimator):
    """Drop collinear columns from the design matrix implied by a dataset.

    The type of the output is the same as the input. For non-categorical
    columns, collinear ones are simply dropped. For categorical columns
    (e.g. in a pandas.DataFrame or a tabmat.SplitMatrix), values whose
    columns in the design matrix would be dropped are replaced with the
    first category. This supposes that the first category is the reference,
    and will be dropped in the subsequent model fitting step.
    """

    def __init__(self, fit_intercept: bool = True) -> None:
        self.fit_intercept = fit_intercept

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
        raise NotImplementedError

    def transform(self, X: ArrayLike, y: Optional[VectorLike]) -> ArrayLike:
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
        raise NotImplementedError
