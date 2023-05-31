"""Some utilities for transforming data before fitting a model."""

from collections import namedtuple
from typing import Optional

import tabmat as tm
from sklearn.base import BaseEstimator, TransformerMixin

from ._glm import ArrayLike, VectorLike

CollinearityResults = namedtuple("CollinearityResults", ["keep_idx", "drop_idx"])


def _find_collinear_columns(X: tm.MatrixBase) -> CollinearityResults:
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
    raise NotImplementedError


class Decollinearizer(TransformerMixin, BaseEstimator):
    """Drop collinear columns from the design matrix implied by a dataset.

    The type of the output is the same as the input. For non-categorical
    columns, collinear ones are simply dropped. For categorical columns
    (e.g. in a pandas.DataFrame or a tabmat.SplitMatrix), values whose
    columns in the design matrix would be dropped are replaced with the
    first category. This supposes that the first category is the reference,
    and will be dropped in the subsequent model fitting step.
    """

    def __init__(self) -> None:
        raise NotImplementedError

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
