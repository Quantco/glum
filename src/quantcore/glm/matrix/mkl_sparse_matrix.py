from typing import List, Union

import numpy as np
from scipy import sparse as sps
from sparse_dot_mkl import dot_product_mkl

from quantcore.glm.matrix.ext.sparse import (
    csc_rmatvec,
    csr_dense_sandwich,
    csr_matvec,
    sparse_sandwich,
)

from . import MatrixBase
from .standardize import _scale_csc_columns_inplace


class MKLSparseMatrix(sps.csc_matrix, MatrixBase):
    """
    A scipy.sparse csc matrix subclass that will use MKL for sparse
    matrix-vector products and will use the fast sparse_sandwich function
    for sandwich products.
    """

    def __init__(self, arg1, shape=None, dtype=None, copy=False):
        """
        Instantiate in the same way as scipy.sparse.csc_matrix
        """
        super().__init__(arg1, shape, dtype, copy)
        if not self.has_sorted_indices:
            self.sort_indices()
        self.x_csr = None

    def _check_csr(self):
        if self.x_csr is None:
            self.x_csr = self.tocsr(copy=False)

    def sandwich(
        self, d: np.ndarray, rows: np.ndarray = None, cols: np.ndarray = None
    ) -> np.ndarray:
        if not hasattr(d, "dtype"):
            d = np.asarray(d)
        if not self.dtype == d.dtype:
            raise TypeError(
                f"""self and d need to be of same dtype, either np.float64
                or np.float32. self is of type {self.dtype}, while d is of type
                {d.dtype}."""
            )

        self._check_csr()

        if rows is None:
            rows = np.arange(self.shape[0], dtype=np.int32)
        if cols is None:
            cols = np.arange(self.shape[1], dtype=np.int32)

        return sparse_sandwich(self.tocsc(copy=False), self.x_csr, d, rows, cols)

    def sandwich_dense(
        self,
        B: np.ndarray,
        d: np.ndarray,
        rows: np.ndarray,
        L_cols: np.ndarray,
        R_cols: np.ndarray,
    ) -> np.ndarray:
        """
        sandwich product: self.T @ diag(d) @ B
        """
        if not hasattr(d, "dtype"):
            d = np.asarray(d)

        if self.dtype != d.dtype or B.dtype != d.dtype:
            raise TypeError(
                f"""self, B and d all need to be of same dtype, either
                np.float64 or np.float32. This matrix is of type {self.dtype},
                B is of type {B.dtype}, while d is of type {d.dtype}."""
            )
        if np.issubdtype(d.dtype, np.signedinteger):
            d = d.astype(float)

        self._check_csr()
        return csr_dense_sandwich(self.x_csr, B, d, rows, L_cols, R_cols)

    def dot(self, vec, rows: np.ndarray = None, cols: np.ndarray = None):
        if not isinstance(vec, np.ndarray) and not sps.issparse(vec):
            vec = np.asarray(vec)
        if rows is None and cols is None:
            if len(vec.shape) == 1:
                return dot_product_mkl(self, vec)
            if len(vec.shape) == 2 and vec.shape[1] == 1:
                return dot_product_mkl(self, vec[:, 0])[:, None]
            # TODO: warn that the rows and cols parameters aren't used with matrix-multiplies
            return super().dot(vec)
        else:
            if rows is None:
                rows = np.arange(self.shape[0], dtype=np.int32)
            if cols is None:
                cols = np.arange(self.shape[1], dtype=np.int32)
            if vec.ndim == 1:
                self._check_csr()
                return csr_matvec(self.x_csr, vec, rows, cols)
            elif vec.ndim == 2 and vec.shape[1] == 1:
                self._check_csr()
                return csr_matvec(self.x_csr, vec[:, 0], rows, cols)[:, None]
            else:
                return self[np.ix_(rows, cols)].dot(vec[cols])

    __array_priority__ = 12

    def transpose_dot(
        self,
        vec: Union[np.ndarray, List],
        rows: np.ndarray = None,
        cols: np.ndarray = None,
    ) -> np.ndarray:
        vec = np.asarray(vec)
        if rows is None and cols is None:
            if vec.ndim == 1:
                return dot_product_mkl(self.T, vec)
            elif vec.ndim == 2 and vec.shape[1] == 1:
                return dot_product_mkl(self.T, vec[:, 0])[:, None]
            return super().T.dot(vec)
        else:
            if rows is None:
                rows = np.arange(self.shape[0], dtype=np.int32)
            if cols is None:
                cols = np.arange(self.shape[1], dtype=np.int32)
            if vec.ndim == 1:
                return csc_rmatvec(self, vec, rows, cols)
            elif vec.ndim == 2 and vec.shape[1] == 1:
                return csc_rmatvec(self, vec[:, 0], rows, cols)[:, None]
            else:
                return self[np.ix_(rows, cols)].T.dot(vec[rows])

    def get_col_stds(self, weights: np.ndarray, col_means: np.ndarray) -> np.ndarray:
        return np.sqrt(self.power(2).T.dot(weights) - col_means ** 2)

    def scale_cols_inplace(self, col_scaling: np.ndarray) -> None:
        _scale_csc_columns_inplace(self, col_scaling)
