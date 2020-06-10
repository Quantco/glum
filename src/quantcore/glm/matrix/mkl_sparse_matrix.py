from typing import List, Union

import numpy as np
from scipy import sparse as sps
from sparse_dot_mkl import dot_product_mkl

from quantcore.glm.matrix.sandwich.sandwich import csr_dense_sandwich, sparse_sandwich

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

    def sandwich_dense(self, B: np.ndarray, d: np.ndarray) -> np.ndarray:
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
        return csr_dense_sandwich(self.x_csr, B, d)

    def dot(self, v, rows: np.ndarray = None, cols: np.ndarray = None):
        if not isinstance(v, np.ndarray) and not sps.issparse(v):
            v = np.asarray(v)
        if rows is None and cols is None:
            if len(v.shape) == 1:
                return dot_product_mkl(self, v)
            if len(v.shape) == 2 and v.shape[1] == 1:
                return dot_product_mkl(self, v[:, 0])[:, None]
            # TODO: warn that the rows and cols parameters aren't used with matrix-multiplies
            return sps.csc_matrix.dot(self, v)
        else:
            if rows is None:
                rows = np.arange(self.shape[0], dtype=np.int32)
            if cols is None:
                cols = np.arange(self.shape[1], dtype=np.int32)
            if len(v.shape) == 1:
                return dot_product_mkl(self[np.ix_(rows, cols)], v[cols])
            elif len(v.shape) == 2 and v.shape[1] == 1:
                return dot_product_mkl(self[np.ix_(rows, cols)], v[cols, 0])[:, None]
            else:
                return self[np.ix_(rows, cols)].dot(v[cols])

    __array_priority__ = 12

    def transpose_dot(
        self,
        vec: Union[np.ndarray, List],
        rows: np.ndarray = None,
        cols: np.ndarray = None,
    ) -> np.ndarray:
        vec = np.squeeze(np.asarray(vec))
        if rows is None and cols is None:
            return dot_product_mkl(self.T, vec)
        else:
            if rows is None:
                rows = np.arange(self.shape[0], dtype=np.int32)
            if cols is None:
                cols = np.arange(self.shape[1], dtype=np.int32)
            return dot_product_mkl(self[np.ix_(rows, cols)].T, vec[rows])

    def get_col_stds(self, weights: np.ndarray, col_means: np.ndarray) -> np.ndarray:
        return np.sqrt(self.power(2).T.dot(weights) - col_means ** 2)

    def scale_cols_inplace(self, col_scaling: np.ndarray) -> None:
        _scale_csc_columns_inplace(self, col_scaling)
