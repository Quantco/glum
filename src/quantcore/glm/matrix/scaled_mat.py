from typing import List, Optional, Union

import numpy as np
from scipy import sparse as sps

from quantcore.glm.matrix import MatrixBase


class ColScaledMat:
    """
    Matrix with ij element equal to mat[i, j] + shift[0, j]
    """

    # TODO: make shift 1d

    __array_priority__ = 11

    def __init__(self, mat: MatrixBase, shift: Union[np.ndarray, List]):
        shift_arr = np.atleast_1d(np.squeeze(shift))
        expected_shape = (mat.shape[1],)
        if not shift_arr.shape == expected_shape:
            raise ValueError(
                f"""Expected shift to be able to conform to shape {expected_shape},
            but it has shape {np.asarray(shift).shape}"""
            )

        self.shift = shift_arr
        self.mat = mat
        self.shape = mat.shape
        self.ndim = mat.ndim
        self.dtype = mat.dtype

    def dot(
        self,
        other_mat: Union[np.ndarray, List],
        rows: np.ndarray = None,
        cols: np.ndarray = None,
    ) -> np.ndarray:
        """
        This function returns a dense output, so it is best geared for the
        matrix-vector case.
        """

        other_mat = np.asarray(other_mat)
        mat_part = self.mat.dot(other_mat, rows, cols)

        if rows is None:
            rows = np.arange(self.shape[0], dtype=np.int32)
        if cols is None:
            cols = np.arange(self.shape[1], dtype=np.int32)
        shift_part = self.shift[cols].dot(other_mat[cols])
        return mat_part + shift_part

    def getcol(self, i: int):
        """
        Returns a ColScaledSpMat.

        >>> from scipy import sparse as sps
        >>> x = ColScaledMat(sps.eye(3), shift=[0, 1, -2])
        >>> col_1 = x.getcol(1)
        >>> isinstance(col_1, ColScaledMat)
        True
        >>> col_1.A
        array([[1.],
               [2.],
               [1.]])
        """
        return ColScaledMat(self.mat.getcol(i), [self.shift[i]])

    def sandwich(
        self, d: np.ndarray, rows: np.ndarray = None, cols: np.ndarray = None
    ) -> np.ndarray:
        """
        Performs a sandwich product: X.T @ diag(d) @ X
        """
        if not hasattr(d, "dtype"):
            d = np.asarray(d)
        if not self.mat.dtype == d.dtype:
            raise TypeError(
                f"""self.mat and d need to be of same dtype, either
                np.float64 or np.float32. This matrix is of type {self.mat.dtype},
                while d is of type {d.dtype}."""
            )

        if rows is None:
            rows = np.arange(self.shape[0], dtype=np.int32)
        if cols is None:
            cols = np.arange(self.shape[1], dtype=np.int32)

        term1 = self.mat.sandwich(d, rows, cols)
        d_mat = self.mat.transpose_dot(d, rows, cols)
        term2 = np.outer(d_mat, self.shift[cols])
        term3_and_4 = np.outer(
            self.shift[cols], d_mat + self.shift[cols] * d[rows].sum()
        )
        res = term2 + term3_and_4
        if isinstance(term1, sps.dia_matrix):
            idx = np.arange(res.shape[0])
            res[idx, idx] += term1.data[0, :]
        else:
            res += term1
        return res

    def unstandardize(self, col_stds: Optional[np.ndarray]) -> MatrixBase:
        """
        Doesn't need to use col_means because those are assumed to equal 'shift'.
        """
        if col_stds is not None:
            self.mat.scale_cols_inplace(col_stds)
        return self.mat

    def transpose_dot(
        self,
        other: Union[np.ndarray, List],
        rows: np.ndarray = None,
        cols: np.ndarray = None,
    ) -> np.ndarray:
        """
        Let self.shape = (N, K) and other.shape = (M, N).
        Let shift_mat = outer(ones(N), shift)

        (X.T @ other)[k, i] = (X.mat.T @ other)[k, i] + (shift_mat @ other)[k, i]
        (shift_mat @ other)[k, i] = (outer(shift, ones(N)) @ other)[k, i]
        = sum_j outer(shift, ones(N))[k, j] other[j, i]
        = sum_j shift[k] other[j, i]
        = shift[k] other.sum(0)[i]
        = outer(shift, other.sum(0))[k, i]
        """
        other = np.asarray(other)
        mat_part = self.mat.transpose_dot(other, rows, cols)

        if rows is None:
            rows = np.arange(self.shape[0], dtype=np.int32)
        if cols is None:
            cols = np.arange(self.shape[1], dtype=np.int32)
        other_sum = np.sum(other[rows], 0)
        shift_part = np.reshape(np.outer(self.shift[cols], other_sum), mat_part.shape)

        return mat_part + shift_part

    def __rmatmul__(self, other: Union[np.ndarray, List]) -> np.ndarray:
        """
        other @ X = (X.T @ other.T).T = X.transpose_dot(other.T).T

        Parameters
        ----------
        other: array-like

        Returns
        -------
        array

        """
        if not hasattr(other, "T"):
            other = np.asarray(other)
        return self.transpose_dot(other.T).T  # type: ignore

    def __matmul__(self, other):
        """ Defines the behavior of 'self @ other'. """
        return self.dot(other)

    def toarray(self) -> np.ndarray:
        return self.mat.A + self.shift[None, :]

    @property
    def A(self) -> np.ndarray:
        return self.toarray()

    def astype(self, dtype, order="K", casting="unsafe", copy=True):
        return type(self)(
            self.mat.astype(dtype, casting=casting, copy=copy),
            self.shift.astype(dtype, order=order, casting=casting, copy=copy),
        )

    def __getitem__(self, item):
        if isinstance(item, tuple):
            row, col = item
        else:
            row = item
            col = slice(None, None, None)

        mat_part = self.mat.__getitem__(item)
        shift_part = self.shift[col]

        if isinstance(row, int):
            return mat_part.A + shift_part

        return ColScaledMat(mat_part, np.atleast_1d(shift_part))
