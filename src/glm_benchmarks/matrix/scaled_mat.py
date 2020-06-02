from typing import List, Union

import numpy as np

from glm_benchmarks.matrix import MatrixBase

from .util import rmatmul_vector_only


class ColScaledMat:
    """
    Matrix with ij element equal to mat[i, j] + shift[1, j]
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

    def dot(self, other_mat: Union[np.ndarray, List]) -> np.ndarray:
        """
        This function returns a dense output, so it is best geared for the
        matrix-vector case.
        """
        return self.mat.dot(other_mat) + self.shift.dot(other_mat)

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

    def sandwich(self, d: np.ndarray) -> np.ndarray:
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
        term1 = self.mat.sandwich(d)
        term2 = (d @ self.mat)[:, np.newaxis] * self.shift
        term3 = term2.T
        term4 = np.outer(self.shift, self.shift) * d.sum()
        return term1 + term2 + term3 + term4

    def unstandardize(self, col_stds: np.ndarray) -> MatrixBase:
        """
        Doesn't need to use col_means because those are assumed to equal 'shift'.
        """
        self.mat.scale_cols_inplace(col_stds)
        return self.mat

    def transpose_dot_vec(self, other: np.ndarray) -> np.ndarray:
        """
        Let self.shape = (N, K) and other.shape = (M, N).
        Remember self.shift = ones(N, 1) x (1, K)

        (other @ X)[i, j] = (other @ x.mat)[i, j] + other @ x.shift
        (other @ shift)[i, j] = (other @ ones(n, 1) @ self.shift)[i, j]
        = sum_k other[i, k] self.shift[j]
        = other.sum(1) @ shift
        """
        other = np.atleast_1d(np.squeeze(other))
        other_sum = other.sum()
        mat_part = self.mat.transpose_dot_vec(other)
        shift_part = self.shift * other_sum
        result = mat_part + shift_part
        return result

    def __rmatmul__(self, other: Union[np.ndarray, List]) -> np.ndarray:
        return rmatmul_vector_only(self, np.asarray(other).T)

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
