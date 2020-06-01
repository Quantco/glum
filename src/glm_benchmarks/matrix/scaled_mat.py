from typing import List, Union

import numpy as np
from scipy import sparse as sps

from glm_benchmarks.matrix import MatrixBase


class ColScaledMat:
    """
    Matrix with ij element equal to mat[i, j] + shift[1, j]
    """

    # TODO: make shift 1d

    __array_priority__ = 11

    def __init__(self, mat: MatrixBase, shift: Union[np.ndarray, List]):
        shift = np.asarray(shift)
        if shift.ndim == 1:
            shift = np.expand_dims(shift, 1 - self.scale_axis())
        else:
            if self.scale_axis() == 0:
                expected_shape = (mat.shape[0], 1)
            else:
                expected_shape = (1, mat.shape[1])
            if not shift.shape == expected_shape:
                raise ValueError(
                    f"""Expected shift to have shape {expected_shape},
                but it has shape {shift.shape}"""
                )

        self.shift = shift
        self.mat = mat
        self.shape = mat.shape
        self.ndim = mat.ndim
        self.dtype = mat.dtype

    @classmethod
    def scale_axis(self) -> int:
        return 1

    def dot(self, other_mat: Union[sps.spmatrix, np.ndarray]):
        """
        Let self.shape = (n, k).

        If other.shape = (k, m):

            result[i, j] = sum_k self[i, k] @ other_mat[k, j]
                         = sum_k (self.mat[i, k] + self.shift[0, k]) @ other_mat[k, j]
            result       = self.mat @ other_mat + self.shift @ other_mat
                           (n, m)                  (1, m)

            If other is sparse, result = ColScaledSpMat(self.mat @ other_mat,
                                                         self.shift @other_mat)
            If other is dense, result = self.mat @ other_mat + self.shift @ other_mat

        If other.shape = (k,):

            result = self.mat @ other_mat + self.shift @ other_mat
                           (n,)                  (1,)
        """
        mat_part = self.mat.dot(other_mat)
        if sps.issparse(other_mat):
            # np.dot doesn't work well with a sparse matrix right argument
            shifter = other_mat.T.dot(self.shift.T).T
        else:
            shifter = self.shift.dot(other_mat)

        if sps.issparse(mat_part):
            return ColScaledMat(mat_part, shifter)
        return mat_part + shifter

    def getcol(self, i: int):
        """
        Returns a ColScaledSpMat.

        >>> x = ColScaledMat(sps.eye(3), shift=[0, 1, -2])
        >>> col_1 = x.getcol(1)
        >>> isinstance(col_1, ColScaledMat)
        True
        >>> col_1.A
        array([[1.],
               [2.],
               [1.]])
        """
        return ColScaledMat(self.mat.getcol(i), [self.shift[0, i]])

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
        term4 = (self.shift.T * self.shift) * d.sum()
        return term1 + term2 + term3 + term4

    def unstandardize(self, col_stds: np.ndarray) -> MatrixBase:
        """
        Doesn't need to use col_means because those are assumed to equal 'shift'.
        """
        self.mat.scale_cols_inplace(col_stds)
        return self.mat

    def __rmatmul__(self, other: np.ndarray) -> np.ndarray:
        """
        Let self.shape = (N, K) and other.shape = (M, N).
        Remember self.shift = ones(N, 1) x (1, K)

        (other @ X)[i, j] = (other @ x.mat)[i, j] + other @ x.shift
        (other @ shift)[i, j] = (other @ ones(n, 1) @ self.shift)[i, j]
        = sum_k other[i, k] self.shift[j]
        = other.sum(1) @ shift
        """
        # TODO: write a check_1d function
        other = np.asarray(other)
        other_sum = other.sum(-1)
        if not sps.issparse(other) and other.ndim > 1:
            other_sum = np.expand_dims(other_sum, 1)
            assert other_sum.shape == (other.shape[0], 1)
        mat_part = other @ self.mat
        if other.ndim == 1:
            # other_sum is a float
            shift_part = self.shift[0, :] * other_sum
            assert shift_part.shape == (self.shape[1],)
        else:
            shift_part = other_sum @ self.shift
        result = mat_part + shift_part
        if other.ndim == 1:
            assert result.shape == (self.shape[1],)
        else:
            assert result.shape == (other.shape[0], self.shape[1])
        return result

    def __matmul__(self, other):
        """ Defines the behavior of 'self @ other'. """
        return self.dot(other)

    def toarray(self) -> np.ndarray:
        return self.mat.A + self.shift

    @property
    def A(self) -> np.ndarray:
        return self.toarray()

    def astype(self, dtype, order="K", casting="unsafe", copy=True):
        return type(self)(
            self.mat.astype(dtype, casting=casting, copy=copy),
            self.shift.astype(dtype, order=order, casting=casting, copy=copy),
        )
