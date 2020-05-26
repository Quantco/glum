from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np
from scipy import sparse as sps

from .matrix_base import MatrixBase
from .mkl_sparse_matrix import MKLSparseMatrix


class ScaledMat(ABC):
    """
    Base class for ColScaledSpMat and RowScaledSpMat. Do not instantiate.
    """

    def __init__(self, mat: sps.spmatrix, shift: np.ndarray):

        if not (sps.issparse(mat) or isinstance(mat, MKLSparseMatrix)):
            raise ValueError("mat should be a sparse matrix.")

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
        self.mat = MKLSparseMatrix(mat)
        self.shape = mat.shape
        self.ndim = mat.ndim
        self.dtype = mat.dtype

    def astype(self, dtype, order="K", casting="unsafe", copy=True):
        return type(self)(
            self.mat.astype(dtype, casting=casting, copy=copy),
            self.shift.astype(dtype, order=order, casting=casting, copy=copy),
        )

    @classmethod
    @abstractmethod
    def scale_axis(self) -> int:
        pass

    def toarray(self) -> np.ndarray:
        return self.mat.A + self.shift

    def multiply(self, other: Union[np.ndarray, float]):
        """
        Let self.shape = (n, k)

        Cases:
        other.shape = (n, k):
            result = self.mat * other + other * self.shift
            sum of sparse matrices
        other_mat.shape = (self.shape[0], 1) or (1, self.shape[1]):
            Broadcast multiplication.
            If axis not equal to 1 is same as self.scale_axis,
            col case:
            result[i, j] = self.mat[i, j] * other[0, j] + self.shift[0, j] * other[0, j]
                         = self.mat * other + self.shift * other
                         = ColScaledSpMat(self.mat * other, self.shift * other)
            row case:
            result[i, j] = self.mat[i, j] other[i, 0] + self.shift[i, 0] * other[i, 0]
                         = self.mat * other + self.shift * other
                         = RowScaledSpMat(self.mat * other, self.shift * other)
            If axis not equal to 1 is not the same as self.scale_axis, result will be
                dense. Not supported.
        other_mat.shape = (self.shape[1],) or (self.shape[0],):
            Expand dims to correct shape, as above.
        other_mat is scalar:
            Same as broadcast case.
        """
        mat_part = self.mat.multiply(other)
        shift_part = self.shift * other
        if shift_part.shape == self.shift.shape:
            return type(self)(mat_part, shift_part)
        else:
            return mat_part + shift_part

    def sum(self, axis: int = None) -> Union[np.ndarray, float]:
        """
        For col case:
        axis = 0:
            result[i, j] = sum_i (mat[i, j] + shift[1, j])
                         = mat.sum(0)[1, j] + shift[j] * self.shape[0]
            result = mat.sum() + shift * self.shape
        axis = 1:
            sum_j (mat[i, j] + shift[j]) = mat.sum(1) + shift.sum()
        axis=None:
            col case:
            mat.sum(None) + sum_i sum_j shift[1, j]
            = mat.sum(None) + mat.shape[0] * shift.sum()

        Row case is symmetric.
        """
        shift_sum = self.shift.sum(axis=axis)

        if axis is None:
            shift_part = shift_sum * self.shape[1 - self.scale_axis()]
            assert np.isscalar(shift_part)
        elif axis == self.scale_axis():
            shift_part = shift_sum
        else:
            shift_part = np.expand_dims(
                shift_sum * self.shape[axis], 1 - self.scale_axis()
            )
        return self.mat.sum(axis) + shift_part

    def __mul__(self, other):
        """ Defines the behavior of "*", element-wise multiplication. """
        return self.multiply(other)

    def mean(self, axis: Optional[int]) -> Union[np.ndarray, float]:
        if axis is None:
            denominator = self.shape[0] * self.shape[1]
        else:
            denominator = self.shape[axis]
        return self.sum(axis) / denominator

    @property
    def A(self) -> np.ndarray:
        return self.toarray()

    @abstractmethod
    def transpose(self):
        pass

    @property
    def T(self):
        return self.transpose()


class ColScaledSpMat(ScaledMat, MatrixBase):
    """
    Matrix with ij element equal to mat[i, j] + shift[1, j]
    """

    def __init__(self, mat: sps.spmatrix, shift: np.ndarray):
        super().__init__(mat, shift)

    @classmethod
    def scale_axis(self) -> int:
        return 1

    def transpose(self):
        return RowScaledSpMat(self.mat.T, self.shift.T)

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
            return ColScaledSpMat(mat_part, shifter)
        return mat_part + shifter

    def power(self, p: float):
        tmp = self.mat.tocsc(copy=True)
        shift_power = self.shift ** p
        # Update data, looping over columns
        for j in range(self.shape[1]):
            start, end = tmp.indptr[j : j + 2]
            data = tmp.data[start:end]
            pow = shift_power[0, j]
            tmp.data[start:end] = (data + self.shift[0, j]) ** p - pow
        return ColScaledSpMat(tmp, shift_power)

    def getcol(self, i: int) -> ScaledMat:
        """
        Returns a ColScaledSpMat.

        >>> x = ColScaledSpMat(sps.eye(3), shift=[0, 1, -2])
        >>> col_1 = x.getcol(1)
        >>> isinstance(col_1, ColScaledSpMat)
        True
        >>> col_1.A
        array([[1.],
               [2.],
               [1.]])
        """
        return ColScaledSpMat(self.mat.getcol(i), [self.shift[0, i]])
        # return np.squeeze(self.mat.getcol(i).toarray()) + self.shift[0, i]

    def getrow(self, i: int) -> np.ndarray:
        """
        >>> x = ColScaledSpMat(sps.eye(3), shift=[0, 1, -2])
        >>> x.getrow(1)
        array([ 0.,  2., -2.])
        """
        return np.squeeze(self.mat.getrow(i).toarray()) + np.squeeze(self.shift)

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

    def sandwich_dense(self, B: np.ndarray, d: np.ndarray) -> np.ndarray:
        """
        sandwich product: self.T @ diag(d) @ B
        """
        if self.mat.dtype != d.dtype or B.dtype != d.dtype:
            raise TypeError(
                f"""self.mat, B and d all need to be of same dtype, either
                np.float64 or np.float32. This matrix is of type {self.mat.dtype},
                B is of type {B.dtype}, while d is of type {d.dtype}."""
            )
        term1 = self.mat.sandwich_dense(B, d)
        term2 = self.shift.T * (d @ B)[np.newaxis, :]
        return term1 + term2

    def unstandardize(self, col_means, col_stds):
        """
        Doesn't need to use col_means because those are assumed to equal 'shift'.
        """
        if sps.isspmatrix_csc(self.mat):
            from .standardize import _scale_csc_columns_inplace

            _scale_csc_columns_inplace(self.mat, col_stds)
            return self.mat
        else:
            return self.mat @ sps.diags(col_stds)


class RowScaledSpMat(ScaledMat):
    """
    Matrix with ij element equal to mat[i, j] + shift[i]
    """

    def __init__(self, mat: sps.spmatrix, shift: np.ndarray):
        super().__init__(mat, shift)

    @classmethod
    def scale_axis(self) -> int:
        return 0

    def transpose(self) -> ColScaledSpMat:
        return ColScaledSpMat(self.mat.T, self.shift.T)

    def dot(self, other_mat: Union[sps.spmatrix, np.ndarray]) -> np.ndarray:
        """
        Let self.shape = (n, k).

        If other.shape = (k, m):

            result[i, j] = sum_k self[i, k] @ other_mat[k, j]
                         = sum_k (self.mat[i, k] + self.shift[i, 1]) @ other_mat[k, j]
            result = self.mat @ other_mat + self.shift @ other_mat.sum(0)
                      (n, m)                 (n, 1) @ (1, m) = (n, m)
            This is dense!

        If other.shape = (k,):

            result[i, j] = self.mat @ other_mat + self.shift @ other_mat.sum(0)
                           (n,)                  (n, 1) @ (1,) = (n,)
        """
        mat_part = self.mat.dot(other_mat)
        other_sum = np.sum(other_mat, 0)
        if not sps.issparse(other_mat):
            # with numpy, sum is of shape (k,); with scipy it is of shape (1, k)
            other_sum = np.expand_dims(other_sum, 0)
        shift_part = self.shift.dot(other_sum)
        return mat_part + shift_part

    def __matmul__(self, other):
        """ Defines the behavior of 'self @ other'. """
        return self.dot(other)

    def power(self, p: float):
        return self.T.power(p).T

    def getcol(self, i: int) -> np.ndarray:
        """
        >>> x = RowScaledSpMat(sps.eye(3), shift=[0, 1, -2])
        >>> x.getcol(1)
        array([[ 0.],
               [ 2.],
               [-2.]])
        """
        return self.mat.getcol(i).toarray() + self.shift

    def getrow(self, i: int) -> ScaledMat:
        """
        Returns a RowScaledSpMat.
        >>> x = RowScaledSpMat(sps.eye(3), shift=[0, 1, -2])
        >>> row_i = x.getrow(1)
        >>> isinstance(row_i, RowScaledSpMat)
        True
        >>> row_i.A
        array([[1., 2., 1.]])
        """
        return RowScaledSpMat(self.mat.getrow(i), [self.shift[i, 0]])
