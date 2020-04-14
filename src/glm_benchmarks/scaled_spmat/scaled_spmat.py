from abc import ABC, abstractclassmethod, abstractmethod
from typing import Union

import numpy as np
from scipy import sparse as sps


class ScaledMat(ABC):
    """
    Base class for ColScaledSpMat and RowScaledSpMat. Do not instantiate.
    """

    def __init__(self, mat: sps.spmatrix, shift: np.ndarray):

        if not sps.issparse(mat):
            raise ValueError("mat should be a sparse matrix.")

        if shift.ndim == 1:
            shift = np.expand_dims(shift, 1 - self.scale_axis)
        else:
            if self.scale_axis == 0:
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

    @property  # type: ignore
    @abstractclassmethod
    def scale_axis(self) -> int:
        return 0

    def todense(self) -> np.ndarray:
        return self.mat.A + self.shift

    @property
    def A(self):
        return self.todense()

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

    def __mul__(self, other):
        """ Defines the beahvior of "*". """
        return self.multiply(other)

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
            shift_part = shift_sum * self.shape[1 - self.scale_axis]
            assert np.isscalar(shift_part)
        elif axis == self.scale_axis:
            shift_part = shift_sum
        else:
            shift_part = np.expand_dims(
                shift_sum * self.shape[axis], 1 - self.scale_axis
            )
        return self.mat.sum(axis) + shift_part

    @abstractmethod
    def transpose(self):
        pass

    @abstractmethod
    def dot(self, other):
        pass

    @property
    def T(self):
        return self.transpose()

    def mean(self, axis: int = None):
        if axis is None:
            denominator = self.shape[0] * self.shape[1]
        else:
            denominator = self.shape[axis]
        return self.sum(axis) / denominator

    def __matmul__(self, other):
        """ Defines the behavior of 'self @ other'. """
        return self.dot(other)

    def __rmatmul__(self, other):
        """
        other @ self = (self.T @ other.T).T
        """
        return (self.T @ other.T).T

    # Higher priority than numpy arrays, so behavior for funcs like "@" defaults to the
    # behavior of ScaledMat
    __array_priority__ = 11


class ColScaledSpMat(ScaledMat):
    """
    Matrix with ij element equal to mat[i, j] + shift[j]
    """

    def __init__(self, mat: sps.spmatrix, shift: np.ndarray):
        super().__init__(mat, shift)

    @property  # type: ignore
    def scale_axis(self):
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


class RowScaledSpMat(ScaledMat):
    """
    Matrix with ij element equal to mat[i, j] + shift[i]
    """

    def __init__(self, mat: sps.spmatrix, shift: np.ndarray):
        super().__init__(mat, shift)

    @property
    def scale_axis(self):
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
        other_sum = other_mat.sum(0)
        if not sps.issparse(other_mat):
            # with numpy, sum is of shape (k,); with scipy it is of shape (1, k)
            other_sum = np.expand_dims(other_sum, 0)
        shift_part = self.shift.dot(other_sum)
        return mat_part + shift_part

    def power(self, p: float):
        return self.T.power(p).T
