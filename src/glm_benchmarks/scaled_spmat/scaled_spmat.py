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

        if not np.squeeze(shift).shape == (mat.shape[self.scale_axis],):
            raise ValueError(
                f"""Shifter is of shape {shift.shape}; expected
            {(mat.shape[self.scale_axis],)}"""
            )

        self.mat = mat
        self.shift = np.squeeze(shift)
        self.shape = mat.shape

    @property  # type: ignore
    @abstractclassmethod
    def scale_axis(self) -> int:
        return 0

    def todense(self) -> np.ndarray:
        return self.mat.A + np.expand_dims(self.shift, 1 - self.scale_axis)

    @property
    def A(self):
        return self.todense()

    def multiply(self, other: Union[np.ndarray, float]):
        """
        If other_mat is a matrix, result is potentially dense. Not supported.
        If other_mat is a column vector,

        X.multiply(Y)[i, j] = X[i, j] * Y[j]
        = (X.mat[i, j] + x.shift[j]) * Y[j]
        = ColScaledSpMat(X.mat.multiply(Y), x.shift * Y)
        """
        if not np.isscalar(other):
            other = np.squeeze(other)
            if other.ndim > 1:
                raise NotImplementedError(
                    """Elementwise multiplication by a >1d array is
                not supported because the result would be dense."""
                )
            other = np.expand_dims(other, 1 - self.scale_axis)

        mat_part = self.mat.multiply(other)
        shift_part = self.shift * np.squeeze(other)
        return type(self)(mat_part, shift_part)

    def sum(self, axis: int = None) -> Union[np.ndarray, float]:
        """
        For col case:
        axis = 0:
            sum_i (mat[i, j] + shift[j]) = sum_i mat[i, j] + shift[j] * mat.shape[0]
            = mat.sum(0) + shift * mat.shape[0]
        axis = 1:
            sum_j (mat[i, j] + shift[j]) = mat.sum(1) + shift.sum()
        axis=None: mat.sum(None) + shift.sum(None)

        Row case is symmetric.
        """
        mat_part = np.squeeze(np.asarray(self.mat.sum(axis=axis)))

        if axis == self.scale_axis:
            shift_part = self.shift.sum()
        else:
            shift_part = (
                np.expand_dims(self.shift, 1 - self.scale_axis).sum(axis=axis)
                * self.shape[1 - self.scale_axis]
            )
        return mat_part + shift_part

    @abstractmethod
    def transpose(self):
        pass

    @property
    def T(self):
        return self.transpose()

    def mean(self, axis: int = None):
        sum_ = self.sum(axis)
        if axis is None:
            return sum_ / (self.shape[0] * self.shape[1])
        return sum_ / self.shape[axis]


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
        return RowScaledSpMat(self.mat.T, self.shift)

    def dot(self, other_mat: Union[sps.spmatrix, np.ndarray]):
        """
        self.mat.dot(x)[i, j] = self.mat.dot(x)[i, j] + sum_k self.shift[k] * x[k, j]
        = self.mat.dot(x)[i, j] + x.dot(self.shift)[j]
        = ColScaledSpMat(self.mat.dot(x), x.dot(self.shift))[i, j]

        If other_mat is a 1d numpy array, return a 1d numpy array.
        If other_mat is a >1d ndarray, return a numpy ndarray (because
        the dot product of a sparse matrix and dense matrix is dense).
        If other_mat is a sparse matrix, return a ColScaledSpMat.
        """
        mat_part = self.mat.dot(other_mat)
        shifter = other_mat.T.dot(self.shift)

        if other_mat.ndim == 1:
            return mat_part + shifter
        if isinstance(other_mat, np.ndarray):
            return mat_part + shifter[None, :]
        return ColScaledSpMat(mat_part, shifter)

    def power(self, p: float):
        tmp = self.mat.tocsc(copy=True)
        shift_power = self.shift ** p
        # Update data, looping over columns
        for j in range(self.shape[1]):
            start, end = tmp.indptr[j : j + 2]
            data = tmp.data[start:end]
            tmp.data[start:end] = (data + self.shift[j]) ** p - shift_power[j]
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
        return ColScaledSpMat(self.mat.T, self.shift)

    def dot(self, other_mat: Union[sps.spmatrix, np.ndarray]) -> np.ndarray:
        """
        if M = RowScaledSpMat(A, b),
        M.dot(x)[i, j] = sum_k (A[i, k] * b[k, j] + shift[i] * b[k, j])
        = A.dot(b)[i, j] + shift[i] * b[:, j].sum()

        If x is >1d, this will generate a dense matrix that cannot be represented as
        a RowScaledSpMat or ColScaledSpMat since it involves an outer product of two
        vectors. Therefore, this will return a dense matrix.

        If x is a vector, the output is a vector, so this will also return
        a dense matrix.
        """
        mat_part = self.mat.dot(other_mat)
        if other_mat.ndim == 1:
            shifter = self.shift * other_mat.sum()
        else:
            shifter = self.shift[:, None] * other_mat.sum(0)[None, :]
        return mat_part + shifter

    def power(self, p: float):
        return self.T.power(p).T
