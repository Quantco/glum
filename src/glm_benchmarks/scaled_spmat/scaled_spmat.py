from typing import Union

import numpy as np
from scipy import sparse as sps


class ScaledMat:
    """
    Base calss for ColScaledSpMat and RowScaledSpMat. Do not use.
    """

    def __init__(self, mat: sps.spmatrix, shift: np.ndarray, mult: np.ndarray = None):
        if not sps.issparse(mat):
            raise ValueError("mat should be a sparse matrix.")
        if mult is None:
            self.shift = shift
        else:
            self.shift = shift * mult
        self.shape = mat.shape


class ColScaledSpMat(ScaledMat):
    """
    Matrix with ij element equal to (mat[i, j] + shift[j]) * mult[j]
    """

    def __init__(self, mat: sps.spmatrix, shift: np.ndarray, mult: np.ndarray = None):
        if not shift.shape == (mat.shape[1],):
            raise ValueError(
                f"""Shifter is of shape {shift.shape}; expected
            {(mat.shape[1],)}"""
            )
        if mult is not None and not mult.shape == (mat.shape[1],):
            raise ValueError
        super().__init__(mat, shift, mult)

        if mult is None:
            self.mat = mat
        else:
            self.mat = mat.multiply(mult)

    @property
    def T(self):
        return RowScaledSpMat(self.mat.T, self.shift)

    @property
    def A(self) -> np.ndarray:
        return self.mat.A + self.shift[None, :]

    def dot(self, other_mat: Union[sps.spmatrix, np.ndarray]):
        """
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
        # TODO: check if tocsc already returns a copy
        tmp = self.mat.copy().tocsc()
        # Update data, looping over columns
        for j in range(self.shape[1]):
            start = tmp.indptr[j]
            end = tmp.indptr[j + 1]
            data = tmp.data[start:end]
            tmp.data[start:end] = (data + self.shift[j]) ** p - self.shift[j] ** p
        return ColScaledSpMat(tmp, self.shift ** p)


class RowScaledSpMat(ScaledMat):
    """
    Matrix with ij element equal to (mat[i, j] + shift[i]) * mult[i]
    """

    def __init__(self, mat: sps.spmatrix, shift: np.ndarray, mult: np.ndarray = None):
        super().__init__(mat, shift, mult)
        if mult is None:
            self.mat = mat
        else:
            self.mat = mat.multiply(mult[:, None])

    @property
    def T(self) -> ColScaledSpMat:
        return ColScaledSpMat(self.mat.T, self.shift)

    @property
    def A(self) -> np.ndarray:
        return self.mat.A + self.shift[:, None]

    def dot(self, other_mat: Union[sps.spmatrix, np.ndarray]):
        """
        if M = RowScaledSpMat(A, b),
        M.dot(x)[i, j] = sum_k (M[i, k] * b[k, j] + shift[i] * b[k, j])
        = M.dot(b)[i, j] + shift[i] * b[:, j].sum()
        """
        mat_part = self.mat.dot(other_mat)
        if other_mat.ndim == 1:
            shifter = self.shift * other_mat.sum()
        else:
            shifter = self.shift[:, None] * other_mat.sum(0)[None, :]
        return mat_part + shifter

    def power(self, p: float):
        pass
