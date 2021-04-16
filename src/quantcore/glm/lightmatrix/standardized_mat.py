from typing import List, Union

import numpy as np
from scipy import sparse as sps

from . import MatrixBase
from .util import _set_up_rows_or_cols, _setup_restrictions


class StandardizedMatrix:
    """
    Matrix with ij element equal to mat[i, j] + shift[0, j]
    """

    __array_priority__ = 11

    def __init__(
        self,
        mat: MatrixBase,
        shift: Union[np.ndarray, List],
        mult: Union[np.ndarray, List] = None,
    ):

        shift_arr = np.atleast_1d(np.squeeze(shift))
        expected_shape = (mat.shape[1],)
        if not isinstance(mat, MatrixBase):
            raise TypeError("mat should be an instance of a MatrixBase subclass.")
        if not shift_arr.shape == expected_shape:
            raise ValueError(
                f"""Expected shift to be able to conform to shape {expected_shape},
            but it has shape {np.asarray(shift).shape}"""
            )

        mult_arr = mult
        if mult_arr is not None:
            mult_arr = np.atleast_1d(np.squeeze(mult_arr))
            if not mult_arr.shape == expected_shape:
                raise ValueError(
                    f"""Expected mult to be able to conform to shape {expected_shape},
                but it has shape {np.asarray(mult).shape}"""
                )

        self.shift = shift_arr
        self.mult = mult_arr
        self.mat = mat
        self.shape = mat.shape
        self.ndim = mat.ndim
        self.dtype = mat.dtype

    def matvec(
        self,
        other_mat: Union[np.ndarray, List],
        cols: np.ndarray = None,
        out: np.ndarray = None,
    ) -> np.ndarray:
        """
        This function returns a dense output, so it is best geared for the
        matrix-vector case.
        """
        cols = _set_up_rows_or_cols(cols, self.shape[1])

        other_mat = np.asarray(other_mat)
        mult_other = other_mat
        if self.mult is not None:
            mult = self.mult
            # Avoiding an outer product by matching dimensions.
            for _ in range(len(other_mat.shape) - 1):
                mult = mult[:, np.newaxis]
            mult_other = mult * other_mat
        mat_part = self.mat.matvec(mult_other, cols, out=out)

        # Add shift part to mat_part
        shift_part = self.shift[cols].dot(other_mat[cols, ...])  # scalar
        mat_part += shift_part
        return mat_part

    def getcol(self, i: int):
        """
        Returns a StandardizedMatrix.

        >>> from scipy import sparse as sps
        >>> x = StandardizedMatrix(SparseMatrix(sps.eye(3).tocsc()), shift=[0, 1, -2])
        >>> col_1 = x.getcol(1)
        >>> isinstance(col_1, StandardizedMatrix)
        True
        >>> col_1.A
        array([[1.],
               [2.],
               [1.]])
        """
        mult = None
        if self.mult is not None:
            mult = [self.mult[i]]
        col = self.mat.getcol(i)
        return StandardizedMatrix(col, [self.shift[i]], mult)

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

        rows, cols = _setup_restrictions(self.shape, rows, cols)

        term1 = self.mat.sandwich(d, rows, cols)
        d_mat = self.mat.transpose_matvec(d, rows, cols)
        if self.mult is not None:
            d_mat *= self.mult[cols]
        term2 = np.outer(d_mat, self.shift[cols])
        term3_and_4 = np.outer(
            self.shift[cols], d_mat + self.shift[cols] * d[rows].sum()
        )
        res = term2 + term3_and_4
        if isinstance(term1, sps.dia_matrix):
            idx = np.arange(res.shape[0])
            to_add = term1.data[0, :]
            if self.mult is not None:
                to_add *= self.mult[cols] ** 2
            res[idx, idx] += to_add
        else:
            to_add = term1
            if self.mult is not None:
                to_add *= np.outer(self.mult[cols], self.mult[cols])
            res += to_add
        return res

    def unstandardize(self) -> MatrixBase:
        return self.mat

    def transpose_matvec(
        self,
        other: Union[np.ndarray, List],
        rows: np.ndarray = None,
        cols: np.ndarray = None,
        out: np.ndarray = None,
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

        With row and col restrictions:

        self.transpose_matvec(other, rows, cols)[i, j]
            = self.mat.transpose_matvec(other, rows, cols)[i, j]
              + (outer(self.shift, ones(N))[rows, cols] @ other[cols])
            = self.mat.transpose_matvec(other, rows, cols)[i, j]
              + shift[cols[i]] other.sum(0)[rows[j]
        """
        other = np.asarray(other)
        out_is_none = out is None
        out = self.mat.transpose_matvec(other, rows, cols, out=out)

        rows, cols = _setup_restrictions(self.shape, rows, cols)
        other_sum = np.sum(other[rows], 0)

        shift_part_tmp = np.outer(self.shift[cols], other_sum)
        output_shape = ((self.shape[1] if cols is None else len(cols)),) + out.shape[1:]
        shift_part = np.reshape(shift_part_tmp, output_shape)

        if self.mult is not None:
            mult = self.mult
            # Avoiding an outer product by matching dimensions.
            for _ in range(len(out.shape) - 1):
                mult = mult[:, np.newaxis]
            out *= mult[cols]

        if out_is_none:
            out += shift_part
        else:
            out[cols] += shift_part
        return out

    def __rmatmul__(self, other: Union[np.ndarray, List]) -> np.ndarray:
        """
        other @ X = (X.T @ other.T).T = X.transpose_matvec(other.T).T

        Parameters
        ----------
        other: array-like

        Returns
        -------
        array

        """
        if not hasattr(other, "T"):
            other = np.asarray(other)
        return self.transpose_matvec(other.T).T  # type: ignore

    def __matmul__(self, other):
        """ Defines the behavior of 'self @ other'. """
        return self.matvec(other)

    def toarray(self) -> np.ndarray:
        mat_part = self.mat.A
        if self.mult is not None:
            mat_part = self.mult[None, :] * mat_part
        return mat_part + self.shift[None, :]

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
        mult_part = self.mult
        if mult_part is not None:
            mult_part = np.atleast_1d(mult_part[col])

        if isinstance(row, int):
            out = mat_part.A
            if mult_part is not None:
                out = out * mult_part
            return out + shift_part

        return StandardizedMatrix(mat_part, np.atleast_1d(shift_part), mult_part)

    def __repr__(self):
        out = f"""StandardizedMat. Mat: {type(self.mat)} of shape {self.mat.shape}.
        Shift: {self.shift}
        Mult: {self.mult}
        """
        return out
