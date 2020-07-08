from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import sparse as sps

from .ext.categorical import dot, sandwich_categorical, transpose_dot
from .ext.split import sandwich_cat_cat, sandwich_cat_dense
from .matrix_base import MatrixBase


def _none_to_slice(arr: Optional[np.ndarray], n: int) -> Union[slice, np.ndarray]:
    if arr is None or len(arr) == n:
        return slice(None, None, None)
    return arr


class CategoricalMatrix(MatrixBase):
    def __init__(
        self,
        cat_vec: Union[List, np.ndarray, pd.Categorical],
        dtype: np.dtype = np.dtype("float64"),
    ):
        """
        Constructs an object that behaves like cat_vec with one-hot encoding, but
        with more memory efficiency and speed.
        ---
        cat_vec: array-like vector of categorical data.
        dtype:
        """
        if isinstance(cat_vec, pd.Categorical):
            self.cat = cat_vec
        else:
            self.cat = pd.Categorical(cat_vec)

        self.shape = (len(self.cat), len(self.cat.categories))
        self.indices = self.cat.codes.astype(np.int32)
        self.x_csc: Optional[Tuple[Optional[np.ndarray], np.ndarray, np.ndarray]] = None
        self.dtype = dtype

    def recover_orig(self) -> np.ndarray:
        """

        Returns
        -------
        1d numpy array with same data as what was initially fed to __init__.
        Test: matrix/test_categorical_matrix::test_recover_orig
        """
        return self.cat.categories[self.cat.codes]

    def dot(
        self,
        other: Union[List, np.ndarray],
        rows: np.ndarray = None,
        cols: np.ndarray = None,
    ) -> np.ndarray:
        # taking 14% of time
        """
        When other is 1d:
        mat.dot(other)[i] = sum_j mat[i, j] other[j]
                          = other[mat.indices[i]]

        When other is 2d:
        mat.dot(other)[i, k] = sum_j mat[i, j] other[j, k]
                            = other[mat.indices[i], k]

        The rows and cols parameters allow restricting to a subset of the
        matrix without making a copy.

        Test:
        matrix/test_matrices::test_dot
        """
        other = np.asarray(other)
        if other.ndim > 1:
            raise NotImplementedError(
                """CategoricalMatrix.dot is only implemented for 1d arrays."""
            )
        if other.shape[0] != self.shape[1]:
            raise ValueError(
                f"""Needed other to have first dimension {self.shape[1]},
                but it has shape {other.shape}"""
            )

        if cols is None:
            other_m = other
        else:
            col_mult = np.zeros(len(self.cat.categories), dtype=self.dtype)
            col_mult[cols] = 1.0
            other_m = other * col_mult

        if rows is None or len(rows) == self.shape[0]:
            if np.issubdtype(other_m.dtype, np.signedinteger):
                other_m = other_m.astype(float)
                res = dot(self.indices, other_m, self.shape[0], other_m.dtype)
                return res.astype(int)
            return dot(self.indices, other_m, self.shape[0], other.dtype)

        return other_m[self.indices[rows]]

    def transpose_dot(
        self,
        vec: Union[np.ndarray, List],
        rows: np.ndarray = None,
        cols: np.ndarray = None,
    ) -> np.ndarray:
        # taking 16% of time
        """
        Perform: self[rows, cols].T @ vec

        The rows and cols parameters allow restricting to a subset of the
        matrix without making a copy.

        Test: tests/test_matrices::test_transpose_dot
        """
        # TODO: write a function that doesn't reference the data
        # TODO: this should look more like the cat_cat_sandwich
        vec = np.asarray(vec)
        if vec.ndim > 1:
            raise NotImplementedError(
                "CategoricalMatrix.transpose_dot is only implemented for 1d arrays."
            )

        res = transpose_dot(self.indices, vec, self.shape[1], vec.dtype, rows)
        if cols is not None and len(cols) < self.shape[1]:
            res = res[cols]
        return res

    def sandwich(
        self,
        d: Union[np.ndarray, List],
        rows: np.ndarray = None,
        cols: np.ndarray = None,
    ) -> sps.dia_matrix:
        # 3%
        """
        sandwich(self, d)[i, j] = (self.T @ diag(d) @ self)[i, j]
            = sum_k (self[k, i] (diag(d) @ self)[k, j])
            = sum_k self[k, i] sum_m diag(d)[k, m] self[m, j]
            = sum_k self[k, i] d[k] self[k, j]
            = 0 if i != j
        sandwich(self, d)[i, i] = sum_k self[k, i] ** 2 * d(k)

        The rows and cols parameters allow restricting to a subset of the
        matrix without making a copy.
        """
        d = np.asarray(d)
        if rows is None:
            rows = np.arange(self.shape[0], dtype=np.int32)
        res_diag = sandwich_categorical(self.indices, d, rows, d.dtype, self.shape[1])
        if cols is not None and len(cols) < self.shape[1]:
            res_diag = res_diag[cols]
        return sps.diags(res_diag)

    def cross_sandwich(
        self,
        other: MatrixBase,
        d: Union[np.ndarray, List],
        rows: Optional[np.ndarray] = None,
        L_cols: Optional[np.ndarray] = None,
        R_cols: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        # 19%
        if isinstance(other, np.ndarray):
            return self._cross_dense(other, d, rows, L_cols, R_cols)
        if isinstance(other, sps.csc_matrix):
            return self._cross_sparse(other, d, rows, L_cols, R_cols)
        if isinstance(other, CategoricalMatrix):
            return self._cross_categorical(other, d, rows, L_cols, R_cols)
        raise TypeError

    def _check_csc(self, force_reset=False) -> Tuple[None, np.ndarray, np.ndarray]:
        if self.x_csc is None or force_reset:
            # Currently taking up a lot of time
            csc = self.tocsr().tocsc()
            self.x_csc = (None, csc.indices, csc.indptr)
        else:
            assert True
        return self.x_csc

    def tocsc(self):
        _, indices, indptr = self._check_csc()
        data = np.ones(self.shape[0])
        return sps.csc_matrix((data, indices, indptr))

    # TODO: best way to return this depends on the use case. See what that is
    # See how csr getcol works
    def getcol(self, i: int) -> sps.csc_matrix:
        i %= self.shape[1]  # wrap-around indexing
        col_i = sps.csc_matrix((self.indices == i).astype(int)[:, None])
        return col_i

    def tocsr(self) -> sps.csr_matrix:
        # TODO: data should be uint8
        data = np.ones(self.shape[0], dtype=int)

        return sps.csr_matrix(
            (data, self.indices, np.arange(self.shape[0] + 1, dtype=int)),
            shape=self.shape,
        )

    def toarray(self) -> np.ndarray:
        return self.tocsr().A

    def astype(self, dtype, order="K", casting="unsafe", copy=True):
        """
        This method doesn't make a lot of sense since indices needs to be of int dtype,
        but it needs to be implemented.
        """
        self.dtype = dtype
        return self

    def get_col_stds(self, weights: np.ndarray, col_means: np.ndarray) -> np.ndarray:
        mean = self.transpose_dot(weights)
        return np.sqrt(mean - col_means ** 2)

    def __getitem__(self, item):
        if isinstance(item, tuple):
            row, col = item
            if not (isinstance(col, slice) and col == slice(None, None, None)):
                raise IndexError("Only column indexing is supported.")
        else:
            row = item
        if isinstance(row, int):
            row = [row]
        return CategoricalMatrix(self.cat[row])

    def _cross_dense(
        self,
        other: np.ndarray,
        d: np.ndarray,
        rows: Optional[np.ndarray],
        L_cols: Optional[np.ndarray],
        R_cols: Optional[np.ndarray],
    ) -> np.ndarray:

        if other.flags["C_CONTIGUOUS"]:
            is_c_contiguous = True
        elif other.flags["F_CONTIGUOUS"]:
            is_c_contiguous = False
        else:
            raise ValueError(
                "Input array needs to be either C-contiguous or F-contiguous."
            )

        if rows is None:
            rows = np.arange(self.shape[0], dtype=np.int32)
        if R_cols is None:
            R_cols = np.arange(other.shape[1], dtype=np.int32)

        res = sandwich_cat_dense(
            self.indices, self.shape[1], d, other, rows, R_cols, is_c_contiguous
        )

        res = res[_none_to_slice(L_cols, self.shape[1]), :]
        return res

    def _cross_categorical(
        self,
        other,
        d: np.ndarray,
        rows: Optional[np.ndarray],
        L_cols: Optional[np.ndarray],
        R_cols: Optional[np.ndarray],
    ) -> np.ndarray:
        if not isinstance(other, CategoricalMatrix):
            raise TypeError

        i_indices = self.indices
        j_indices = other.indices
        if rows is None:
            rows = np.arange(self.shape[0], dtype=np.int32)

        res = sandwich_cat_cat(
            i_indices, j_indices, self.shape[1], other.shape[1], d, rows
        )

        L_cols = _none_to_slice(L_cols, self.shape[1])
        R_cols = _none_to_slice(R_cols, other.shape[1])
        res = res[L_cols, :][:, R_cols]
        return res

    def _cross_sparse(
        self,
        other: sps.csc_matrix,
        d: np.ndarray,
        rows: Optional[np.ndarray],
        L_cols: Optional[np.ndarray],
        R_cols: Optional[np.ndarray],
    ) -> np.ndarray:

        term_1 = self.tocsr()
        term_1.data = term_1.data * d

        rows = _none_to_slice(rows, self.shape[0])
        L_cols = _none_to_slice(L_cols, self.shape[1])
        term_1 = term_1[rows, :][:, L_cols]

        res = term_1.T.dot(other[rows, :][:, _none_to_slice(R_cols, other.shape[1])]).A

        return res
