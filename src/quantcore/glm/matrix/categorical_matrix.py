from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import sparse as sps

from quantcore.glm.matrix.matrix_base import MatrixBase
from quantcore.glm.matrix.sandwich.categorical_sandwich import sandwich_categorical


def sandwich_old(indices: np.ndarray, indptr: np.ndarray, d: np.ndarray) -> np.ndarray:
    """
    Returns a 1d array. The sandwich output is a diagonal matrix with this array on
    the diagonal.
    """
    tmp = d[indices]
    res = np.zeros(len(indptr) - 1, dtype=d.dtype)
    for i in range(len(res)):
        res[i] = tmp[indptr[i] : indptr[i + 1]].sum()
    return res


class CategoricalCSRMatrix(MatrixBase):
    def __init__(self, cat_vec: Union[List, np.ndarray, pd.Categorical]):
        if isinstance(cat_vec, pd.Categorical):
            self.cat = cat_vec
        else:
            self.cat = pd.Categorical(cat_vec)

        self.shape = (len(self.cat), len(self.cat.categories))
        self.indices = self.cat.codes
        self.dtype = self.indices.dtype
        self.x_csc: Optional[Tuple[np.ndarray, np.ndarray]] = None

    def recover_orig(self) -> np.ndarray:
        return self.cat.categories[self.cat.codes]

    def dot(self, other: Union[List, np.ndarray]) -> np.ndarray:
        """
        When other is 1d:
        mat.dot(other)[i] = sum_j mat[i, j] other[j] = other[mat.indices[i]]

        When other is 2d:
        mat.dot(other)[i, k] = sum_j mat[i, j] other[j, k] = other[mat.indices[i], k]
        """
        other = np.asarray(other)
        if other.shape[0] != self.shape[1]:
            raise ValueError(
                f"""Needed other to have first dimension {self.shape[1]},
                but it has shape {other.shape}"""
            )
        return np.asarray(other)[self.indices, ...]

    def _check_csc(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.x_csc is None:
            csc = self.tocsr().tocsc()
            self.x_csc = (csc.indices, csc.indptr)
        return self.x_csc

    # TODO: best way to return this depends on the use case. See what that is
    # See how csr getcol works
    def getcol(self, i: int) -> np.ndarray:
        i %= self.shape[1]  # wrap-around indexing
        return (self.indices == i).astype(int)[:, None]

    def sandwich(self, d: Union[np.ndarray, List]) -> sps.spmatrix:
        d = np.asarray(d)
        indices, indptr = self._check_csc()
        res = sandwich_categorical(indices, indptr, d)
        return sps.diags(res)

    def sandwich_old(self, d: Union[np.ndarray, List]) -> sps.spmatrix:
        d = np.asarray(d)
        indices, indptr = self._check_csc()
        res = sandwich_old(indices, indptr, d)
        return sps.diags(res)

    def tocsr(self) -> sps.csr_matrix:
        return sps.csr_matrix(
            (
                # TODO: uint8
                np.ones(self.shape[0], dtype=int),
                self.indices,
                np.arange(self.shape[0] + 1, dtype=int),
            )
        )

    def toarray(self) -> np.ndarray:
        return self.tocsr().A

    def transpose_dot(self, vec: Union[np.ndarray, List]) -> np.ndarray:
        return self.tocsr().T.dot(vec)

    def astype(self, dtype, order="K", casting="unsafe", copy=True):
        """
        This method doesn't make a lot of sense since indices needs to be of int dtype,
        but it needs to be implemented.
        """
        self.dtype = dtype
        return self

    def get_col_stds(self, weights: np.ndarray, col_means: np.ndarray) -> np.ndarray:
        return np.sqrt(self.transpose_dot(weights) - col_means ** 2)

    def scale_cols_inplace(self, col_scaling: np.ndarray) -> None:
        raise NotImplementedError(
            """CategoricalMatrix does not currently support scaling columns."""
        )

    def __getitem__(self, item):
        if isinstance(item, tuple):
            row, col = item
            if not col == slice(None, None, None):
                raise IndexError("Only column indexing is supported.")
        else:
            row = item
        if isinstance(row, int):
            row = [row]
        return CategoricalCSRMatrix(self.cat[row])
