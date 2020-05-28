from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import sparse as sps

from glm_benchmarks.matrix.matrix_base import MatrixBase


def csr_dot_categorical(mat_indices: np.ndarray, vec: np.ndarray) -> np.ndarray:
    return np.asarray(vec)[mat_indices]


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
        return _dot(self, other)

    def _check_csc(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.x_csc is None:
            csc = self.tocsr().tocsc()
            self.x_csc = (csc.indices, csc.indptr)
        return self.x_csc

    def _matvec(self, other: np.ndarray) -> np.ndarray:
        return csr_dot_categorical(self.indices, other)

    # TODO: best way to return this depends on the use case. See what that is
    # See how csr getcol works
    def getcol(self, i: int) -> np.ndarray:
        return (self.indices == i).astype(int)[:, None]

    def sandwich(self, d: Union[np.ndarray, List]) -> np.ndarray:
        d = np.asarray(d)
        indices, indptr = self._check_csc()
        tmp = d[indices]
        res = np.zeros(self.shape[1], dtype=d.dtype)
        for i in range(len(res)):
            res[i] = tmp[indptr[i] : indptr[i + 1]].sum()
        return np.diag(res)

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

    def transpose(self):
        return self.tocsr().T

    def astype(self, dtype, order="K", casting="unsafe", copy=True):
        """
        This method doesn't make a lot of sense since indices needs to be of int dtype,
        but it needs to be implemented.
        """
        return self


def _dot(mat: CategoricalCSRMatrix, other: Union[np.ndarray, List],) -> np.ndarray:
    other = np.asarray(other)
    original_ndim = other.ndim
    if original_ndim > 2:
        raise NotImplementedError
    other = np.squeeze(other)
    if other.ndim > 1:
        raise NotImplementedError
    if other.shape[0] != mat.shape[1]:
        raise ValueError(
            f"""Needed vec to have first dimension {mat.shape[1]}, but vec is of
        shape {other.shape[0]}"""
        )
    res = mat._matvec(other)
    if original_ndim == 1:
        return res
    return res[:, np.newaxis]
