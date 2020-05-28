from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from scipy import sparse as sps

from glm_benchmarks.matrix.matrix_base import MatrixBase


def csr_dot_categorical(mat_indices: np.ndarray, vec: np.ndarray) -> np.ndarray:
    return np.asarray(vec)[mat_indices]


def csc_dot_categorical(
    indices: np.ndarray, indptr: np.ndarray, vec: np.ndarray, n_rows: int
) -> np.ndarray:
    n_col = len(indptr) - 1
    res = np.zeros(n_rows, dtype=vec.dtype)

    for j in range(n_col):
        idx = indices[indptr[j] : indptr[j + 1]]
        res[idx] = vec[j]
    return res


def construct_csc(int_vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # TODO: figure out how csc to csr works
    # TODO: see if argsort helps
    use_csr = True
    if use_csr:
        mat = sps.csr_matrix(
            (np.ones(len(int_vec)), int_vec, np.arange(len(int_vec) + 1))
        ).tocsc()
        return mat.indices, mat.indptr

    n_cols = int_vec.max() + 1
    res: List[List[int]] = [[] for _ in range(n_cols)]
    for idx, elt in enumerate(int_vec):
        res[elt].append(idx)

    indices = np.zeros(len(int_vec), dtype=int)
    indptr = np.zeros(n_cols + 1, dtype=int)
    current_ptr = 0

    for i, list_ in enumerate(res):
        len_ = len(list_)
        next_ptr = current_ptr + len_
        indices[current_ptr:next_ptr] = list_
        indptr[i + 1] = next_ptr
        current_ptr = next_ptr

    return indices, indptr


def _get_cat(
    cat_vec: Union[List, np.ndarray]
) -> Tuple[pd.Categorical, Tuple[int, int]]:
    cat = pd.Categorical(cat_vec)
    shape = (len(cat_vec), len(cat.categories))
    return cat, shape


def _recover_orig(cat: pd.Categorical) -> np.ndarray:
    return cat.categories[cat.codes]


class CategoricalCSCMatrix(MatrixBase):
    def __init__(
        self,
        cat_vec: Union[List, np.ndarray, Tuple[pd.Categorical, np.ndarray, np.ndarray]],
    ):
        if isinstance(cat_vec, tuple):
            self.cat, self.indices, self.indptr = cat_vec
            self.shape = (len(self.cat), len(self.cat.categories))
        else:
            self.cat, self.shape = _get_cat(cat_vec)
            self.indices, self.indptr = construct_csc(self.cat.codes)
        self.x_csr = None

    def dot(self, other: Union[List, np.ndarray]) -> np.ndarray:
        return _dot(self, other)

    def _check_csr(self):
        if self.x_csr is None:
            # WRONG
            self.x_csr = self.to_cat_csr()

    def _matvec(self, other: np.ndarray) -> np.ndarray:
        return csc_dot_categorical(self.indices, self.indptr, other, self.shape[0])

    def getcol(self, i: int):
        res = np.zeros((self.shape[0], 1), dtype=int)
        res[self.indices[self.indptr[i] : self.indptr[i + 1]], 0] = 1
        return res

    def sandwich(self, d: np.ndarray) -> np.ndarray:
        pass

    def tocsc(self) -> sps.csc_matrix:
        return sps.csc_matrix(
            (np.ones(len(self.indices), dtype=int), self.indices, self.indptr)
        )

    def tocsr(self) -> sps.csr_matrix:
        return self.tocsc().tocsr()

    def toarray(self) -> np.ndarray:
        return self.tocsc().A

    def to_cat_csr(self):
        return CategoricalCSRMatrix(_recover_orig(self.cat))

    def astype(self, dtype, order="K", casting="unsafe", copy=True):
        pass

    def transpose(self):
        pass


class CategoricalCSRMatrix(MatrixBase):
    def __init__(self, cat_vec: Union[List, np.ndarray, pd.Categorical]):
        if isinstance(cat_vec, pd.Categorical):
            self.cat = cat_vec
            self.shape = (len(self.cat), len(self.cat.categories))
        else:
            self.cat, self.shape = _get_cat(cat_vec)
        self.indices = self.cat.codes
        self.dtype = self.indices.dtype
        self.x_csc = None

    def dot(self, other: Union[List, np.ndarray]) -> np.ndarray:
        return _dot(self, other)

    def _check_csc(self):
        if self.x_csc is None:
            self.x_csc = self.transpose()

    def _matvec(self, other: np.ndarray) -> np.ndarray:
        return csr_dot_categorical(self.indices, other)

    # TODO: best way to return this depends on the use case. See what that is
    # See how csr getcol works
    def getcol(self, i: int) -> np.ndarray:
        return (self.indices == i).astype(int)[:, None]

    def sandwich(self, d: np.ndarray) -> np.ndarray:
        pass

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
        return CategoricalCSCMatrix(self.indices)

    def astype(self, dtype, order="K", casting="unsafe", copy=True):
        """
        This method doesn't make a lot of sense since indices needs to be of int dtype,
        but it needs to be implemented.
        """
        return self


def _dot(
    mat: Union[CategoricalCSCMatrix, CategoricalCSRMatrix],
    other: Union[np.ndarray, List],
) -> np.ndarray:
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


if __name__ == "__main__":
    csc = CategoricalCSCMatrix([1, 0, 1])
