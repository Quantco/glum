from typing import List, Union

import numpy as np
from scipy import sparse as sps
from sklearn.preprocessing import OrdinalEncoder

from .matrix_base import MatrixBase


def csr_dot(mat: sps.csr_matrix, vec: np.ndarray) -> np.ndarray:
    n_row = mat.shape[0]
    res = np.zeros(n_row, dtype=np.result_type(mat.dtype, vec.dtype))
    for i in range(n_row):
        for j in range(mat.indptr[i], mat.indptr[i + 1]):
            res[i] += mat.data[j] * vec[mat.indices[j]]
    return res


def csr_dot_categorical(mat_indices: np.ndarray, vec: np.ndarray) -> np.ndarray:
    return np.asarray(vec)[mat_indices]


class CategoricalMatrix(MatrixBase):
    def __init__(self, cat: Union[List, np.ndarray]):
        cat = np.squeeze(np.asarray(cat))
        if cat.ndim > 1:
            raise ValueError
        self.enc = OrdinalEncoder().fit(cat[:, None])
        self.indices = np.squeeze(self.enc.transform(cat[:, None]).astype(np.int32))
        assert np.issubdtype(self.indices.dtype, np.signedinteger)
        n_categories = len(self.enc.categories_[0])
        self.shape = (len(cat), n_categories)
        self.dtype = self.indices.dtype

    def dot(self, vec: np.ndarray) -> np.ndarray:
        vec = np.asarray(vec)
        if not self.shape[1] == vec.shape[0]:
            raise ValueError(
                f"""Needed vec to have first dimension {self.shape[1]}, but vec is of
                shape {vec.shape}"""
            )

        if np.squeeze(vec).ndim > 1:
            raise NotImplementedError
        assert np.issubdtype(self.indices.dtype, np.signedinteger)
        return csr_dot_categorical(self.indices, vec)

    # TODO: sparse version?
    def getcol(self, i: int) -> np.ndarray:
        return (self.indices == i).astype(int)[:, None]

    def sandwich(self, d: np.ndarray) -> np.ndarray:
        pass

    def tocsr(self) -> sps.csr_matrix:
        return sps.csr_matrix(
            (
                np.ones(self.shape[0], dtype=int),
                self.indices,
                np.arange(self.shape[0] + 1, dtype=int),
            )
        )

    def toarray(self) -> np.ndarray:
        return self.tocsr().A

    def transpose(self):
        return self.tocsr().transpose()

    def astype(self, dtype, order="K", casting="unsafe", copy=True):
        """
        This method doesn't make a lot of sense since indices needs to be of int dtype,
        but it needs to be implemented.
        """
        return self
