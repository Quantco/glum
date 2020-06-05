import copy as copy_
from typing import List, Union

import numpy as np
from scipy import sparse as sps

from . import MatrixBase
from .dense_glm_matrix import DenseGLMDataMatrix
from .mkl_sparse_matrix import MKLSparseMatrix


class SplitMatrix(MatrixBase):
    def __init__(
        self, X: Union[sps.csc_matrix, MKLSparseMatrix], threshold: float = 0.1
    ):
        if not isinstance(X, sps.csc_matrix) and not isinstance(X, MKLSparseMatrix):
            raise TypeError(
                "X must be of type scipy.sparse.csc_matrix or matrix.MKLSparseMatrix"
            )
        self.shape = X.shape
        self.threshold = threshold
        self.dtype = X.dtype

        densities = np.diff(X.indptr) / X.shape[0]
        self.dense_indices = np.where(densities > threshold)[0]
        self.sparse_indices = np.setdiff1d(
            np.arange(densities.shape[0]), self.dense_indices
        )

        self.X_dense_F = DenseGLMDataMatrix(
            np.asfortranarray(X.toarray()[:, self.dense_indices])
        )
        self.X_sparse = MKLSparseMatrix(X[:, self.sparse_indices])

    def astype(self, dtype, order="K", casting="unsafe", copy=True):
        if copy:
            new = copy_.copy(self)
            new.X_dense_F = self.X_dense_F.astype(dtype, order, casting, copy=copy)
            new.X_sparse = self.X_sparse.astype(dtype, casting, copy)
            new.dtype = new.X_dense_F.dtype
            return new
        self.X_dense_F = self.X_dense_F.astype(dtype, order, casting, copy=copy)
        self.X_sparse = self.X_sparse.astype(dtype, casting, copy)
        self.dtype = self.X_dense_F.dtype
        return self

    def toarray(self) -> np.ndarray:
        out = np.empty(self.shape)
        out[:, self.dense_indices] = self.X_dense_F
        out[:, self.sparse_indices] = self.X_sparse.A
        return out

    def getcol(self, i: int) -> Union[np.ndarray, sps.csr_matrix]:
        # wrap-around indexing
        i %= self.shape[1]
        if i in self.dense_indices:
            idx = np.where(self.dense_indices == i)[0][0]
            return self.X_dense_F.getcol(idx)
        idx = np.where(self.sparse_indices == i)[0][0]
        return self.X_sparse.getcol(idx)

    def sandwich(self, d: np.ndarray) -> np.ndarray:
        out = np.empty((self.shape[1], self.shape[1]))
        if self.X_sparse.shape[1] > 0:
            SS = self.X_sparse.sandwich(d)
            out[np.ix_(self.sparse_indices, self.sparse_indices)] = SS
        if self.X_dense_F.shape[1] > 0:
            DD = self.X_dense_F.sandwich(d)
            out[np.ix_(self.dense_indices, self.dense_indices)] = DD
            if self.X_sparse.shape[1] > 0:
                DS = self.X_sparse.sandwich_dense(self.X_dense_F, d)
                out[np.ix_(self.sparse_indices, self.dense_indices)] = DS
                out[np.ix_(self.dense_indices, self.sparse_indices)] = DS.T
        return out

    def get_col_means(self, weights: np.ndarray) -> np.ndarray:
        col_means = np.empty(self.shape[1], dtype=self.dtype)
        col_means[self.dense_indices] = self.X_dense_F.get_col_means(weights)
        col_means[self.sparse_indices] = self.X_sparse.get_col_means(weights)
        return col_means

    def get_col_stds(self, weights: np.ndarray, col_means: np.ndarray) -> np.ndarray:
        dense_col_stds = self.X_dense_F.get_col_stds(
            weights, col_means[self.dense_indices]
        )
        sparse_col_stds = self.X_sparse.get_col_stds(
            weights, col_means[self.sparse_indices]
        )

        col_stds = np.empty(self.shape[1], dtype=self.dtype)
        col_stds[self.dense_indices] = dense_col_stds
        col_stds[self.sparse_indices] = sparse_col_stds
        return col_stds

    def dot(self, v: np.ndarray) -> np.ndarray:
        assert not isinstance(v, sps.spmatrix)
        v = np.asarray(v)
        if v.shape[0] != self.shape[1]:
            raise ValueError(f"shapes {self.shape} and {v.shape} not aligned")
        dense_out = self.X_dense_F.dot(v[self.dense_indices, ...])
        sparse_out = self.X_sparse.dot(v[self.sparse_indices, ...])
        return dense_out + sparse_out

    def transpose_dot(self, vec: Union[np.ndarray, List]) -> np.ndarray:
        """
        self.T.dot(vec)[i] = sum_k self[k, i] vec[k]
        = sum_{k in self.dense_indices} self[k, i] vec[k] +
          sum_{k in self.sparse_indices} self[k, i] vec[k]
        = self.X_dense.T.dot(vec) + self.X_sparse.T.dot(vec)
        """
        vec = np.asarray(vec)
        dense_component = self.X_dense_F.transpose_dot(vec)
        sparse_component = self.X_sparse.transpose_dot(vec)
        out_shape = list(dense_component.shape)
        out_shape[0] = self.shape[1]
        out = np.empty(out_shape, dtype=vec.dtype)
        out[self.dense_indices, ...] = dense_component
        out[self.sparse_indices, ...] = sparse_component
        return out
        # return self.__rmatmul__(np.transpose(vec)).T

    def scale_cols_inplace(self, col_scaling: np.ndarray):
        self.X_sparse.scale_cols_inplace(col_scaling[self.sparse_indices])
        self.X_dense_F.scale_cols_inplace(col_scaling[self.dense_indices])

    __array_priority__ = 13
