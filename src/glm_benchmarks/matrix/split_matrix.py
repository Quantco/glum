import copy as copy_
from typing import Any, Tuple, Union

import numpy as np
from scipy import sparse as sps

from .dense_glm_matrix import DenseGLMDataMatrix
from .matrix_base import MatrixBase
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
        if i in self.dense_indices:
            idx = np.where(self.dense_indices == i)[0]
            return self.X_dense_F[:, idx]
        idx = np.where(self.sparse_indices == i)[0]
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

    def standardize(
        self, weights: np.ndarray, scale_predictors: bool
    ) -> Tuple[Any, np.ndarray, np.ndarray]:
        self.X_dense_F, dense_col_means, dense_col_stds = self.X_dense_F.standardize(
            weights, scale_predictors
        )
        self.X_sparse, sparse_col_means, sparse_col_stds = self.X_sparse.standardize(
            weights, scale_predictors
        )

        col_means = np.empty((1, self.shape[1]), dtype=self.dtype)
        col_means[0, self.dense_indices] = dense_col_means
        col_means[0, self.sparse_indices] = sparse_col_means

        col_stds = np.empty(self.shape[1], dtype=self.dtype)
        col_stds[self.dense_indices] = dense_col_stds
        col_stds[self.sparse_indices] = sparse_col_stds

        return self, col_means, col_stds

    def unstandardize(
        self, col_means: np.ndarray, col_stds: np.ndarray, scale_predictors
    ):
        self.X_dense_F = self.X_dense_F.unstandardize(
            col_means[0, self.dense_indices],
            col_stds[self.dense_indices],
            scale_predictors,
        )
        self.X_sparse = self.X_sparse.unstandardize(
            col_means[0, self.sparse_indices],
            col_stds[self.sparse_indices],
            scale_predictors,
        )
        return self

    def dot(self, v: np.ndarray) -> np.ndarray:
        v = np.asarray(v)
        if v.shape[0] != self.shape[1]:
            raise ValueError(f"shapes {self.shape} and {v.shape} not aligned")
        dense_out = self.X_dense_F.dot(v[self.dense_indices, ...])
        sparse_out = self.X_sparse.dot(v[self.sparse_indices, ...])
        return dense_out + sparse_out

    def __rmatmul__(self, v) -> np.ndarray:
        dense_component = self.X_dense_F.__rmatmul__(v)
        sparse_component = self.X_sparse.__rmatmul__(v)
        out_shape = list(dense_component.shape)
        out_shape[-1] = self.shape[1]
        out = np.empty(out_shape, dtype=v.dtype)
        out[..., self.dense_indices] = dense_component
        out[..., self.sparse_indices] = sparse_component
        return out

    def transpose(self):
        raise NotImplementedError("Oops, this library is not finished.")

    __array_priority__ = 13
