import numpy as np

from glm_benchmarks.scaled_spmat.mkl_sparse_matrix import MKLSparseMatrix
from glm_benchmarks.sklearn_fork.dense_glm_matrix import DenseGLMDataMatrix


class SplitMatrix:
    skip_sklearn_check = True

    def __init__(self, X, threshold):
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

    def sandwich(self, d):
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

    def standardize(self, weights, scale_predictors):
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

    def unstandardize(self, col_means, col_stds):
        self.X_dense_F = self.X_dense_F.unstandardize(
            col_means[0, self.dense_indices], col_stds[self.dense_indices]
        )
        self.X_sparse = self.X_sparse.unstandardize(
            col_means[0, self.sparse_indices], col_stds[self.sparse_indices]
        )
        return self

    def dot(self, v):
        if v.shape[0] != self.shape[1]:
            raise ValueError(f"shapes {self.shape} and {v.shape} not aligned")
        dense_out = self.X_dense_F.dot(v[self.dense_indices])
        sparse_out = self.X_sparse.dot(v[self.sparse_indices])
        return dense_out + sparse_out

    def __matmul__(self, v):
        return self.dot(v)

    def __rmatmul__(self, v):
        dense_component = self.X_dense_F.__rmatmul__(v)
        sparse_component = self.X_sparse.__rmatmul__(v)
        out_shape = list(dense_component.shape)
        out_shape[-1] = self.shape[1]
        out = np.empty(out_shape, dtype=v.dtype)
        out[..., self.dense_indices] = dense_component
        out[..., self.sparse_indices] = sparse_component
        return out

    __array_priority__ = 13
