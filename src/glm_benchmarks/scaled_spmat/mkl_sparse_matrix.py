from typing import Union

import numpy as np
from sparse_dot_mkl import dot_product_mkl

from glm_benchmarks.sandwich.sandwich import sparse_sandwich


class MKLSparseMatrix:
    """
    A wrapper around a scipy.sparse matrix that will use MKL for sparse
    matrix-vector products and will use the fast sparse_sandwich function
    for sandwich products.
    """

    def __init__(self, X):
        while isinstance(X, MKLSparseMatrix):
            X = X.X
        self.X = X
        self.shape = X.shape
        self.ndim = X.ndim
        self.dtype = X.dtype

    @property
    def indptr(self):
        return self.X.indptr

    @property
    def indices(self):
        return self.X.indices

    @property
    def data(self):
        return self.X.data

    @property
    def A(self):
        return self.X.todense()

    def tocsc(self, *args, **kwargs):
        return MKLSparseMatrix(self.X.tocsc(*args, **kwargs))

    def tocsr(self, *args, **kwargs):
        return MKLSparseMatrix(self.X.tocsr(*args, **kwargs))

    def toarray(self):
        return self.X.toarray()

    @property
    def T(self):
        return self.transpose()

    def transpose(self):
        return MKLSparseMatrix(self.X.T)

    def getrow(self, i):
        return self.X.getrow(i)

    def getcol(self, j):
        return self.X.getcol(j)

    def sandwich(self, d):
        if not hasattr(self, "X_csr"):
            self.X_csr = self.X.tocsr()
        return sparse_sandwich(self.X.tocsc(copy=False), self.X_csr, d)

    def multiply(self, x):
        raise Exception("NOT IMPLEMENTED")

    def sum(self, axis: int = None) -> Union[np.ndarray, float]:
        return self.X.sum(axis)

    def dot(self, v):
        if len(v.shape) == 1:
            return dot_product_mkl(self.X, v)
        else:
            return self.X.dot(v)

    def __matmul__(self, other):
        return self.dot(other)

    def __rmatmul__(self, v):
        if len(v.shape) == 1:
            return dot_product_mkl(self.X.T, v)
        else:
            return v @ self.X

    __array_priority__ = 12

    def standardize(self, weights, scale_predictors):
        from glm_benchmarks.scaled_spmat.standardize import standardize, zero_center

        if scale_predictors:
            return standardize(self, weights=weights)
        else:
            X, col_means = zero_center(self, weights=weights)
            col_stds = np.ones(self.shape[1])
            return X, col_means, col_stds
