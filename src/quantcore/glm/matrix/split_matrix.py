from typing import List, Tuple, Union

import numpy as np
from scipy import sparse as sps

from . import MatrixBase
from .dense_glm_matrix import DenseGLMDataMatrix
from .mkl_sparse_matrix import MKLSparseMatrix


class SplitMatrix(MatrixBase):
    def __init__(
        self,
        arg1: Union[
            sps.csc_matrix, Tuple[np.ndarray, sps.csc_matrix, np.ndarray, np.ndarray]
        ],
        threshold: float = 0.1,
    ):
        if isinstance(arg1, tuple):
            dense, sparse, self.dense_indices, self.sparse_indices = arg1
            if not dense.shape[0] == sparse.shape[0]:
                raise ValueError(
                    f"""
                    X_dense_F and X_sparse should have the same length,
                    but X_dense_F has shape {dense.shape} and X_sparse has shape {sparse.shape}.
            """
                )
            if not dense.shape[1] == len(self.dense_indices):
                raise ValueError(
                    f"""dense_indices should have length X_dense_F.shape[1],
                but dense_indices has shape {self.dense_indices.shape} and X_dense_F
                has shape {dense.shape}"""
                )
            if not sparse.shape[1] == len(self.sparse_indices):
                raise ValueError(
                    f"""sparse_indices should have length X_sparse.shape[1],
                but sparse_indices has shape {self.sparse_indices.shape} and X_sparse
                has shape {sparse.shape}"""
                )
            self.X_dense_F = DenseGLMDataMatrix(dense)
            self.X_sparse = MKLSparseMatrix(sparse)
        else:
            if not isinstance(arg1, sps.csc_matrix):
                raise TypeError(
                    "X must be of type scipy.sparse.csc_matrix or matrix.MKLSparseMatrix"
                )
            if not 0 <= threshold <= 1:
                raise ValueError("Threshold must be between 0 and 1.")
            densities = np.diff(arg1.indptr) / arg1.shape[0]
            self.dense_indices = np.where(densities > threshold)[0]
            self.sparse_indices = np.setdiff1d(
                np.arange(densities.shape[0]), self.dense_indices
            )

            self.X_dense_F = DenseGLMDataMatrix(
                np.asfortranarray(arg1.toarray()[:, self.dense_indices])
            )
            self.X_sparse = MKLSparseMatrix(arg1[:, self.sparse_indices])

        self.dtype = self.X_sparse.dtype
        self.shape = (
            self.X_dense_F.shape[0],
            len(self.dense_indices) + len(self.sparse_indices),
        )

    def astype(self, dtype, order="K", casting="unsafe", copy=True):
        dense = self.X_dense_F.astype(dtype, order, casting, copy=copy)
        sparse = self.X_sparse.astype(dtype=dtype, casting=casting, copy=copy)
        return SplitMatrix((dense, sparse, self.dense_indices, self.sparse_indices,))

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

    def sandwich(
        self, d: np.ndarray, rows: np.ndarray = None, cols: np.ndarray = None
    ) -> np.ndarray:
        d = np.asarray(d)
        if rows is None:
            rows = np.arange(self.shape[0], dtype=np.int32)
        if cols is None:
            cols = np.arange(self.shape[1], dtype=np.int32)

        (
            dense_cols,
            dense_cols_indices,
            sparse_cols,
            sparse_cols_indices,
        ) = self.split_col_subsets(cols)

        out = np.empty((cols.shape[0], cols.shape[0]))
        if sparse_cols_indices.shape[0] > 0:
            SS = self.X_sparse.sandwich(d, rows, sparse_cols)
            out[np.ix_(sparse_cols_indices, sparse_cols_indices)] = SS
        if dense_cols_indices.shape[0] > 0:
            DD = self.X_dense_F.sandwich(d, rows, dense_cols)
            out[np.ix_(dense_cols_indices, dense_cols_indices)] = DD
            if sparse_cols_indices.shape[0] > 0:
                DS = self.X_sparse[rows, :].sandwich_dense(
                    self.X_dense_F[rows, :], d[rows]
                )[np.ix_(sparse_cols, dense_cols)]
                out[np.ix_(sparse_cols_indices, dense_cols_indices)] = DS
                out[np.ix_(dense_cols_indices, sparse_cols_indices)] = DS.T
        return out

    def split_col_subsets(self, cols):
        next_dense_idx = 0
        next_sparse_idx = 0
        dense_cols_indices = []
        dense_cols = []
        sparse_cols_indices = []
        sparse_cols = []
        for i in range(cols.shape[0]):
            while (
                next_dense_idx < self.dense_indices.shape[0]
                and self.dense_indices[next_dense_idx] < cols[i]
            ):
                next_dense_idx += 1
            while (
                next_sparse_idx < self.sparse_indices.shape[0]
                and self.sparse_indices[next_sparse_idx] < cols[i]
            ):
                next_sparse_idx += 1

            if (
                next_dense_idx < self.dense_indices.shape[0]
                and self.dense_indices[next_dense_idx] == cols[i]
            ):
                dense_cols_indices.append(i)
                dense_cols.append(next_dense_idx)
                next_dense_idx += 1
            else:
                is_sparse = (
                    next_sparse_idx < self.sparse_indices.shape[0]
                    and self.sparse_indices[next_sparse_idx] == cols[i]
                )
                assert is_sparse
                sparse_cols_indices.append(i)
                sparse_cols.append(next_sparse_idx)
                next_sparse_idx += 1
        return (
            np.array(dense_cols, dtype=np.int32),
            np.array(dense_cols_indices),
            np.array(sparse_cols, dtype=np.int32),
            np.array(sparse_cols_indices),
        )

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

    def dot(
        self, v: np.ndarray, rows: np.ndarray = None, cols: np.ndarray = None
    ) -> np.ndarray:
        assert not isinstance(v, sps.spmatrix)
        v = np.asarray(v)
        if v.shape[0] != self.shape[1]:
            raise ValueError(f"shapes {self.shape} and {v.shape} not aligned")

        if rows is None:
            rows = np.arange(self.shape[0], dtype=np.int32)
        if cols is None:
            cols = np.arange(self.shape[1], dtype=np.int32)
        (
            dense_cols,
            dense_cols_indices,
            sparse_cols,
            sparse_cols_indices,
        ) = self.split_col_subsets(cols)

        dense_out = self.X_dense_F.dot(v[self.dense_indices, ...], rows, dense_cols)
        sparse_out = self.X_sparse.dot(v[self.sparse_indices, ...], rows, sparse_cols)
        return dense_out + sparse_out

    def transpose_dot(
        self,
        vec: Union[np.ndarray, List],
        rows: np.ndarray = None,
        cols: np.ndarray = None,
    ) -> np.ndarray:
        """
        self.T.dot(vec)[i] = sum_k self[k, i] vec[k]
        = sum_{k in self.dense_indices} self[k, i] vec[k] +
          sum_{k in self.sparse_indices} self[k, i] vec[k]
        = self.X_dense.T.dot(vec) + self.X_sparse.T.dot(vec)
        """

        vec = np.asarray(vec)
        if rows is None:
            rows = np.arange(self.shape[0], dtype=np.int32)
        if cols is None:
            cols = np.arange(self.shape[1], dtype=np.int32)
        (
            dense_cols,
            dense_cols_indices,
            sparse_cols,
            sparse_cols_indices,
        ) = self.split_col_subsets(cols)

        vec = np.asarray(vec)
        out_shape = [cols.shape[0]] + list(vec.shape[1:])
        out = np.empty(out_shape, dtype=vec.dtype)
        if dense_cols_indices.shape[0] > 0:
            dense_component = self.X_dense_F.transpose_dot(vec, rows, dense_cols)
            out[dense_cols_indices, ...] = dense_component
        if sparse_cols_indices.shape[0] > 0:
            sparse_component = self.X_sparse.transpose_dot(vec, rows, sparse_cols)
            out[sparse_cols_indices, ...] = sparse_component
        return out

    def scale_cols_inplace(self, col_scaling: np.ndarray):
        self.X_sparse.scale_cols_inplace(col_scaling[self.sparse_indices])
        self.X_dense_F.scale_cols_inplace(col_scaling[self.dense_indices])

    def __getitem__(self, key):
        if isinstance(key, tuple):
            row, col = key
        else:
            row = key
            col = slice(None, None, None)  # all columns

        if col == slice(None, None, None):
            if isinstance(row, int):
                row = [row]

            return SplitMatrix(
                (
                    self.X_dense_F[row, :],
                    self.X_sparse[row, :],
                    self.dense_indices,
                    self.sparse_indices,
                )
            )
        else:
            raise NotImplementedError(
                f"Only row indexing is supported. Index passed was {key}."
            )

    __array_priority__ = 13
