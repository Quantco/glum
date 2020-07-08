import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
from scipy import sparse as sps

from . import MatrixBase
from .categorical_matrix import CategoricalMatrix
from .dense_glm_matrix import DenseGLMDataMatrix
from .ext.split import split_col_subsets
from .mkl_sparse_matrix import MKLSparseMatrix


def split_sparse_and_dense_parts(
    arg1: sps.csc_matrix, threshold: float = 0.1
) -> Tuple[DenseGLMDataMatrix, MKLSparseMatrix, np.ndarray, np.ndarray]:
    if not isinstance(arg1, sps.csc_matrix):
        raise TypeError(
            f"X must be of type scipy.sparse.csc_matrix or matrix.MKLSparseMatrix, not {type(arg1)}"
        )
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1.")
    densities = np.diff(arg1.indptr) / arg1.shape[0]
    dense_indices = np.where(densities > threshold)[0]
    sparse_indices = np.setdiff1d(np.arange(densities.shape[0]), dense_indices)

    X_dense_F = DenseGLMDataMatrix(np.asfortranarray(arg1.toarray()[:, dense_indices]))
    X_sparse = MKLSparseMatrix(arg1[:, sparse_indices])
    return X_dense_F, X_sparse, dense_indices, sparse_indices


def csc_to_split(mat: sps.csc_matrix, threshold=0.1):
    dense, sparse, dense_idx, sparse_idx = split_sparse_and_dense_parts(mat, threshold)
    return SplitMatrix([dense, sparse], [dense_idx, sparse_idx])


class SplitMatrix(MatrixBase):
    def __init__(
        self,
        matrices: List[Union[DenseGLMDataMatrix, MKLSparseMatrix, CategoricalMatrix]],
        indices: Optional[List[np.ndarray]] = None,
    ):

        if indices is None:
            indices = []
            current_idx = 0
            for mat in matrices:
                indices.append(
                    np.arange(current_idx, current_idx + mat.shape[1], dtype=np.int)
                )
                current_idx += mat.shape[1]

        assert isinstance(indices, list)
        n_row = matrices[0].shape[0]
        self.dtype = matrices[0].dtype

        for i, (mat, idx) in enumerate(zip(matrices, indices)):
            if not mat.shape[0] == n_row:
                raise ValueError(
                    f"""
                    All matrices should have the same first dimension,
                    but the first matrix has first dimension {n_row} and matrix {i} has
                    first dimension {mat.shape[0]}."""
                )
            if not isinstance(mat, MatrixBase):
                raise ValueError(
                    "Expected all elements of matrices to be subclasses of MatrixBase."
                )
            if isinstance(mat, SplitMatrix):
                raise ValueError("Elements of matrices cannot be SplitMatrix.")
            if not mat.shape[1] == len(idx):
                raise ValueError(
                    f"""Element {i} of indices should should have length {mat.shape[1]},
                but it has shape {idx.shape}"""
                )
            if mat.dtype != self.dtype:
                warnings.warn(
                    f"""Matrices do not all have the same dtype. Dtypes are
                {[elt.dtype for elt in matrices]}."""
                )

        # If there are multiple spares and dense matrices, combine them
        for mat_type_, stack_fn in [
            (DenseGLMDataMatrix, np.hstack),
            (MKLSparseMatrix, sps.hstack),
        ]:
            this_type_matrices = [
                i for i, mat in enumerate(matrices) if isinstance(mat, mat_type_)
            ]
            if len(this_type_matrices) > 1:
                matrices[this_type_matrices[0]] = mat_type_(
                    stack_fn([matrices[i] for i in this_type_matrices])
                )
                assert matrices[this_type_matrices[0]].shape[0] == n_row
                indices[this_type_matrices[0]] = np.concatenate(
                    [indices[i] for i in this_type_matrices]
                )
                indices = [
                    idx
                    for i, idx in enumerate(indices)
                    if i not in this_type_matrices[1:]
                ]
                matrices = [
                    mat
                    for i, mat in enumerate(matrices)
                    if i not in this_type_matrices[1:]
                ]

        self.matrices = matrices
        self.indices = [np.asarray(I) for I in indices]
        self.shape = (n_row, sum([len(elt) for elt in indices]))
        assert self.shape[1] > 0

    def _split_col_subsets(
        self, cols
    ) -> Tuple[List[np.ndarray], List[Optional[np.ndarray]], int]:
        if cols is None:
            subset_cols_indices = self.indices
            subset_cols = [None for i in range(len(self.indices))]
            return subset_cols_indices, subset_cols, self.shape[1]

        return split_col_subsets(self, cols)

    def astype(self, dtype, order="K", casting="unsafe", copy=True):
        if copy:
            new_matrices = [
                mat.astype(dtype=dtype, order=order, casting=casting, copy=True)
                for mat in self.matrices
            ]
            return SplitMatrix(new_matrices, self.indices)
        for i in range(len(self.matrices)):
            self.matrices[i] = self.matrices[i].astype(
                dtype=dtype, order=order, casting=casting, copy=False
            )
        return SplitMatrix(self.matrices, self.indices)

    def toarray(self) -> np.ndarray:
        out = np.empty(self.shape)
        for mat, idx in zip(self.matrices, self.indices):
            out[:, idx] = mat.A
        return out

    def getcol(self, i: int) -> Union[np.ndarray, sps.csr_matrix]:
        # wrap-around indexing
        i %= self.shape[1]
        for mat, idx in zip(self.matrices, self.indices):
            if i in idx:
                loc = np.where(idx == i)[0][0]
                return mat.getcol(loc)
        raise RuntimeError(f"Column {i} was not found.")

    # taking 27%
    def sandwich(
        self,
        d: Union[np.ndarray, List],
        rows: np.ndarray = None,
        cols: np.ndarray = None,
    ) -> np.ndarray:
        if np.shape(d) != (self.shape[0],):
            raise ValueError
        d = np.asarray(d)

        subset_cols_indices, subset_cols, n_cols = self._split_col_subsets(cols)

        out = np.zeros((n_cols, n_cols))
        for i in range(len(self.indices)):
            idx_i = subset_cols_indices[i]
            mat_i = self.matrices[i]
            res = mat_i.sandwich(d, rows, subset_cols[i])
            if isinstance(res, sps.dia_matrix):
                out[(idx_i, idx_i)] += np.squeeze(res.data)
            else:
                out[np.ix_(idx_i, idx_i)] = res

            for j in range(i + 1, len(self.indices)):
                idx_j = subset_cols_indices[j]
                mat_j = self.matrices[j]
                res = mat_i.cross_sandwich(
                    mat_j, d, rows, subset_cols[i], subset_cols[j]
                )

                out[np.ix_(idx_i, idx_j)] = res
                out[np.ix_(idx_j, idx_i)] = res.T

        return out

    def get_col_means(self, weights: np.ndarray) -> np.ndarray:
        col_means = np.empty(self.shape[1], dtype=self.dtype)
        for idx, mat in zip(self.indices, self.matrices):
            col_means[idx] = mat.get_col_means(weights)
        return col_means

    def get_col_stds(self, weights: np.ndarray, col_means: np.ndarray) -> np.ndarray:
        col_stds = np.empty(self.shape[1], dtype=self.dtype)
        for idx, mat in zip(self.indices, self.matrices):
            col_stds[idx] = mat.get_col_stds(weights, col_means[idx])

        return col_stds

    # 28%
    def dot(
        self, v: np.ndarray, rows: np.ndarray = None, cols: np.ndarray = None
    ) -> np.ndarray:
        assert not isinstance(v, sps.spmatrix)
        v = np.asarray(v)
        if v.shape[0] != self.shape[1]:
            raise ValueError(f"shapes {self.shape} and {v.shape} not aligned")

        if cols is None:
            cols = np.arange(self.shape[1], dtype=np.int32)
        _, subset_cols, n_cols = self._split_col_subsets(cols)

        out_shape_base = [self.shape[0]] if rows is None else [rows.shape[0]]
        out_shape = out_shape_base + ([] if v.ndim == 1 else list(v.shape[1:]))
        out = np.zeros(out_shape, np.result_type(self.dtype, v.dtype))
        for sub_cols, idx, mat in zip(subset_cols, self.indices, self.matrices):
            one = v[idx, ...]
            tmp = mat.dot(one, rows, sub_cols)
            # 27% of time in this function is spent on this line
            out += tmp
        return out

    def transpose_dot(
        self,
        vec: Union[np.ndarray, List],
        rows: np.ndarray = None,
        cols: np.ndarray = None,
    ) -> np.ndarray:
        # 16.5%
        """
        self.T.dot(vec)[i] = sum_k self[k, i] vec[k]
        = sum_{k in self.dense_indices} self[k, i] vec[k] +
          sum_{k in self.sparse_indices} self[k, i] vec[k]
        = self.X_dense.T.dot(vec) + self.X_sparse.T.dot(vec)
        """

        vec = np.asarray(vec)
        if cols is None:
            cols = np.arange(self.shape[1], dtype=np.int32)
        subset_cols_indices, subset_cols, n_cols = self._split_col_subsets(cols)

        out_shape = [n_cols] + list(vec.shape[1:])
        out = np.empty(out_shape, dtype=vec.dtype)

        for idx, sub_cols, mat in zip(subset_cols_indices, subset_cols, self.matrices):
            out[idx, ...] = mat.transpose_dot(vec, rows, sub_cols)
        return out

    def __getitem__(self, key):
        if isinstance(key, tuple):
            row, col = key
        else:
            row = key
            col = slice(None, None, None)  # all columns

        if col == slice(None, None, None):
            if isinstance(row, int):
                row = [row]

            return SplitMatrix([mat[row, :] for mat in self.matrices], self.indices)
        else:
            raise NotImplementedError(
                f"Only row indexing is supported. Index passed was {key}."
            )

    __array_priority__ = 13
