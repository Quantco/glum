import numpy as np
cimport numpy as np

from libcpp.vector cimport vector
from libc.stdlib cimport malloc, free
from cython cimport floating

# This is necessary because it's quite difficult to have a dynamic array of
# memoryviews in Cython
# https://stackoverflow.com/questions/56915835/how-to-have-a-list-of-memory-views-in-cython
cdef struct ArrayableMemoryView:
    long* data
    long length


def sandwich_cat_dense(
    int[:] i_indices,
    int i_ncol,
    int j_ncol,
    floating[:] d,
    floating[:, :] mat_j
):
    """
    Expect mat_j to be C-contiguous (row major)
    """

    cdef Py_ssize_t i, j, k
    cdef Py_ssize_t n_rows = len(i_indices)
    cdef int* i_indices_p = &i_indices[0]

    res = np.zeros((i_ncol, j_ncol))
    for k in range(n_rows):
        i = i_indices_p[k]
        for j in range(j_ncol):
            tmp = d[k] * mat_j[k, j]
            res[i, j] += tmp
    return res


def _sandwich_cat_cat(
    int[:] i_indices,
    int[:] j_indices,
    int i_ncol,
    int j_ncol,
    floating[:] d
):
    """
    (X1.T @ diag(d) @ X2)[i, j] = sum_k X1[k, i] d[k] X2[k, j]
    """

    # TODO: only use i_cols, j_cols. A csc setup might be better for that
    # TODO: this is probably not right when columns are scaled
    # TODO: support for single-precision d

    cdef floating[:, :] res
    res = np.zeros((i_ncol, j_ncol))
    cdef int i
    cdef int j
    cdef int k
    cdef floating d_

    for k in range(len(d)):
        i = i_indices[k]
        j = j_indices[k]
        d_ = d[k]
        res[i, j] += d_

    return res



def split_col_subsets(self, int[:] cols):
    cdef int[:] next_subset_idx = np.zeros(len(self.indices), dtype=np.int32)
    cdef vector[vector[int]] subset_cols_indices
    cdef vector[vector[int]] subset_cols
    cdef vector[int] empty = []

    cdef int j
    cdef int n_matrices = len(self.indices)
    cdef ArrayableMemoryView* indices_arrs = <ArrayableMemoryView*> malloc(
        sizeof(ArrayableMemoryView) * n_matrices
    );
    cdef long[:] this_idx_view
    for j in range(n_matrices):
        this_idx_view = self.indices[j]
        indices_arrs[j].length = len(this_idx_view)
        if indices_arrs[j].length > 0:
            indices_arrs[j].data = &(this_idx_view[0])
        subset_cols_indices.push_back(empty)
        subset_cols.push_back(empty)

    cdef int i
    cdef int n_cols = cols.shape[0]

    for i in range(n_cols):
        for j in range(n_matrices):
            while (
                next_subset_idx[j] < indices_arrs[j].length
                and indices_arrs[j].data[next_subset_idx[j]] < cols[i]
            ):
                next_subset_idx[j] += 1
            if (
                next_subset_idx[j] < indices_arrs[j].length
                and indices_arrs[j].data[next_subset_idx[j]] == cols[i]
            ):
                subset_cols_indices[j].push_back(i)
                subset_cols[j].push_back(next_subset_idx[j])
                next_subset_idx[j] += 1
                break

    free(indices_arrs)
    return (
        [
            np.array(subset_cols_indices[j], dtype=np.int32)
            for j in range(n_matrices)
        ],
        [
            np.array(subset_cols[j], dtype=np.int32)
            for j in range(n_matrices)
        ],
        n_cols
    )
