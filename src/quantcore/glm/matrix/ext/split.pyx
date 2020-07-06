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
ctypedef np.uint8_t uint8
ctypedef np.int8_t int8


def sandwich_cat_dense(
    np.ndarray i_indices_,
    int i_ncol,
    floating[:] d,
    floating[:, :] mat_j,
    int[:] rows,
    int[:] j_cols
):
    """
    Expect mat_j to be C-contiguous (row major)
    """

    cdef floating[:, :] res
    res = np.zeros((i_ncol, len(j_cols)))
    cdef const int8[:] i_indices = i_indices_.view(dtype=np.int8)

    if len(d) == 0 or len(rows) == 0 or len(j_cols) == 0 or i_ncol == 0:
        return np.asarray(res)

    # Ben says pointers are probably a < 5%-10% improvement over memoryviews
    cdef size_t i, j, k, k_idx, j_idx
    cdef floating* d_p = &d[0]
    cdef const int8* i_indices_p = &i_indices[0]
    cdef int* rows_p = &rows[0]
    cdef int* j_cols_p = &j_cols[0]

    for k_idx in range(len(rows)):
        k = rows_p[k_idx]
        i = i_indices_p[k]
        for j_idx in range(len(j_cols)):
            j = j_cols_p[j_idx]
            res[i, j_idx] += d_p[k] * mat_j[k, j]

    return np.asarray(res)


def _sandwich_cat_cat(
    np.ndarray i_indices_,
    np.ndarray j_indices_,
    int i_ncol,
    int j_ncol,
    floating[:] d,
    int[:] rows,
):
    """
    (X1.T @ diag(d) @ X2)[i, j] = sum_k X1[k, i] d[k] X2[k, j]
    """
    # TODO: support for single-precision d
    cdef const int8[:] i_indices = i_indices_.view(dtype=np.int8)
    cdef const int8[:] j_indices = j_indices_.view(dtype=np.int8)

    cdef floating[:, :] res
    res = np.zeros((i_ncol, j_ncol))
    cdef size_t k_idx, k, i, j

    for k_idx in range(len(rows)):
        k = rows[k_idx]
        i = i_indices[k]
        j = j_indices[k]
        res[i, j] += d[k]

    return np.asarray(res)


# This seems slower, so not using it for now
def _sandwich_cat_cat_limited_rows_cols(
    int[:] i_indices,
    int[:] j_indices,
    int i_ncol,
    int j_ncol,
    floating[:] d,
    int[:] rows,
    int[:] i_cols,
    int[:] j_cols
):
    """
    (X1.T @ diag(d) @ X2)[i, j] = sum_k X1[k, i] d[k] X2[k, j]
    """

    # TODO: support for single-precision d
    # TODO: this is writing an output of the wrong shape; filtering on rows
    # and cols still needs to happen after
    # TODO: Look into sparse matrix multiplication algorithms. Should one or both be csc?

    cdef floating[:, :] res
    res = np.zeros((i_ncol, j_ncol))
    cdef size_t k_idx, k, i, j

    cdef uint8[:] i_col_included = np.zeros(i_ncol, dtype=np.uint8)
    for Ci in range(i_ncol):
        i_col_included[i_cols[Ci]] = True

    cdef uint8[:] j_col_included = np.zeros(j_ncol, dtype=np.uint8)
    for Ci in range(j_ncol):
        j_col_included[j_cols[Ci]] = True

    for k_idx in range(len(rows)):
        k = rows[k_idx]
        i = i_indices[k]
        if i_col_included[i]:
            j = j_indices[k]
            if j_col_included[j]:
                res[i, j] += d[k]

    return np.asarray(res)


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
