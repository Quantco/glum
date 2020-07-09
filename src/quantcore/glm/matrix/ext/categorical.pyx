# cython: boundscheck=False, wraparound=False, cdivision=True

import numpy as np
cimport numpy as np

from cython cimport floating
cimport cython
from cython.parallel import prange
ctypedef np.uint8_t uint8
ctypedef np.int8_t int8


cdef extern from "cat_split_helpers.cpp":
    void _transpose_dot_all_rows[F](int, int*, F*, F*, int)


def transpose_dot(int[:] indices, floating[:] other, int n_cols, dtype,
                  rows):
    cdef floating[:] res = np.zeros(n_cols, dtype=dtype)
    cdef int i, n_keep_rows
    cdef int n_rows = len(indices)
    cdef int[:] rows_view

    if rows is None or len(rows) == n_rows:
        _transpose_dot_all_rows(n_rows, &indices[0], &other[0], &res[0], res.size)
    else:
        rows_view = rows
        n_keep_rows = len(rows_view)
        for k in range(n_keep_rows):
            i = rows_view[k]
            res[indices[i]] += other[i]

    return np.asarray(res)


def dot(const int[:] indices, floating[:] other, int n_rows, dtype, int[:] cols,
        int n_cols):
    cdef floating[:] res = np.zeros(n_rows, dtype=dtype)
    cdef int i, col, Ci, k
    cdef uint8[:] col_included

    if cols is None:
        for i in prange(n_rows, nogil=True):
            res[i] = other[indices[i]]
    else:
        col_included = np.zeros(n_cols, dtype=np.uint8)
        for Ci in range(len(cols)):
            col_included[cols[Ci]] = 1

        for i in prange(n_rows, nogil=True):
            col = indices[i]
            if col_included[col] == 1:
                res[i] = other[indices[i]]

    return np.asarray(res)


def sandwich_categorical(const int[:] indices, floating[:] d,
                        int[:] rows, dtype, int n_cols):
    cdef floating[:] res = np.zeros(n_cols, dtype=dtype)
    cdef size_t i, k, k_idx
    cdef int n_rows = len(rows)

    for k_idx in range(n_rows):
        k = rows[k_idx]
        i = indices[k]
        res[i] += d[k]
    return np.asarray(res)

