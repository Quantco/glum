# cython: boundscheck=False, wraparound=False, cdivision=True

import numpy as np
cimport numpy as np

from cython cimport floating
cimport cython
from cython.parallel import prange
ctypedef np.uint8_t uint8
ctypedef np.int8_t int8


def transpose_dot(const int[:] indices, floating[:] other, int n_cols, dtype,
                  rows):
    cdef floating[:] res = np.zeros(n_cols, dtype=dtype)
    cdef int i, n_keep_rows
    cdef int n_rows = len(indices)
    cdef int[:] rows_view

    if rows is None or len(rows) == n_rows:
        for i in range(n_rows):
            res[indices[i]] += other[i]
    else:
        rows_view = rows
        n_keep_rows = len(rows_view)
        for k in range(n_keep_rows):
            i = rows_view[k]
            res[indices[i]] += other[i]

    return np.asarray(res)


def dot(const int[:] indices, floating[:] other, int n_rows, dtype):
    cdef floating[:] res = np.empty(n_rows, dtype=dtype)
    cdef int i

    for i in prange(n_rows, nogil=True):
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

