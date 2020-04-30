# distutils: extra_compile_args=-fopenmp -O3 -ffast-math -march=native
# distutils: extra_link_args=-fopenmp
import numpy as np

import cython
from cython cimport floating
from cython.parallel import prange

@cython.boundscheck(False)
@cython.wraparound(False)
def sparse_sandwich(A, AT, floating[:] d):
    # AT is CSC
    # A is CSC
    # Computes AT @ diag(d) @ A

    cdef floating[:] Adata = A.data
    cdef int[:] Aindices = A.indices
    cdef int[:] Aindptr = A.indptr

    cdef floating[:] ATdata = AT.data
    cdef int[:] ATindices = AT.indices
    cdef int[:] ATindptr = AT.indptr

    cdef floating* Adatap = &Adata[0]
    cdef int* Aindicesp = &Aindices[0]
    cdef floating* ATdatap = &ATdata[0]
    cdef int* ATindicesp = &ATindices[0]
    cdef int* ATindptrp = &ATindptr[0]

    cdef floating* dp = &d[0]

    cdef int m = Aindptr.shape[0] - 1
    cdef int n = d.shape[0]
    cdef int nnz = Adata.shape[0]
    out = np.zeros((m,m), dtype=A.dtype)
    cdef floating[:, :] out_view = out
    cdef floating* outp = &out_view[0,0]

    cdef int AT_idx, A_idx
    cdef int AT_row, A_col
    cdef int i, j, k
    cdef floating A_val, AT_val

    for j in prange(m, nogil=True):
        for A_idx in range(Aindptr[j], Aindptr[j+1]):
            k = Aindicesp[A_idx]
            A_val = Adatap[A_idx] * dp[k]
            for AT_idx in range(ATindptrp[k], ATindptrp[k+1]):
                i = ATindicesp[AT_idx]
                if i > j:
                    break
                AT_val = ATdatap[AT_idx]
                outp[j * m + i] = outp[j * m + i] + AT_val * A_val

    out += np.tril(out, -1).T
    return out

cdef extern from "dense.c":
    void _dense_sandwich(double*, double*, double*, int, int) nogil
    void _sparse_dense_sandwich(double*, int*, int*, double*, double*, double*, int, int, int) nogil

def sparse_dense_sandwich(A, double[:,:] B, double[:] d):
    # computes where (A.T * d) @ B
    # assumes that A is in csr form
    cdef double[:] Adata = A.data
    cdef int[:] Aindices = A.indices
    cdef int[:] Aindptr = A.indptr

    # A has shape (n, m)
    # B has shape (n, r)
    cdef int m = A.shape[1]
    cdef int n = d.shape[0]
    cdef int r = B.shape[1]

    out = np.zeros((m, r), dtype=A.dtype)
    cdef double[:, :] out_view = out
    cdef double* outp = &out_view[0,0]

    _sparse_dense_sandwich(&Adata[0], &Aindices[0], &Aindptr[0], &B[0,0], &d[0], outp, m, n, r)

    # cdef int i, j, k
    # cdef int A_idx
    # cdef double Q

    # for i in range(m):
    #     for j in range(r):
    #         for A_idx in range(Aindptr[i], Aindptr[i+1]):
    #             k = Aindices[A_idx]
    #             Q = Adata[A_idx] * d[k]

    #             out_view[i, j] = out_view[i, j] + Q * B[k, j]

    # for i in range(m):
    #     for A_idx in range(Aindptr[i], Aindptr[i+1]):
    #         k = Aindices[A_idx]
    #         Q = Adata[A_idx] * d[k]
    #         for j in range(r):
    #             out_view[i, j] = out_view[i, j] + Q * B[k, j]
    
    # for k in range(n):
    #     for A_idx in range(Aindptr[k], Aindptr[k+1]):
    #         i = Aindices[A_idx]
    #         Q = d[k] * Adata[A_idx]
    #         for j in range(r):
    #             # TODO: i, j is lesss efficient
    #             out_view[i, j] = out_view[i, j] + B[k, j] * Q

    # for k in range(n):
    #     for j in range(r):
    #         for A_idx in range(Aindptr[k], Aindptr[k+1]):
    #             i = Aindices[A_idx]
    #             Q = d[k] * Adata[A_idx]
    #             # TODO: i, j is lesss efficient
    #             out_view[i, j] = out_view[i, j] + B[k, j] * Q
    return out



def dense_sandwich(double[:,:] X, double[:] d):
    cdef int m = X.shape[1]
    cdef int n = X.shape[0]

    out = np.zeros((m,m))
    cdef double[:, :] out_view = out
    cdef double* outp = &out_view[0,0]

    cdef double* Xp = &X[0,0]
    cdef double* dp = &d[0]
    _dense_sandwich(Xp, dp, outp, m, n)
    out += np.tril(out, -1).T
    return out
