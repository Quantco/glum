# distutils: extra_compile_args=-fopenmp -O3 -ffast-math -march=native -g
# distutils: extra_link_args=-fopenmp
# distutils: language = c++
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

    #TODO: see what happens when we swap to having k as the outer loop here?
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

cdef extern from "dense.cpp":
    void _dense_sandwich[F](F*, F*, F*, int, int, int, int, int, int) nogil
    void _csr_dense_sandwich[F](F*, int*, int*, F*, F*, F*, int, int, int) nogil

def csr_dense_sandwich(A, floating[:,:] B, floating[:] d):
    # computes where (A.T * d) @ B
    # assumes that A is in csr form
    cdef floating[:] Adata = A.data
    cdef int[:] Aindices = A.indices
    cdef int[:] Aindptr = A.indptr

    # A has shape (n, m)
    # B has shape (n, r)
    cdef int m = A.shape[1]
    cdef int n = d.shape[0]
    cdef int r = B.shape[1]

    out = np.zeros((m, r), dtype=A.dtype)
    cdef floating[:, :] out_view = out
    cdef floating* outp = &out_view[0,0]

    _csr_dense_sandwich(&Adata[0], &Aindices[0], &Aindptr[0], &B[0,0], &d[0], outp, m, n, r)
    return out



def dense_sandwich(X, floating[:] d, int thresh1d, int parlevel, int kratio, int kstep):
    cdef int n = X.shape[0]
    cdef int m = X.shape[1]

    out = np.zeros((m,m), dtype=X.dtype)
    cdef floating[:, :] out_view = out
    cdef floating* outp = &out_view[0,0]

    cdef floating[:, :] Xmemview = X;
    cdef floating* Xp = &Xmemview[0,0]
    cdef floating* dp = &d[0]
    _dense_sandwich(Xp, dp, outp, m, n, thresh1d, parlevel, kratio, kstep)
    return out
