# distutils: extra_compile_args=-fopenmp -O3 -ffast-math -march=native -msse -msse2 -mavx
# distutils: extra_link_args=-fopenmp
import numpy as np
cimport numpy as np
import scipy as sp
import scipy.sparse
from cython cimport view
import cython
from cython.parallel import parallel, prange
from libc.math cimport ceil, sqrt
cimport openmp
from libc.stdlib cimport malloc, free


@cython.boundscheck(False)
@cython.wraparound(False)
def fast_sandwich(A, AT, double[:] d):
    # AT is CSC
    # A is CSC
    # Computes AT @ diag(d) @ A

    cdef double[:] Adata = A.data
    cdef int[:] Aindices = A.indices
    cdef int[:] Aindptr = A.indptr

    cdef double[:] ATdata = AT.data
    cdef int[:] ATindices = AT.indices
    cdef int[:] ATindptr = AT.indptr

    cdef int ncols = Aindptr.shape[0] - 1
    cdef int nrows = d.shape[0]
    cdef int nnz = Adata.shape[0]
    out = np.zeros((ncols,ncols))
    cdef double[:, :] out_view = out

    cdef int AT_idx, A_idx
    cdef int AT_row, A_col
    cdef int i, j, k
    cdef double A_val, AT_val

    for j in prange(ncols, nogil=True):
        for A_idx in range(Aindptr[j], Aindptr[j+1]):
            k = Aindices[A_idx]
            A_val = Adata[A_idx] * d[k]
            for AT_idx in range(ATindptr[k], ATindptr[k+1]):
                i = ATindices[AT_idx]
                if i < j:
                    continue
                AT_val = ATdata[AT_idx]
                out_view[i, j] = out_view[i, j] + AT_val * A_val

    out += np.tril(out, -1).T
    return out

# @cython.boundscheck(False)
# @cython.wraparound(False)
# def dense_sandwich_base(double[:,:] X, double[:] d):
#     # unrolling doesn't seem to help
#     cdef int m = X.shape[1]
#     cdef int n = X.shape[0]
#     cdef int it = 0
#     cdef int jt, kt, i, j, k
# 
#     out = np.zeros((m,m))
#     cdef double[:, :] out_view = out
# 
#     cdef int blocksize = 50
#     cdef int nblocks = n / blocksize
#     cdef int kb
# 
#     cdef double Q
#     # for i in prange(m, nogil=True):
#     for kb in range(nblocks):
#         for i in range(m):
#             for k in range(kb * blocksize, (kb + 1) * blocksize):
#                 Q = X[k, i] * d[k]
#                 for j in range(i + 1):
#                     out_view[i, j] = out_view[i,j] + Q * X[k, j]
#     out += np.tril(out, -1).T
#     return out
# 
# @cython.boundscheck(False)
# @cython.wraparound(False)
# def dense_sandwich2(double[:,:] X, double[:] d):
#     # unrolling doesn't seem to help
#     cdef int m = X.shape[1]
#     cdef int n = X.shape[0]
#     cdef int it = 0
#     cdef int jt, kt, i, j, k
# 
#     out = np.zeros((m,m))
#     cdef double[:, :] out_view = out
# 
#     cdef int blocksize = 100000
#     cdef int nblocks = n / blocksize
#     cdef int kb
# 
#     cdef double Q
#     # for i in prange(m, nogil=True):
#     for kb in range(nblocks):
#         for i in prange(m, nogil=True):
#             for k in range(kb * blocksize, (kb + 1) * blocksize):
#                 Q = X[k, i] * d[k]
#                 j = 0
#                 while j < i + 2 - 4:
#                     out_view[i, j] = out_view[i, j] + Q * X[k, j]
#                     out_view[i, j + 1] = out_view[i,j + 1] + Q * X[k, j + 1]
#                     out_view[i, j + 2] = out_view[i,j + 2] + Q * X[k, j + 2]
#                     out_view[i, j + 3] = out_view[i,j + 3] + Q * X[k, j + 3]
#                     j = j + 4
#                 while j < i + 1:
#                     out_view[i, j] = out_view[i, j] + Q * X[k, j]
#                     j = j + 1
#     out += np.tril(out, -1).T
#     return out
# 
# 
# cdef extern from "immintrin.h":  # in this example, we use SSE2
#     ctypedef double __m256d
#     __m256d _mm256_loadu_pd (double* __P) nogil
#     __m256d _mm256_add_pd   (__m256d __A, __m256d __B) nogil
#     __m256d _mm256_mul_pd   (__m256d __A, __m256d __B) nogil
#     void   _mm256_store_pd (double* __P, __m256d __A) nogil
#     __m256d _mm256_set1_pd (double P) nogil
# 
# @cython.boundscheck(False)
# @cython.wraparound(False)
# def dense_sandwich4(double[:,:] X, double[:] d):
#     # unrolling doesn't seem to help
#     cdef int m = X.shape[1]
#     cdef int n = X.shape[0]
#     cdef int it = 0
#     cdef int jt, kt, i, j, k
# 
#     out = np.zeros((m,m))
#     cdef double[:, :] out_view2 = out
#     cdef double* outp = &out_view2[0,0]
# 
#     cdef double* Xp = &X[0,0]
#     cdef double* dp = &d[0]
# 
#     cdef int kblocksize = 100
#     cdef int nkblocks = n / kblocksize
#     cdef int kb
#     cdef int min_k, max_k
# 
#     cdef int jblocksize = 16
#     cdef int njblocks = m / jblocksize
#     cdef int jb
#     cdef int min_j, max_j
# 
#     cdef int iblocksize = 4
#     cdef int niblocks = m / iblocksize
#     cdef int ib
#     cdef int min_i, max_i
# 
#     cdef double Q, accum
#     cdef int idx
# 
# 
#     cdef __m256d accumavx
#     cdef __m256d XTavx
#     cdef __m256d davx
#     cdef __m256d Xavx
#     cdef double[4] output
# 
#     # for idx in prange(niblocks * njblocks, nogil=True):
#     for idx in range(niblocks * njblocks):
#         ib = idx // njblocks
#         jb = idx % njblocks
#         min_j = jb * jblocksize
#         min_i = ib * iblocksize
#         max_i = min(m, (ib + 1) * iblocksize)
#         for kb in range(nkblocks):
#             min_k = kb * kblocksize
#             max_k = min(n, (kb + 1) * kblocksize)
#             for i in range(min_i, max_i):
#                 # max_j = min(i + 1, (jb + 1) * jblocksize)
#                 # for j in range(min_j, max_j):
#                 #     accum = 0
#                 #     for k in range(min_k, max_k):
#                 #          accum += 
#                 #     outp[i * m + j] += accum
# 
#                 max_j = min(i + 1, (jb + 1) * jblocksize)
#                 if max_k != (kb + 1) * kblocksize:
#                     for j in range(min_j, max_j):
#                         accum = 0
#                         for k in range(min_k, max_k):
#                              accum += Xp[i * n + k] * dp[k] * Xp[j * n + k]
#                         outp[i * m + j] += accum
#                 else:
#                     for j in range(min_j, max_j):
#                         accumavx = _mm256_set1_pd(0.0)
#                         for k in range(min_k, max_k, 4):
#                             XTavx = _mm256_loadu_pd(&Xp[i * n + k])
#                             Xavx = _mm256_loadu_pd(&Xp[j * n + k])
#                             davx = _mm256_loadu_pd(&d[k])
#                             accumavx = _mm256_add_pd(_mm256_mul_pd(_mm256_mul_pd(XTavx, davx), Xavx), accumavx)
#                         _mm256_store_pd(output, accumavx)
#                         outp[i * m + j] += output[0] + output[1] + output[2] + output[3]
# 
# 
#     out += np.tril(out, -1).T
#     return out

cdef extern from "dense.c":
    void dense_C(double*, double*, double*, int, int) nogil
    void dense_C2(double*, double*, double*, int, int) nogil

def dense_sandwich(double[:,:] X, double[:] d):
    cdef int m = X.shape[1]
    cdef int n = X.shape[0]

    out = np.zeros((m,m))
    cdef double[:, :] out_view2 = out
    cdef double* outp = &out_view2[0,0]

    cdef double* Xp = &X[0,0]
    cdef double* dp = &d[0]
    dense_C(Xp, dp, outp, m, n)
    out += np.tril(out, -1).T
    return out

def dense_sandwich2(double[:,:] X, double[:] d):
    cdef int m = X.shape[1]
    cdef int n = X.shape[0]

    out = np.zeros((m,m))
    cdef double[:, :] out_view2 = out
    cdef double* outp = &out_view2[0,0]

    cdef double* Xp = &X[0,0]
    cdef double* dp = &d[0]
    dense_C2(Xp, dp, outp, m, n)
    out += np.tril(out, -1).T
    return out
