import numpy as np
cimport numpy as np

import cython
from cython cimport floating
from cython.parallel import prange

ctypedef np.uint8_t uint8

@cython.boundscheck(False)
@cython.wraparound(False)
def sparse_sandwich(A, AT, floating[:] d, int[:] rows, int[:] cols):
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

    cdef int m = cols.shape[0]
    out = np.zeros((m, m), dtype=A.dtype)
    cdef floating[:, :] out_view = out
    cdef floating* outp = &out_view[0,0]

    cdef int AT_idx, A_idx
    cdef int AT_row, A_col
    cdef int Ci, i, Cj, j, Ck, k
    cdef floating A_val, AT_val

    cdef uint8[:] row_included = np.zeros(d.shape[0], dtype=np.uint8)
    for Ci in range(rows.shape[0]):
        row_included[rows[Ci]] = True

    cdef int[:] col_map = np.full(A.shape[1], -1, dtype=np.int32)
    for Cj in range(cols.shape[0]):
        col_map[cols[Cj]] = Cj

    #TODO: see what happens when we swap to having k as the outer loop here?
    for Cj in prange(m, nogil=True):
        j = cols[Cj]
        Ck = 0
        for A_idx in range(Aindptr[j], Aindptr[j+1]):
            k = Aindicesp[A_idx]
            if not row_included[k]:
                continue

            A_val = Adatap[A_idx] * dp[k]
            Ci = 0
            for AT_idx in range(ATindptrp[k], ATindptrp[k+1]):
                i = ATindicesp[AT_idx]
                if i > j:
                    break

                Ci = col_map[i]
                if Ci == -1:
                    continue

                AT_val = ATdatap[AT_idx]
                outp[Cj * m + Ci] = outp[Cj * m + Ci] + AT_val * A_val

    out += np.tril(out, -1).T
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
def csr_matvec(X, floating[:] v, int[:] rows, int[:] cols):
    cdef floating[:] Xdata = X.data
    cdef int[:] Xindices = X.indices
    cdef int[:] Xindptr = X.indptr

    cdef int n = rows.shape[0]
    cdef int m = cols.shape[0]
    out = np.zeros(n, dtype=X.dtype)
    cdef floating[:] out_view = out;

    cdef floating* Xdatap = &Xdata[0];
    cdef int* Xindicesp = &Xindices[0];
    cdef int* Xindptrp = &Xindptr[0];
    cdef floating* outp = &out_view[0];

    cdef int Ci, i, Cj, X_idx, j
    cdef floating Xval, vval

    cdef uint8[:] col_included = np.zeros(X.shape[1], dtype=np.uint8)
    for Cj in range(cols.shape[0]):
        col_included[cols[Cj]] = True

    for Ci in prange(n, nogil=True):
        i = rows[Ci]
        for X_idx in range(Xindptrp[i], Xindptrp[i+1]):
            j = Xindicesp[X_idx]
            if not col_included[j]:
                continue
            Xval = Xdatap[X_idx]
            vval = v[j]
            outp[Ci] = outp[Ci] + Xval * vval;
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
def csc_rmatvec(XT, floating[:] v, int[:] rows, int[:] cols):
    cdef floating[:] XTdata = XT.data
    cdef int[:] XTindices = XT.indices
    cdef int[:] XTindptr = XT.indptr

    cdef int n = rows.shape[0]
    cdef int m = cols.shape[0]
    out = np.zeros(m, dtype=XT.dtype)
    cdef floating[:] out_view = out;

    cdef floating* XTdatap = &XTdata[0];
    cdef int* XTindicesp = &XTindices[0];
    cdef int* XTindptrp = &XTindptr[0];
    cdef floating* outp = &out_view[0];
    cdef int* rowsp
    cdef int* colsp

    cdef int Ci, i, Cj, XT_idx, j
    cdef floating XTval, vval

    cdef uint8[:] row_included = np.zeros(XT.shape[0], dtype=np.uint8)
    for Ci in range(rows.shape[0]):
        row_included[rows[Ci]] = True

    for Cj in prange(m, nogil=True):
        j = cols[Cj]
        for XT_idx in range(XTindptrp[j], XTindptrp[j+1]):
            i = XTindicesp[XT_idx]
            if not row_included[i]:
                continue
            XTval = XTdatap[XT_idx];
            vval = v[i]
            outp[Cj] = outp[Cj] + XTval * vval;
    return out

cdef extern from "sparse_helpers.cpp":
    void _csr_denseC_sandwich[F](
        F*, int*, int*, F*, F*, F*, int, int, int,
        int*, int*, int*, int, int, int
    ) nogil
    void _csr_denseF_sandwich[F](
        F*, int*, int*, F*, F*, F*, int, int, int,
        int*, int*, int*, int, int, int
    ) nogil

def csr_dense_sandwich(A, B, floating[:] d, int[:] rows, int[:] A_cols, int[:] B_cols):
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

    cdef int nr = rows.shape[0]
    cdef int nAc = A_cols.shape[0]
    cdef int nBc = B_cols.shape[0]

    out = np.zeros((nAc, nBc), dtype=A.dtype)
    if nr == 0 or nAc == 0 or nBc == 0 or (Aindptr[-1] - Aindptr[0]) == 0:
        return out

    cdef floating[:, :] out_view = out
    cdef floating* outp = &out_view[0,0]

    cdef floating[:, :] B_view = B;
    cdef floating* Bp = &B_view[0, 0];

    cdef int* rowsp = &rows[0];
    cdef int* A_colsp = &A_cols[0];
    cdef int* B_colsp = &B_cols[0];

    if B.flags['C_CONTIGUOUS']:
        _csr_denseC_sandwich(
            &Adata[0], &Aindices[0], &Aindptr[0], Bp, &d[0], outp, m, n, r, 
            rowsp, A_colsp, B_colsp, nr, nAc, nBc
        )
    elif B.flags['F_CONTIGUOUS']:
        _csr_denseF_sandwich(
            &Adata[0], &Aindices[0], &Aindptr[0], Bp, &d[0], outp, m, n, r, 
            rowsp, A_colsp, B_colsp, nr, nAc, nBc
        )
    else:
        raise Exception()
    return out
