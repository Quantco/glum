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

cdef extern from "mkl.h":

    ctypedef int MKL_INT

    ctypedef enum sparse_index_base_t:
        SPARSE_INDEX_BASE_ZERO = 0
        SPARSE_INDEX_BASE_ONE = 1


    ctypedef enum sparse_status_t:
        SPARSE_STATUS_SUCCESS = 0 # the operation was successful
        SPARSE_STATUS_NOT_INITIALIZED = 1 # empty handle or matrix arrays
        SPARSE_STATUS_ALLOC_FAILED = 2 # internal error: memory allocation failed
        SPARSE_STATUS_INVALID_VALUE = 3 # invalid input value
        SPARSE_STATUS_EXECUTION_FAILED = 4 # e.g. 0-diagonal element for triangular solver, etc.
        SPARSE_STATUS_INTERNAL_ERROR = 5 # internal error
        SPARSE_STATUS_NOT_SUPPORTED = 6 # e.g. operation for double precision doesn't support other types */

    ctypedef enum sparse_operation_t:
        SPARSE_OPERATION_NON_TRANSPOSE = 10
        SPARSE_OPERATION_TRANSPOSE = 11
        SPARSE_OPERATION_CONJUGATE_TRANSPOSE = 12


    ctypedef enum sparse_matrix_type_t:
        SPARSE_MATRIX_TYPE_GENERAL = 20 # General case
        SPARSE_MATRIX_TYPE_SYMMETRIC = 21 # Triangular part of the matrix is to be processed
        SPARSE_MATRIX_TYPE_HERMITIAN = 22
        SPARSE_MATRIX_TYPE_TRIANGULAR = 23
        SPARSE_MATRIX_TYPE_DIAGONAL = 24 # diagonal matrix; only diagonal elements will be processed
        SPARSE_MATRIX_TYPE_BLOCK_TRIANGULAR = 25
        SPARSE_MATRIX_TYPE_BLOCK_DIAGONAL = 26 # block-diagonal matrix; only diagonal blocks will be processed

    ctypedef enum sparse_fill_mode_t:
        SPARSE_FILL_MODE_LOWER = 40 # lower triangular part of the matrix is stored
        SPARSE_FILL_MODE_UPPER = 41 # upper triangular part of the matrix is stored
        SPARSE_FILL_MODE_FULL = 42 # upper triangular part of the matrix is stored

    ctypedef enum sparse_diag_type_t:
        SPARSE_DIAG_NON_UNIT = 50 # triangular matrix with non-unit diagonal
        SPARSE_DIAG_UNIT = 51 # triangular matrix with unit diagonal

    ctypedef enum sparse_layout_t:
        SPARSE_LAYOUT_ROW_MAJOR = 101 # C-style
        SPARSE_LAYOUT_COLUMN_MAJOR = 102 # Fortran-style

    ctypedef enum sparse_request_t:
        SPARSE_STAGE_FULL_MULT = 90
        SPARSE_STAGE_NNZ_COUNT = 91
        SPARSE_STAGE_FINALIZE_MULT = 92
        SPARSE_STAGE_FULL_MULT_NO_VAL = 93
        SPARSE_STAGE_FINALIZE_MULT_NO_VAL = 94

    struct sparse_matrix:
        pass

    ctypedef sparse_matrix* sparse_matrix_t


    struct matrix_descr:
        sparse_matrix_type_t type  # matrix type: general, diagonal or triangular / symmetric / hermitian
        sparse_fill_mode_t mode  # upper or lower triangular part of the matrix ( for triangular / symmetric / hermitian case)
        sparse_diag_type_t diag  # unit or non-unit diagonal ( for triangular / symmetric / hermitian case)

    sparse_status_t mkl_sparse_d_mv(
        sparse_operation_t operation,
        double alpha,
        const sparse_matrix_t A,
        matrix_descr descr,
        const double *x,
        double beta,
        double *y
    )

    sparse_status_t mkl_sparse_d_create_csr(
        sparse_matrix_t* A,
        const sparse_index_base_t indexing, # indexing: C-style or Fortran-style
        const MKL_INT rows,
        const MKL_INT cols,
        MKL_INT *rows_start,
        MKL_INT *rows_end,
        MKL_INT *col_indx,
        double *values
    )

    sparse_status_t mkl_sparse_d_create_csc(
        sparse_matrix_t* A,
        const sparse_index_base_t indexing,
        const MKL_INT rows,
        const MKL_INT cols,
        MKL_INT *cols_start,
        MKL_INT *cols_end,
        MKL_INT *row_indx,
        double *values
    )

cdef class MklSparseMatrix:
    cdef sparse_matrix_t A
    cdef nrow, ncol

    def __cinit__(self, A_py):
        self.A = to_mkl_matrix(A_py)
        self.nrow = A_py.shape[0]
        self.ncol = A_py.shape[1]

    @property
    def shape(self):
        return self.nrow, self.ncol

    def dot(self, x, transpose=False):
        result = np.zeros(self.shape[transpose])
        cdef double[:] x_view = x
        cdef double[:] result_view = result
        matvec_status = mkl_plain_matvec(
            self.A, &x_view[0], &result_view[0], int(transpose)
        )
        return result

class MKLCallError(Exception):
    pass

cdef sparse_matrix_t to_mkl_matrix(A_py):

    cdef MKL_INT rows = A_py.shape[0]
    cdef MKL_INT cols = A_py.shape[1]
    cdef sparse_matrix_t A
    cdef sparse_index_base_t base_index=SPARSE_INDEX_BASE_ZERO

    cdef MKL_INT[:] indptr_view = A_py.indptr
    cdef MKL_INT[:] indices_view = A_py.indices
    cdef double[:] value_view = A_py.data

    cdef MKL_INT* start = &indptr_view[0]
    cdef MKL_INT* end = &indptr_view[1]
    cdef MKL_INT* index = &indices_view[0]
    cdef double* values = &value_view[0]

    if A_py.getformat() == 'csr':
        mkl_sparse_d_create = mkl_sparse_d_create_csr
    else:
        mkl_sparse_d_create = mkl_sparse_d_create_csc

    create_status = mkl_sparse_d_create(
        &A, base_index, rows, cols,
        start, end, index, values
    )
    if create_status != SPARSE_STATUS_SUCCESS:
        raise MKLCallError("Creating an MKL sparse matrix failed.")
    return A

cdef mkl_plain_matvec(
        sparse_matrix_t A, const double* x, double* result, bint transpose
    ):

    cdef sparse_operation_t operation
    if transpose:
        operation = SPARSE_OPERATION_TRANSPOSE
    else:
        operation = SPARSE_OPERATION_NON_TRANSPOSE
    cdef double alpha = 1.
    cdef double beta = 0.
    cdef matrix_descr mat_descript
    mat_descript.type = SPARSE_MATRIX_TYPE_GENERAL
    status = mkl_sparse_d_mv(
        operation, alpha, A, mat_descript, x, beta, result
    )
    return status


@cython.boundscheck(False)
@cython.wraparound(False)
def sparse_sandwich(A, AT, double[:] d):
    # AT is CSC
    # A is CSC
    # Computes AT @ diag(d) @ A

    cdef double[:] Adata = A.data
    cdef int[:] Aindices = A.indices
    cdef int[:] Aindptr = A.indptr

    cdef double[:] ATdata = AT.data
    cdef int[:] ATindices = AT.indices
    cdef int[:] ATindptr = AT.indptr

    cdef double* Adatap = &Adata[0]
    cdef int* Aindicesp = &Aindices[0]
    cdef double* ATdatap = &ATdata[0]
    cdef int* ATindicesp = &ATindices[0]
    cdef int* ATindptrp = &ATindptr[0]

    cdef double* dp = &d[0]

    cdef int m = Aindptr.shape[0] - 1
    cdef int n = d.shape[0]
    cdef int nnz = Adata.shape[0]
    out = np.zeros((m,m))
    cdef double[:, :] out_view = out
    cdef double* outp = &out_view[0,0]

    cdef int AT_idx, A_idx
    cdef int AT_row, A_col
    cdef int i, j, k
    cdef double A_val, AT_val

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
