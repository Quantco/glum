# distutils: extra_compile_args=-fopenmp -O3 -ffast-math -march=native -mavx
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

    sparse_status_t mkl_sparse_d_export_csr(
        const sparse_matrix_t source,
        sparse_index_base_t *indexing,
        MKL_INT *rows,
        MKL_INT *cols,
        MKL_INT **rows_start,
        MKL_INT **rows_end,
        MKL_INT **col_indx,
        double **values
    )

    sparse_status_t mkl_sparse_d_export_csc(
        const sparse_matrix_t source,
        sparse_index_base_t *indexing,
        MKL_INT *rows,
        MKL_INT *cols,
        MKL_INT **cols_start,
        MKL_INT **cols_end,
        MKL_INT **row_indx,
        double **values
    )

    sparse_status_t mkl_sparse_d_mv(
        sparse_operation_t operation,
        double alpha,
        const sparse_matrix_t A,
        matrix_descr descr,
        const double *x,
        double beta,
        double *y
    )

    sparse_status_t mkl_sparse_spmm(
        sparse_operation_t operation,
        const sparse_matrix_t A,
        const sparse_matrix_t B,
        sparse_matrix_t *C
    )

    sparse_status_t mkl_sparse_d_spmmd(
        sparse_operation_t operation,
        const sparse_matrix_t A,
        const sparse_matrix_t B,
        sparse_layout_t layout,
        double *C,
        MKL_INT ldc # 'leading dimension' size of the matrix C
    )

    sparse_status_t mkl_sparse_sypr(
        sparse_operation_t operation, 
        const sparse_matrix_t A, 
        const sparse_matrix_t B, 
        const matrix_descr descrB, 
        sparse_matrix_t *C, 
        const sparse_request_t request
    )

    sparse_status_t mkl_sparse_d_syrkd(
        sparse_operation_t operation,
        const sparse_matrix_t A,
        double alpha,
        double beta,
        double *C,
        sparse_layout_t layout,
        MKL_INT ldc
    )

cdef struct matrix_descr:
    sparse_matrix_type_t type
    sparse_fill_mode_t mode  # upper or lower triangular part of the matrix ( for triangular / symmetric / hermitian case)
    sparse_diag_type_t diag  # unit or non-unit diagonal ( for triangular / symmetric / hermitian case)


class MKLCallError(Exception):
    pass


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


def mkl_matvec(A_py, x, transpose=False):
    return MklSparseMatrix(A_py).dot(x, transpose)


def mkl_matmat(A_py, B_py, transpose=False, return_dense=False):

#    if A_py.getformat() != B_py.getformat():
#        raise TypeError(
#            'The storage formats of the two matrice must coincide. '
#            f'A: {A_py.getformat()}, B {B_py.getformat()}'
#        )

    if A_py.shape[1 - int(transpose)] != B_py.shape[0]:
        raise TypeError("The matrices have incompatible dimensions.")

    cdef sparse_operation_t operation
    cdef bint transpose_flag = int(transpose)
    cdef sparse_matrix_t C
    cdef sparse_matrix_t A = to_mkl_matrix(A_py)
    cdef sparse_matrix_t B = to_mkl_matrix(B_py)
    cdef sparse_layout_t layout
    cdef MKL_INT nrow_C
    cdef double[:, :] C_view

    if transpose:
        operation = SPARSE_OPERATION_TRANSPOSE
    else:
        operation = SPARSE_OPERATION_NON_TRANSPOSE

    if return_dense:
        layout = SPARSE_LAYOUT_ROW_MAJOR
        C_py = np.zeros((A_py.shape[transpose], B_py.shape[1]))
        nrow_C = C_py.shape[1]
        C_view = C_py
        mkl_sparse_d_spmmd(operation, A, B, layout, &C_view[0, 0], nrow_C)
    else:
        mkl_sparse_spmm(operation, A, B, &C)
        C_py = to_scipy_matrix(C, A_py.getformat())

    return C_py


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


cdef to_scipy_matrix(sparse_matrix_t A, format):
    cdef MKL_INT rows
    cdef MKL_INT cols
    cdef sparse_index_base_t base_index=SPARSE_INDEX_BASE_ZERO
    cdef MKL_INT* start
    cdef MKL_INT* end
    cdef MKL_INT* index
    cdef double* values

    if format == 'csr':
        mkl_sparse_d_export = mkl_sparse_d_export_csr
        sp_sparse_matrix = sp.sparse.csr_matrix
    else:
        mkl_sparse_d_export = mkl_sparse_d_export_csc
        sp_sparse_matrix = sp.sparse.csc_matrix

    export_status = mkl_sparse_d_export(
        A, &base_index, &rows, &cols, &start, &end, &index, &values
    )
    if export_status != SPARSE_STATUS_SUCCESS:
        print(export_status)
        raise MKLCallError("Exporting from an MKL sparse matrix failed.")

    cdef int nnz = start[rows]
    data = to_numpy_array(values, nnz)
    indices = to_numpy_array(index, nnz)
    indptr = np.empty(rows + 1, dtype=np.int32)
    indptr[:-1] = to_numpy_array(start, rows)
    indptr[-1] = nnz
    A_py = sp_sparse_matrix((data, indices, indptr), shape=(rows, cols))
    return A_py


ctypedef fused int_or_double:
    MKL_INT
    double

cdef to_numpy_array(int_or_double* c_array, arr_length):
    return np.asarray(<int_or_double[:arr_length]> c_array)


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


def mkl_sypr(A_py, B_py, transpose: bool=True):
    """Computes the symmetric triple product of a sparse matrix and a dense 
    matrix and stores the result as a dense matrix.

    C:=alpha*A*B*op(A) + beta*C
    or
    C:=alpha*op(A)*B*A + beta*C
    """
    cdef sparse_operation_t operation
    cdef sparse_matrix_t C
    cdef sparse_matrix_t A = to_mkl_matrix(A_py)
    cdef sparse_matrix_t B = to_mkl_matrix(B_py)

    if transpose:
        operation = SPARSE_OPERATION_TRANSPOSE
    else:
        operation = SPARSE_OPERATION_NON_TRANSPOSE

    cdef matrix_descr descrB
    descrB.type = SPARSE_MATRIX_TYPE_DIAGONAL
    descrB.mode = SPARSE_FILL_MODE_UPPER
    descrB.diag = SPARSE_DIAG_NON_UNIT


    cdef sparse_request_t request
    request = SPARSE_STAGE_NNZ_COUNT

    res = mkl_sparse_sypr(
        operation, 
        A, 
        B, 
        descrB,
        &C,
        request
    )

    C_py = to_scipy_matrix(C, A_py.getformat())

    return C_py


def mkl_syrkd(A_py, transpose: bool=True):
    """Computes the product of sparse matrix with its transpose 
    (or conjugate transpose) and stores the result as a dense matrix.

    C := beta*C + alpha*A*op(A)
    or
    C := beta*C + alpha*op(A)*A
    """
    cdef sparse_operation_t operation
    cdef sparse_matrix_t A = to_mkl_matrix(A_py)
    cdef double[:, :] C_view
    cdef double zero = 0
    cdef double one = 1

    if transpose:
        operation = SPARSE_OPERATION_TRANSPOSE
    else:
        operation = SPARSE_OPERATION_NON_TRANSPOSE

    layout = SPARSE_LAYOUT_ROW_MAJOR
    C_py = np.zeros((A_py.shape[transpose], A_py.shape[transpose]))
    nrow_C = C_py.shape[1]
    C_view = C_py

    res = mkl_sparse_d_syrkd(
        operation, 
        A,
        one,
        zero,
        &C_view[0, 0],
        layout,
        nrow_C,
    )

    return C_py

@cython.boundscheck(False)
@cython.wraparound(False)
def fast_row_scale(double[:] data, int[:] indices, double[:] scaling, bint inv):
    cdef int nnz = data.shape[0]
    if inv:
        for i in range(nnz):
            data[i] /= scaling[indices[i]]
    else:
        for i in range(nnz):
            data[i] *= scaling[indices[i]]

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
            A_val = Adata[A_idx]
            for AT_idx in range(ATindptr[k], ATindptr[k+1]):
                i = ATindices[AT_idx]
                if i < j:
                    continue
                AT_val = ATdata[AT_idx]
                out_view[i, j] = out_view[i, j] + AT_val * d[k] * A_val

    out += np.tril(out, -1).T
    return out

cdef dense_base(double[:,:] X, double[:] d, int n_A, int n_B):
    # Compute C

@cython.boundscheck(False)
@cython.wraparound(False)
def dense_sandwich(double[:,:] X, double[:] d):
    cdef int m = X.shape[1]
    cdef int n = X.shape[0]
    cdef int it = 0
    cdef int jt, kt, i, j, k
    return 0

    # while it < m:
    #     jt = 0
    #     while jt < n;
    #         
    #     it += tilesize
    # int it, jt, kt, i, j, k;
    # for (it = 0; it < m; it += tileSize) {
    #     for (jt = 0; jt < n; jt += tileSize) {
    #         for (kt = 0; kt < rowLen; kt += tileSize) {
    #             for (i = it; i < it + tileSize; ++i) {
    #                 for (j = jt; j < jt + tileSize; ++j) {
    #                     C[i*n + j] = 0.0;
    #                     for (k = kt; k < kt + tileSize; ++k) {
    #                         C[i*n + j] += A[i*rowLen + k]*B[j*rowLen + k];
