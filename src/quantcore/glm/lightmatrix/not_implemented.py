import scipy.sparse as sps

from .matrix_base import MatrixBase


class SparseMatrix(sps.csc_matrix, MatrixBase):
    """Placeholder for SparseMatrix."""

    def __init__(self, arg1, shape=None, dtype=None, copy=False):
        raise NotImplementedError


class CategoricalMatrix(MatrixBase):
    """Placeholder for CategoricalMatrix."""

    def __init__(self, cat_vec, dtype):
        raise NotImplementedError


class SplitMatrix(MatrixBase):
    """Placeholder for SplitMatrix."""

    def __init__(self, matrices, indices):
        raise NotImplementedError
