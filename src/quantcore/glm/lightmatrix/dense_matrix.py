from typing import List, Optional, Union

import numpy as np

from .matrix_base import MatrixBase
from .util import (
    _check_matvec_out_shape,
    _check_transpose_matvec_out_shape,
    _setup_restrictions,
)


class DenseMatrix(np.ndarray, MatrixBase):
    """
    Lightweight wrapper around numpy.ndarray to implement certain methods.

    * sandwich product
    * getcol
    * toarray

    np.ndarray subclassing is explained here:
    https://docs.scipy.org/doc/numpy/user/basics.subclassing.html#slightly-more-
    realistic-example-attribute-added-to-existing-array
    """

    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        if not np.issubdtype(obj.dtype, np.floating):
            raise NotImplementedError("DenseMatrix is only implemented for float data")
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return

    def getcol(self, i):
        return self[:, [i]]

    def toarray(self):
        return np.asarray(self)

    def sandwich(
        self, d: np.ndarray, rows: np.ndarray = None, cols: np.ndarray = None
    ) -> np.ndarray:
        d = np.asarray(d)
        rows, cols = _setup_restrictions(self.shape, rows, cols)
        return (self[np.ix_(rows, cols)].T * d[rows]) @ self[np.ix_(rows, cols)]

    def get_col_stds(self, weights: np.ndarray, col_means: np.ndarray) -> np.ndarray:
        sqrt_arg = self.T ** 2 @ weights - col_means ** 2
        # Minor floating point errors above can result in a very slightly
        # negative sqrt_arg (e.g. -5e-16). We just set those values equal to
        # zero.
        sqrt_arg[sqrt_arg < 0] = 0
        return np.sqrt(sqrt_arg)

    def matvec_helper(
        self,
        vec: Union[List, np.ndarray],
        rows: Optional[np.ndarray],
        cols: Optional[np.ndarray],
        out: Optional[Union[np.ndarray]],
        transpose: bool,
    ):
        rows, cols = _setup_restrictions(self.shape, rows, cols)
        X = self[np.ix_(rows, cols)].T if transpose else self[np.ix_(rows, cols)]
        vec = np.asarray(vec[rows]) if transpose else np.asarray(vec[cols])

        if out is None:
            out = X.dot(vec)
        else:
            out += X.dot(vec)
        return out

    def transpose_matvec(
        self,
        vec: Union[np.ndarray, List],
        rows: np.ndarray = None,
        cols: np.ndarray = None,
        out: np.ndarray = None,
    ) -> np.ndarray:
        _check_transpose_matvec_out_shape(self, out)
        return self.matvec_helper(vec, rows, cols, out, True)

    def matvec(
        self,
        vec: Union[np.ndarray, List],
        cols: np.ndarray = None,
        out: np.ndarray = None,
    ) -> np.ndarray:
        _check_matvec_out_shape(self, out)
        return self.matvec_helper(vec, None, cols, out, False)
