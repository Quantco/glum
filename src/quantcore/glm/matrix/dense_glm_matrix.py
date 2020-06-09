from typing import List, Union

import numpy as np
from glm_benchmarks.matrix.sandwich.sandwich import (
    dense_matvec,
    dense_rmatvec,
    dense_sandwich,
)

from .matrix_base import MatrixBase


class DenseGLMDataMatrix(np.ndarray, MatrixBase):
    """
    We want to add several function to a numpy ndarray so that it conforms to
    the sparse matrix interface we expect for the GLM algorithms below:

    * sandwich product
    * getcol
    * toarray

    np.ndarray subclassing is explained here: https://docs.scipy.org/doc/numpy/user/basics.subclassing.html#slightly-more-realistic-example-attribute-added-to-existing-array
    """

    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        if not np.issubdtype(obj.dtype, np.floating):
            raise NotImplementedError(
                "DenseGLMDataMatrix is only implemented for float data"
            )
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return

    def getcol(self, i):
        return self[:, [i]]

    def toarray(self):
        return np.asarray(self)

    def sandwich(self, d: np.ndarray, rows: np.ndarray, cols: np.ndarray):
        d = np.asarray(d)
        return dense_sandwich(self, d, rows, cols)

    def limited_rmatvec(self, v: np.ndarray, rows: np.ndarray, cols: np.ndarray):
        # Because the dense_rmatvec takes a row array and col array, it has
        # added overhead compared to a raw matrix vector product. So, when
        # we're not filtering at all, let's just use default numpy dot product.
        #
        # TODO: related to above, it could be nice to have a version that only
        # filters rows and a version that only filters columns. How do we do
        # this without an explosion of code?
        if rows.shape[0] == self.shape[0] and cols.shape[0] == self.shape[1]:
            return self.T.dot(v)
        else:
            return dense_rmatvec(self, v, rows, cols)

    def limited_matvec(self, v: np.ndarray, rows: np.ndarray, cols: np.ndarray):
        if rows.shape[0] == self.shape[0] and cols.shape[0] == self.shape[1]:
            return self.dot(v)
        else:
            return dense_matvec(self, v, rows, cols)

    def get_col_stds(self, weights: np.ndarray, col_means: np.ndarray) -> np.ndarray:
        # TODO: avoid copying X - the X ** 2 makes a copy
        return np.sqrt((self ** 2).T.dot(weights) - col_means ** 2)

    def transpose_dot(self, vec: Union[np.ndarray, List]) -> np.ndarray:
        return self.T.dot(vec)

    def scale_cols_inplace(self, col_scaling: np.ndarray) -> None:
        self *= col_scaling[None, :]
