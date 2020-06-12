from typing import List, Union

import numpy as np

from quantcore.glm.matrix.ext.dense import dense_matvec, dense_rmatvec, dense_sandwich

from .matrix_base import MatrixBase
from .util import setup_restrictions


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

    def sandwich(
        self, d: np.ndarray, rows: np.ndarray = None, cols: np.ndarray = None
    ) -> np.ndarray:
        d = np.asarray(d)
        if rows is None:
            rows = np.arange(self.shape[0], dtype=np.int32)
        if cols is None:
            cols = np.arange(self.shape[1], dtype=np.int32)
        return dense_sandwich(self, d, rows, cols)

    def get_col_stds(self, weights: np.ndarray, col_means: np.ndarray) -> np.ndarray:
        # TODO: avoid copying X - the X ** 2 makes a copy
        return np.sqrt((self ** 2).T.dot(weights) - col_means ** 2)

    def dot_helper(self, vec, rows, cols, transpose):
        # Because the dense_rmatvec takes a row array and col array, it has
        # added overhead compared to a raw matrix vector product. So, when
        # we're not filtering at all, let's just use default numpy dot product.
        #
        # TODO: related to above, it could be nice to have a version that only
        # filters rows and a version that only filters columns. How do we do
        # this without an explosion of code?
        X = self.T if transpose else self
        vec = np.asarray(vec)

        # NOTE: We assume that rows and cols are unique
        unrestricted_rows = rows is None or rows.shape[0] == self.shape[0]
        unrestricted_cols = cols is None or cols.shape[0] == self.shape[1]
        if unrestricted_rows and unrestricted_cols:
            return X.toarray().dot(vec)
        else:
            rows, cols = setup_restrictions(self.shape, rows, cols)
            fast_fnc = dense_rmatvec if transpose else dense_matvec
            if vec.ndim == 1:
                return fast_fnc(self, vec, rows, cols)
            elif vec.ndim == 2 and vec.shape[1] == 1:
                return fast_fnc(self, vec[:, 0], rows, cols)[:, None]
            subset = self[np.ix_(rows, cols)]
            return subset.T.dot(vec[rows]) if transpose else subset.dot(vec[cols])

    def transpose_dot(
        self,
        vec: Union[np.ndarray, List],
        rows: np.ndarray = None,
        cols: np.ndarray = None,
    ) -> np.ndarray:
        return self.dot_helper(vec, rows, cols, True)

    def dot(
        self,
        vec: Union[np.ndarray, List],
        rows: np.ndarray = None,
        cols: np.ndarray = None,
    ) -> np.ndarray:
        return self.dot_helper(vec, rows, cols, False)

    def scale_cols_inplace(self, col_scaling: np.ndarray) -> None:
        self *= col_scaling[None, :]
