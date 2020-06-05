from typing import List, Union

import numpy as np

from quantcore.glm.matrix.sandwich.sandwich import dense_sandwich

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

    def sandwich(self, d: np.ndarray):
        d = np.asarray(d)
        return dense_sandwich(self, d)

    def get_col_stds(self, weights: np.ndarray, col_means: np.ndarray) -> np.ndarray:
        # TODO: avoid copying X - the X ** 2 makes a copy
        return np.sqrt((self ** 2).T.dot(weights) - col_means ** 2)

    def transpose_dot(self, vec: Union[np.ndarray, List]) -> np.ndarray:
        return self.T.dot(vec)

    def scale_cols_inplace(self, col_scaling: np.ndarray) -> None:
        self *= col_scaling[None, :]
