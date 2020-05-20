from typing import Iterable, Tuple

import numpy as np

from glm_benchmarks.matrix.standardize import one_over_var_inf_to_zero
from glm_benchmarks.sandwich.sandwich import dense_sandwich

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

    def standardize(self, weights: Iterable, scale_predictors: bool) -> Tuple:
        col_means = self.T.dot(weights)[None, :]
        self -= col_means
        if scale_predictors:
            # TODO: avoid copying X -- the X ** 2 makes a copy
            col_stds = np.sqrt((self ** 2).T.dot(weights))
            self *= one_over_var_inf_to_zero(col_stds)
        else:
            col_stds = np.ones(self.shape[1], dtype=self.dtype)
        return self, col_means, col_stds

    def unstandardize(self, col_means, col_stds):
        self *= col_stds
        self += col_means
        return self
