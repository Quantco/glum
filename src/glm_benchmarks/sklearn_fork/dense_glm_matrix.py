import numpy as np

from glm_benchmarks.scaled_spmat.standardize import one_over_var_inf_to_zero


class DenseGLMDataMatrix(np.ndarray):
    """
    We want to add several function to a numpy ndarray so that it conforms to
    the sparse matrix interface we expect for the GLM algorithms below:

    * sandwich product
    * getcol
    * toarray

    np.ndarray subclassing is explained here: https://docs.scipy.org/doc/numpy/user/basics.subclassing.html#slightly-more-realistic-example-attribute-added-to-existing-array
    """

    def __new__(cls, input_array):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return

    def getcol(self, j):
        return self[:, j]

    def toarray(self):
        return self

    def sandwich(self, d):
        sqrtD = np.sqrt(d)[:, np.newaxis]
        xd = self * sqrtD
        return xd.T @ xd

    def standardize(self, weights, scale_predictors):
        col_means = self.T.dot(weights)[None, :]
        self -= col_means
        if scale_predictors:
            # TODO: avoid copying X -- the X ** 2 makes a copy
            col_stds = np.sqrt((self ** 2).T.dot(weights))
            self *= one_over_var_inf_to_zero(col_stds)
        else:
            col_stds = np.ones(self.shape[1])
        return self, col_means, col_stds

    def unstandardize(self, col_means, col_stds):
        self *= col_stds
        self += col_means
        return self
