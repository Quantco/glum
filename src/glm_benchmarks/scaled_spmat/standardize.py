import numpy as np
from scipy import sparse as sps

from glm_benchmarks.scaled_spmat import ColScaledSpMat


def zero_center(mat: sps.spmatrix) -> ColScaledSpMat:
    """
    >>> mat = sps.eye(2)
    >>> centered = zero_center(mat)
    >>> centered.A
    array([[ 0.5, -0.5],
           [-0.5,  0.5]])
    """
    means = np.asarray(mat.mean(0))
    return ColScaledSpMat(mat, -means)


def standardize(mat: sps.spmatrix) -> ColScaledSpMat:
    """
    >>> mat = sps.eye(2)
    >>> standardize(mat).A
    array([[ 1., -1.],
           [-1.,  1.]])
    """
    centered_mat = zero_center(mat)
    st_devs = np.sqrt(centered_mat.power(2).mean(0))
    return centered_mat.multiply((1 / st_devs)[None, :])
