import numpy as np
from scipy import sparse as sps

from glm_benchmarks.scaled_spmat import ColScaledSpMat


def zero_center(mat: sps.spmatrix, weights: np.ndarray = None) -> ColScaledSpMat:
    """
    >>> mat = sps.eye(2)
    >>> centered = zero_center(mat)
    >>> centered.A
    array([[ 0.5, -0.5],
           [-0.5,  0.5]])
    >>> zero_center(mat, weights=[3, 1]).A
    array([[ 0.25, -0.25],
           [-0.75,  0.75]])
    """
    if weights is None:
        means = np.asarray(mat.mean(0))
    else:
        means = mat.T.dot(weights) / np.sum(weights)
    return ColScaledSpMat(mat, -means)


def standardize(mat: sps.spmatrix, weights: np.ndarray = None) -> ColScaledSpMat:
    """
    >>> mat = sps.eye(2)
    >>> standardize(mat).A
    array([[ 1., -1.],
           [-1.,  1.]])
    >>> standardize(mat, weights=[9, 1]).A
    array([[ 0.33333333, -0.33333333],
           [-3.        ,  3.        ]])
    """
    centered_mat = zero_center(mat, weights)
    mat_squared = centered_mat.power(2)
    if weights is None:
        avg_mat_squared = mat_squared.mean(0)
    else:
        avg_mat_squared = mat_squared.T.dot(weights) / np.sum(weights)
    st_devs = np.sqrt(avg_mat_squared)
    return centered_mat.multiply((1 / st_devs)[None, :])
