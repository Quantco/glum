from typing import Tuple

import numpy as np
from scipy import sparse as sps

from glm_benchmarks.scaled_spmat import ColScaledSpMat


def zero_center(
    mat: sps.spmatrix, weights: np.ndarray = None
) -> Tuple[ColScaledSpMat, np.ndarray]:
    """
    >>> mat = sps.eye(2)
    >>> centered = zero_center(mat)[0]
    >>> centered.A
    array([[ 0.5, -0.5],
           [-0.5,  0.5]])
    >>> zero_center(mat, weights=[3, 1])[0].A
    array([[ 0.25, -0.25],
           [-0.75,  0.75]])
    """
    if weights is None:
        means = np.asarray(mat.mean(0))
    else:
        means = (mat.T.dot(weights) / np.sum(weights))[None, :]
    return ColScaledSpMat(mat, -means), means


def one_over_var_inf_to_zero(arr: np.ndarray) -> np.ndarray:
    zeros = np.where(arr == 0)
    one_over = 1 / arr
    one_over[zeros] = 0
    return one_over


def standardize(
    mat: sps.spmatrix, weights: np.ndarray = None
) -> Tuple[ColScaledSpMat, np.ndarray, np.ndarray]:
    """
    >>> mat = sps.eye(2)
    >>> R, means, st_devs = standardize(mat)
    >>> R.A
    array([[ 1., -1.],
           [-1.,  1.]])
    >>> R, means, st_devs = standardize(mat, weights=np.array([9, 1]))
    >>> means
    array([0.9, 0.1])
    >>> st_devs
    array([0.3, 0.3])
    >>> R.A
    array([[ 0.33333333, -0.33333333],
           [-3.        ,  3.        ]])
    """
    centered_mat, means = zero_center(mat, weights)

    # TODO: this copies!!
    mat_squared = centered_mat.power(2)
    if weights is None:
        avg_mat_squared = mat_squared.mean(0)
    else:
        avg_mat_squared = mat_squared.T.dot(weights) / np.sum(weights)
    st_devs = np.squeeze(np.array(np.sqrt(avg_mat_squared)))

    # Zero-variance columns can't be scaled to have a standard deviation of 1, so
    # leave them as zeros
    scaling_factor = one_over_var_inf_to_zero(st_devs)

    return centered_mat @ sps.diags(scaling_factor), means, st_devs
