from typing import Any, Tuple

import numpy as np
from scipy import sparse as sps


def one_over_var_inf_to_zero(arr: np.ndarray) -> np.ndarray:
    zeros = np.where(arr == 0)
    with np.errstate(divide="ignore"):
        one_over = 1 / arr
    one_over[zeros] = 0
    return one_over


def standardize(
    mat, weights: np.ndarray, scale_predictors: bool
) -> Tuple[Any, np.ndarray, np.ndarray]:
    """
    >>> mat = sps.eye(2)
    >>> R, means, st_devs = standardize(mat)
    >>> R.A
    array([[ 1., -1.],
           [-1.,  1.]])
    >>> R, means, st_devs = standardize(mat, weights=np.array([9., 1.]))
    >>> means
    array([[0.9, 0.1]])
    >>> st_devs
    array([0.3, 0.3])
    >>> R.A
    array([[ 0.33333333, -0.33333333],
           [-3.        ,  3.        ]])
    """
    return mat.standardize(weights, scale_predictors)


def _scale_csc_columns_inplace(mat: sps.spmatrix, v: np.ndarray):
    assert isinstance(mat, sps.csc_matrix)
    for i in range(mat.shape[1]):
        start_idx = mat.indptr[i]
        end_idx = mat.indptr[i + 1]
        mat.data[start_idx:end_idx] *= v[i]
