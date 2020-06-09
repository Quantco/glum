import numpy as np
from scipy import sparse as sps


def one_over_var_inf_to_zero(arr: np.ndarray) -> np.ndarray:
    zeros = np.where(arr == 0)
    with np.errstate(divide="ignore"):
        one_over = 1 / arr
    one_over[zeros] = 0
    return one_over


def _scale_csc_columns_inplace(mat: sps.csc_matrix, v: np.ndarray):
    assert isinstance(mat, sps.csc_matrix)
    for i in range(mat.shape[1]):
        start_idx = mat.indptr[i]
        end_idx = mat.indptr[i + 1]
        mat.data[start_idx:end_idx] *= v[i]
