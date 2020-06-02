from typing import List, Union

import numpy as np


def check_1d(vec: Union[np.ndarray, List]) -> np.ndarray:
    """

    Parameters
    ----------
    vec: array-like

    Returns
    -------
    Numpy array that is either 1d or 2d with second dimension 1, depending on input.
    If vec cannot be coerced to that shape, errors.
    """
    vec = np.asarray(vec)
    if vec.ndim == 1 or (vec.ndim == 2 and vec.shape[1] == 1):
        return vec
    raise ValueError(f"""Expected vec to be a vector, but it has shape {vec.shape}.""")


def rmatmul_vector_only(mat, other: Union[np.ndarray, List]) -> np.ndarray:
    other = check_1d(other)
    ndim = other.ndim
    other = np.atleast_1d(np.squeeze(other))
    res = mat.transpose_dot_vec(other)
    if ndim == 1:
        return res
    else:
        return res[None, :]
