import numpy as np
from scipy import sparse as sps


def spmatrix_col_var(x: sps.spmatrix) -> np.ndarray:
    mean = x.mean(0)
    sd = x.power(2).mean(0) - sps.csc_matrix(mean).power(2)
    return np.squeeze(np.asarray(sd))


def spmatrix_col_sd(x: sps.spmatrix) -> np.ndarray:
    return np.sqrt(spmatrix_col_var(x))
