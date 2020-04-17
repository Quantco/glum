import numpy as np
from scipy.linalg.blas import dsyrk


def sandwich_dot_whiten(x: np.ndarray, d: np.ndarray) -> np.ndarray:
    x_tilde = x * np.sqrt(d[:, None])
    # Returns a triangular matrix, need to fix
    out = dsyrk(alpha=1.0, a=x_tilde.T)
    out += out.T
    out -= np.diag(np.diag(out) / 2)
    return out


def sandwich_dot_dumb(x: np.ndarray, d: np.ndarray) -> np.ndarray:
    return (x.T * d) @ x
