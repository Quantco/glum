"""
This implements the algorithm given in the glmnet paper.
"""
import numpy as np


def soft_threshold(z: float, threshold: float) -> float:
    if np.abs(z) <= threshold:
        return 0.0
    if z > 0:
        return z - threshold
    return z + threshold


def get_coordinate_wise_update(
    y: np.ndarray,
    x: np.ndarray,
    j: int,
    beta: np.ndarray,
    alpha: float,
    l1_ratio: float,
):
    prediction_not_j = x.dot(beta) - x[:, j] * beta[j]
    resid = y - prediction_not_j
    mean_resid = x[:, j].dot(resid) / len(y)
    numerator = soft_threshold(mean_resid, l1_ratio * alpha)
    denominator = 1 + alpha * (1 - l1_ratio)
    return numerator / denominator


def cd_inner_update(
    y: np.ndarray, x: np.ndarray, beta: np.ndarray, alpha: float, l1_ratio: float
) -> np.ndarray:
    for i in range(len(beta)):
        beta[i] = get_coordinate_wise_update(y, x, i, beta, alpha, l1_ratio)
    return beta


def fit_glmnet(
    y: np.ndarray, x: np.ndarray, alpha: float, l1_ratio: float, n_iters: int = 10
) -> np.ndarray:
    beta = np.zeros(x.shape[1])
    for i in range(n_iters):
        beta = cd_inner_update(y, x, beta, alpha, l1_ratio)
    return beta
