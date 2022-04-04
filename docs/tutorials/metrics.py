import numpy as np


def root_mean_squared_percentage_error(y_true, y_pred):
    """Compute RMSPE."""
    mask = y_true > 0.0
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    score = (y_true - y_pred) / y_pred
    return np.sqrt(np.mean(score**2)) * 100
