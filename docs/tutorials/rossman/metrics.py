import numpy as np


def root_mean_squared_percentage_error(y_true, y_pred):
    """Compute RMSPE."""
    mask = y_true > 0.0
    return np.sqrt(
        np.mean(
            np.power(
                (np.asanyarray(y_true[mask]) - np.asanyarray(y_pred[mask]))
                / np.asanyarray(y_true[mask]),
                2,
            )
        )
    )
