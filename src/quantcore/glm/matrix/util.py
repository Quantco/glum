import numpy as np


def setup_restrictions(shape, rows, cols):
    if rows is None:
        rows = np.arange(shape[0], dtype=np.int32)
    if cols is None:
        cols = np.arange(shape[1], dtype=np.int32)
    return rows, cols
