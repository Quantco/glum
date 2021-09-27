import logging
from typing import Union

import numpy as np
import pandas as pd
from quantcore.matrix import MatrixBase, StandardizedMatrix
from scipy import sparse

_logger = logging.getLogger(__name__)


def _asanyarray(x, **kwargs):
    """``np.asanyarray`` with passthrough for scalars."""
    return x if pd.api.types.is_scalar(x) else np.asanyarray(x, **kwargs)


def _align_df_dtypes(df, dtypes) -> pd.DataFrame:
    """Align data types for prediction.

    This function checks that columns are numeric if expected to and that
    categorical columns have the right categories in the right order. Invalid
    entries will be cast to ``numpy.nan``.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected `pandas.DataFrame'; got {type(df)}.")
    if dtypes.keys() - set(df.columns):
        raise KeyError(f"Missing columns: {dtypes.keys() - set(df.columns)}.")

    changed_dtypes = {}

    numeric_dtypes = [
        column
        for column, dtype in dtypes.items()
        if pd.api.types.is_numeric_dtype(dtype)
    ]

    categorical_dtypes = [
        column
        for column, dtype in dtypes.items()
        if pd.api.types.is_categorical_dtype(dtype)
    ]

    for column in numeric_dtypes:
        if not pd.api.types.is_numeric_dtype(df[column]):
            _logger.warning(f"Casting {column} to numeric.")
            changed_dtypes[column] = pd.to_numeric(df[column], errors="coerce")
    for column in categorical_dtypes:
        if not pd.api.types.is_categorical_dtype(df[column]):
            _logger.warning(f"Casting {column} to categorical.")
            changed_dtypes[column] = df[column].astype(dtypes[column])
        elif list(df[column].cat.categories) != list(dtypes[column].categories):
            _logger.warning(f"Aligning categories of {column}.")
            changed_dtypes[column] = df[column].cat.set_categories(
                dtypes[column].categories
            )

    if changed_dtypes:
        df = df.assign(**changed_dtypes)
    if df.columns.to_list != list(dtypes.keys()):
        _logger.warning(f"Reordering columns to {list(dtypes.keys())}.")
        df = df[list(dtypes.keys())]

    return df


def _safe_lin_pred(
    X: Union[MatrixBase, StandardizedMatrix],
    coef: np.ndarray,
    offset: np.ndarray = None,
) -> np.ndarray:
    """Compute the linear predictor taking care if intercept is present."""
    idx_offset = 0 if X.shape[1] == coef.shape[0] else 1
    nonzero_coefs = np.where(coef[idx_offset:] != 0.0)[0].astype(np.int32)
    res = X.matvec(coef[idx_offset:], cols=nonzero_coefs)

    if idx_offset == 1:
        res += coef[0]
    if offset is not None:
        return res + offset
    return res


def _safe_sandwich_dot(
    X: Union[MatrixBase, StandardizedMatrix],
    d: np.ndarray,
    rows: np.ndarray = None,
    cols: np.ndarray = None,
    intercept=False,
) -> np.ndarray:
    """
    Compute sandwich product ``X.T @ diag(d) @ X``.

    With ``intercept=True``, ``X`` is treated as if a column of 1 were appended
    as first column of ``X``. ``X`` can be sparse; ``d`` must be an ndarray.
    Always returns an ndarray.
    """
    result = X.sandwich(d, rows, cols)
    if isinstance(result, sparse.dia_matrix):
        result = result.A

    if intercept:
        dim = result.shape[0] + 1
        res_including_intercept = np.empty((dim, dim), dtype=X.dtype)
        res_including_intercept[0, 0] = d.sum()
        res_including_intercept[1:, 0] = X.transpose_matvec(d, rows, cols)
        res_including_intercept[0, 1:] = res_including_intercept[1:, 0]
        res_including_intercept[1:, 1:] = result
    else:
        res_including_intercept = result
    return res_including_intercept


def _safe_toarray(X) -> np.ndarray:
    """Return a numpy array."""
    if sparse.issparse(X):
        return X.toarray()
    else:
        return np.asarray(X)
