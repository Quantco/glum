import logging
import warnings
from collections.abc import Sequence
from functools import wraps
from typing import Union

import numpy as np
import pandas as pd
from scipy import sparse
from tabmat import MatrixBase, StandardizedMatrix

_logger = logging.getLogger(__name__)


def _asanyarray(x, **kwargs):
    """``np.asanyarray`` with passthrough for scalars."""
    return x if pd.api.types.is_scalar(x) else np.asanyarray(x, **kwargs)


def _align_df_categories(
    df, dtypes, has_missing_category, cat_missing_method
) -> pd.DataFrame:
    """Align data types for prediction.

    This function checks that categorical columns have same categories in the
    same order as specified in ``dtypes``. If an entry has a category not
    specified in ``dtypes``, it will be set to ``numpy.nan``.

    Parameters
    ----------
    df : pandas.DataFrame
    dtypes : Dict[str, Union[str, type, pandas.core.dtypes.base.ExtensionDtype]]
    has_missing_category : Dict[str, bool]
    missing_method : str
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected `pandas.DataFrame'; got {type(df)}.")

    changed_dtypes = {}

    categorical_dtypes = [
        column
        for column, dtype in dtypes.items()
        if isinstance(dtype, pd.CategoricalDtype) and (column in df)
    ]

    for column in categorical_dtypes:
        if not isinstance(df[column].dtype, pd.CategoricalDtype):
            _logger.info(f"Casting {column} to categorical.")
            changed_dtypes[column] = df[column].astype(dtypes[column])
        elif list(df[column].cat.categories) != list(dtypes[column].categories):
            _logger.info(f"Aligning categories of {column}.")
            changed_dtypes[column] = df[column].cat.set_categories(
                dtypes[column].categories
            )
        else:
            continue

        if cat_missing_method == "convert" and not has_missing_category[column]:
            unseen_categories = set(df[column].unique())
            unseen_categories = unseen_categories - set(dtypes[column].categories)
        else:
            unseen_categories = set(df[column].dropna().unique())
            unseen_categories = unseen_categories - set(dtypes[column].categories)

        if unseen_categories:
            raise ValueError(
                f"Column {column} contains unseen categories: {unseen_categories}."
            )

    if changed_dtypes:
        df = df.assign(**changed_dtypes)

    return df


def _add_missing_categories(
    df,
    dtypes,
    feature_names: Sequence[str],
    categorical_format: str,
    cat_missing_name: str,
) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected `pandas.DataFrame'; got {type(df)}.")

    changed_dtypes = {}

    categorical_dtypes = [
        column
        for column, dtype in dtypes.items()
        if isinstance(dtype, pd.CategoricalDtype) and (column in df)
    ]

    for column in categorical_dtypes:
        if (
            categorical_format.format(name=column, category=cat_missing_name)
            in feature_names
        ):
            if cat_missing_name in df[column].cat.categories:
                raise ValueError(
                    f"Missing category {cat_missing_name} already exists in {column}."
                )
            _logger.info(f"Adding missing category {cat_missing_name} to {column}.")
            changed_dtypes[column] = df[column].cat.add_categories(cat_missing_name)
            if df[column].isnull().any():
                changed_dtypes[column] = changed_dtypes[column].fillna(cat_missing_name)

    if changed_dtypes:
        df = df.assign(**changed_dtypes)

    return df


def _expand_categorical_penalties(
    penalty, X, drop_first, has_missing_category
) -> Union[np.ndarray, str]:
    """Determine penalty matrices ``P1`` or ``P2`` after expanding categorical columns.

    If ``P1`` or ``P2`` has the same shape as ``X`` before expanding categorical
    columns, we assume that the penalty at the location of categorical columns
    is the same for all levels.
    """
    if isinstance(penalty, str):
        return penalty
    if not sparse.issparse(penalty):
        penalty = np.asanyarray(penalty)

    if penalty.shape[0] == X.shape[1]:
        if penalty.ndim == 2:
            raise ValueError(
                "When the penalty is two-dimensional, it must have the "
                "same length as the number of columns in the design "
                "matrix `X` after expanding categorical columns."
            )

        expanded_penalty = []  # type: ignore

        for element, (column, dt) in zip(penalty, X.dtypes.items()):
            if isinstance(dt, pd.CategoricalDtype):
                length = len(dt.categories) + has_missing_category[column] - drop_first
                expanded_penalty.extend(element for _ in range(length))
            else:
                expanded_penalty.append(element)

        return np.array(expanded_penalty)

    else:
        return penalty


def _is_contiguous(X) -> bool:
    if isinstance(X, np.ndarray):
        return X.flags["C_CONTIGUOUS"] or X.flags["F_CONTIGUOUS"]
    elif isinstance(X, pd.DataFrame):
        return _is_contiguous(X.values)
    else:
        # If not a numpy array or pandas data frame, we assume it is contiguous.
        return True


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
        result = result.toarray()

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


def _positional_args_deprecated(unchanged_args=(), unchanged_args_number=None):
    """
    Raise a FutureWarning if more than `unchanged_args_number` positional
    arguments are passed.
    """
    if unchanged_args_number is None:
        unchanged_args_number = len(unchanged_args)

    def decorator(func):
        first_part = "Arguments" if unchanged_args else "All arguments"
        exceptions = (
            " other than " + ", ".join(f"`{arg}`" for arg in unchanged_args)
            if unchanged_args
            else ""
        )

        msg = (
            f"{first_part} to `{func.__qualname__}`{exceptions} "
            "will become keyword-only in 3.0.0."
        )

        @wraps(func)
        def wrapper(*args, **kwargs):
            if len(args) > unchanged_args_number + 1:  # +1 for self
                warnings.warn(
                    msg,
                    FutureWarning,
                )
            return func(*args, **kwargs)

        return wrapper

    return decorator
