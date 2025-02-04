import logging
from collections.abc import Sequence
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import tabmat as tm
from scipy import sparse

_logger = logging.getLogger(__name__)


def align_df_categories(
    df, dtypes, has_missing_category, cat_missing_method
) -> pd.DataFrame:
    """Align data types for prediction.

    This function checks that categorical columns have same categories in the
    same order as specified in ``dtypes``. If an entry has a category not
    specified in ``dtypes``, it will be set to ``numpy.nan``.

    Parameters
    ----------
    df : pandas.DataFrame
    dtypes : Dict[str, str | type | pandas.core.dtypes.base.ExtensionDtype]
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


def add_missing_categories(
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


def expand_categorical_penalties(
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


def is_contiguous(X) -> bool:
    if isinstance(X, np.ndarray):
        return X.flags["C_CONTIGUOUS"] or X.flags["F_CONTIGUOUS"]
    elif isinstance(X, pd.DataFrame):
        return is_contiguous(X.values)
    else:
        # If not a numpy array or pandas data frame, we assume it is contiguous.
        return True


def safe_toarray(X) -> np.ndarray:
    """Return a numpy array."""
    if sparse.issparse(X):
        return X.toarray()
    else:
        return np.asarray(X)


def standardize(
    X: tm.MatrixBase,
    sample_weight: np.ndarray,
    center_predictors: bool,
    estimate_as_if_scaled_model: bool,
    lower_bounds: Optional[np.ndarray],
    upper_bounds: Optional[np.ndarray],
    A_ineq: Optional[np.ndarray],
    P1: Union[np.ndarray, sparse.spmatrix],
    P2: Union[np.ndarray, sparse.spmatrix],
) -> tuple[
    tm.StandardizedMatrix,
    np.ndarray,
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Any,
    Any,
]:
    """
    Standardize the data matrix ``X`` and adjust the bounds and penalties to
    match the standardized data matrix, so that standardizing does not affect
    estimates.

    This is only done for computational reasons and does not affect final
    estimates or alter the input data. Columns are always scaled to have unit
    standard deviation.

    Bounds, inequality constraints and regularization coefficients are modified
    appropriately so that the estimates remain unchanged compared to an
    unstandardized problem.

    Parameters
    ----------
    X : MatrixBase
    sample_weight : numpy.ndarray
    center_predictors : bool
        If ``True``, adjust the data matrix so that columns have mean zero.
    estimate_as_if_scaled_model : bool
        If ``True``, estimates returned equal those from a model where
        predictors have been standardized to have unit standard deviation, with
        penalty unchanged. Note that, internally, for purely computational
        reasons, we always scale predictors; whether estimates match a scaled
        model depends on whether we modify the penalty. If ``False``, penalties
        are rescaled to match the original scale, canceling out the effect of
        rescaling X.
    lower_bounds
    upper_bounds
    A_ineq
    P1
    P2
    """
    X, col_means, col_stds = X.standardize(sample_weight, center_predictors, True)

    if col_stds is not None:
        inv_col_stds = _one_over_var_inf_to_val(col_stds, 1.0)
        # We copy the bounds when multiplying here so the we avoid
        # side effects.
        if lower_bounds is not None:
            lower_bounds = lower_bounds / inv_col_stds
        if upper_bounds is not None:
            upper_bounds = upper_bounds / inv_col_stds
        if A_ineq is not None:
            A_ineq = A_ineq * inv_col_stds

    if not estimate_as_if_scaled_model and col_stds is not None:
        P1 *= inv_col_stds
        if sparse.issparse(P2):
            inv_col_stds_mat = sparse.diags(inv_col_stds)
            P2 = inv_col_stds_mat @ P2 @ inv_col_stds_mat
        elif P2.ndim == 1:
            P2 *= inv_col_stds**2
        else:
            P2 = (inv_col_stds[:, None] * P2) * inv_col_stds[None, :]

    return X, col_means, col_stds, lower_bounds, upper_bounds, A_ineq, P1, P2


def standardize_warm_start(  # noda D
    coef: np.ndarray, col_means: np.ndarray, col_stds: Optional[np.ndarray]
) -> None:
    if col_stds is None:
        coef[0] += np.squeeze(col_means).dot(coef[1:])
    else:
        coef[1:] *= col_stds
        coef[0] += np.squeeze(col_means * _one_over_var_inf_to_val(col_stds, 1)).dot(
            coef[1:]
        )


def unstandardize(  # noda D
    col_means: np.ndarray,
    col_stds: Optional[np.ndarray],
    intercept: Union[float, np.ndarray],
    coef: np.ndarray,
) -> tuple[Union[float, np.ndarray], np.ndarray]:
    if col_stds is None:
        intercept -= np.squeeze(np.squeeze(col_means).dot(np.atleast_1d(coef).T))
    else:
        penalty_mult = _one_over_var_inf_to_val(col_stds, 1.0)
        intercept -= np.squeeze(
            np.squeeze(col_means * penalty_mult).dot(np.atleast_1d(coef).T)
        )
        coef *= penalty_mult
    return intercept, coef


def _one_over_var_inf_to_val(arr: np.ndarray, val: float) -> np.ndarray:
    """
    Return 1/arr unless the values are zeros.

    If values are zeros, return val.
    """
    zeros = np.where(np.abs(arr) < 10 * np.sqrt(np.finfo(arr.dtype).eps))
    with np.errstate(divide="ignore"):
        one_over = 1 / arr
    one_over[zeros] = val
    return one_over
