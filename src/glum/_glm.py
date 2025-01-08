"""
Generalized Linear Models with Exponential Dispersion Family

Modified from code submitted as a PR to sklearn:
https://github.com/scikit-learn/scikit-learn/pull/9405

Original attribution from:
https://github.com/scikit-learn/scikit-learn/pull/9405/files
#diff-38e412190dc50455611b75cfcf2d002713dcf6d537a78b9a22cc6b1c164390d1
'''
Author: Christian Lorentzen <lorentzen.ch@googlemail.com>
some parts and tricks stolen from other sklearn files.
'''
"""

# License: BSD 3 clause
import copy
import re
import sys
import warnings
from collections.abc import Iterable, Mapping, Sequence
from time import perf_counter
from typing import Any, NamedTuple, Optional, Union, cast

import numpy as np
import pandas as pd
import scipy.sparse as sps
import scipy.sparse.linalg as splinalg
import sklearn as skl
import tabmat as tm
from formulaic import Formula, FormulaSpec
from formulaic.parser import DefaultFormulaParser
from formulaic.utils.constraints import LinearConstraintParser
from formulaic.utils.context import capture_context
from packaging import version
from scipy import linalg, sparse, stats
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_array
from sklearn.utils.validation import (
    assert_all_finite,
    check_consistent_length,
    check_is_fitted,
    column_or_1d,
)

from ._distribution import (
    BinomialDistribution,
    ExponentialDispersionModel,
    GammaDistribution,
    GeneralizedHyperbolicSecant,
    InverseGaussianDistribution,
    NegativeBinomialDistribution,
    NormalDistribution,
    PoissonDistribution,
    TweedieDistribution,
    guess_intercept,
)
from ._link import CloglogLink, IdentityLink, Link, LogitLink, LogLink, TweedieLink
from ._solvers import (
    IRLSData,
    _cd_solver,
    _irls_solver,
    _lbfgs_solver,
    _least_squares_solver,
    _trust_constr_solver,
)
from ._util import (
    _add_missing_categories,
    _align_df_categories,
    _expand_categorical_penalties,
    _is_contiguous,
    _safe_toarray,
)

if version.parse(skl.__version__).release < (1, 6):
    keyword_finiteness = "force_all_finite"
    validate_data = BaseEstimator._validate_data
else:
    keyword_finiteness = "ensure_all_finite"
    from sklearn.utils.validation import validate_data  # type: ignore

_float_itemsize_to_dtype = {8: np.float64, 4: np.float32, 2: np.float16}

VectorLike = Union[np.ndarray, pd.api.extensions.ExtensionArray, pd.Index, pd.Series]

ArrayLike = Union[
    list,
    tm.MatrixBase,
    tm.StandardizedMatrix,
    pd.DataFrame,
    sparse.spmatrix,
    VectorLike,
]

ShapedArrayLike = Union[
    tm.MatrixBase,
    tm.StandardizedMatrix,
    pd.DataFrame,
    sparse.spmatrix,
    VectorLike,
]


class WaldTestResult(NamedTuple):
    test_statistic: float
    p_value: float
    df: int


def check_array_tabmat_compliant(mat: ArrayLike, drop_first: bool = False, **kwargs):
    to_copy = kwargs.get("copy", False)

    if isinstance(mat, pd.DataFrame):
        raise RuntimeError("DataFrames should have been converted by this point.")

    if isinstance(mat, tm.SplitMatrix):
        kwargs.update({"ensure_min_features": 0})
        new_matrices = [
            check_array_tabmat_compliant(m, drop_first=drop_first, **kwargs)
            for m in mat.matrices
        ]
        new_indices = [elt.copy() for elt in mat.indices] if to_copy else mat.indices
        return tm.SplitMatrix(new_matrices, new_indices)

    if isinstance(mat, tm.CategoricalMatrix):
        if to_copy:
            return copy.copy(mat)
        return mat

    if isinstance(mat, tm.StandardizedMatrix):
        return tm.StandardizedMatrix(
            check_array_tabmat_compliant(mat.mat, drop_first=drop_first, **kwargs),
            check_array(mat.shift, **kwargs),
        )

    original_type = type(mat)
    if isinstance(mat, (tm.DenseMatrix, tm.SparseMatrix)):
        res = check_array(mat.unpack(), **kwargs)
    else:
        res = check_array(mat, **kwargs)

    if res is not mat and original_type in (tm.DenseMatrix, tm.SparseMatrix):
        res = original_type(
            res,
            column_names=mat.column_names,  # type: ignore
            term_names=mat.term_names,  # type: ignore
        )

    return res


def check_X_y_tabmat_compliant(
    X: ArrayLike, y: Union[VectorLike, sparse.spmatrix], **kwargs
) -> tuple[Union[tm.MatrixBase, sparse.spmatrix, np.ndarray], np.ndarray]:
    """
    See the documentation for :func:`sklearn.utils.check_X_y`. This function
    behaves identically for inputs that are not from the Matrix package and
    fixes some parameters, such as ``'force_all_finite'``, to match the needs of
    GLMs.

    Returns
    -------
    X_converted : array-like
        The converted and validated X.
    y_converted : numpy.ndarray
        The converted and validated y.
    """
    if y is None:
        raise ValueError("y cannot be None")

    y = column_or_1d(y, warn=True)
    assert_all_finite(y)
    if y.dtype.kind == "O":
        y = y.astype(np.float64)

    check_consistent_length(X, y)
    X = check_array_tabmat_compliant(X, **kwargs)

    return X, y


def _check_weights(
    sample_weight: Optional[Union[float, VectorLike]],
    n_samples: int,
    dtype,
    force_all_finite: bool = True,
) -> np.ndarray:
    """Check that sample weights are non-negative and have the right shape."""
    if sample_weight is None:
        return np.ones(n_samples, dtype=dtype)
    if np.isscalar(sample_weight):
        if sample_weight <= 0:  # type: ignore
            raise ValueError("Sample weights must be non-negative.")
        return np.full(n_samples, sample_weight, dtype=dtype)

    sample_weight = check_array(
        sample_weight,
        accept_sparse=False,
        ensure_2d=False,
        dtype=[np.float64, np.float32],
        **{keyword_finiteness: force_all_finite},
    )

    if sample_weight.ndim > 1:  # type: ignore
        raise ValueError("Sample weights must be 1D array or scalar.")
    if sample_weight.shape[0] != n_samples:  # type: ignore
        raise ValueError("Sample weights must have the same length as y.")
    if np.any(sample_weight < 0):  # type: ignore
        raise ValueError("Sample weights must be non-negative.")
    if np.sum(sample_weight) == 0:  # type: ignore
        raise ValueError("Sample weights must have at least one positive element.")

    return sample_weight  # type: ignore


def _check_offset(
    offset: Optional[Union[VectorLike, float]], n_rows: int, dtype
) -> Optional[np.ndarray]:
    """
    Unlike weights, if the offset is ``None``, it can stay ``None``, so we only
    need to validate it when it is not.
    """
    if offset is None:
        return None
    if np.isscalar(offset):
        return np.full(n_rows, offset)

    offset = check_array(
        offset,
        accept_sparse=False,
        ensure_2d=False,
        dtype=dtype,
        **{keyword_finiteness: True},
    )

    offset = cast(np.ndarray, offset)

    if offset.ndim > 1:  # type: ignore
        raise ValueError("Offsets must be 1D array or scalar.")
    if offset.shape[0] != n_rows:  # type: ignore
        raise ValueError("Offsets must have the same length as y.")

    return offset


def _parse_formula(
    formula: FormulaSpec, include_intercept: bool = True
) -> tuple[Optional[Formula], Formula]:
    """
    Parse and transform the formula for use in a GeneralizedLinearRegressor.

    The left-hand side and right-hand side of the formula are separated. If an
    intercept is present, it will be removed from the right-hand side, and a
    boolean flag to indicate whether or not an intercept should be added to
    the model will be returned.

    Parameters
    ----------
    formula : formulaic.FormulaSpec
        The formula to parse.
    include_intercept: bool, default True
        Whether to include an intercept column.

    Returns
    -------
    tuple[formulaic.Formula, formulaic.Formula]
        The left-hand side and right-hand sides of the formula.
    """
    if isinstance(formula, str):
        parser = DefaultFormulaParser(include_intercept=include_intercept)
        terms = parser.get_terms(formula)
    elif isinstance(formula, Formula):
        terms = formula
    else:
        raise TypeError("formula must be a string or Formula object.")

    if hasattr(terms, "lhs"):
        lhs_terms = terms.lhs
        if len(lhs_terms) != 1:
            raise ValueError(
                "formula must have exactly one term on the left-hand side."
            )
        rhs_terms = terms.rhs
    else:
        lhs_terms = None
        rhs_terms = terms

    return lhs_terms, rhs_terms


def check_bounds(
    bounds: Optional[Union[float, VectorLike]], n_features: int, dtype
) -> Optional[np.ndarray]:
    """Check that the bounds have the right shape."""
    if bounds is None:
        return None
    if np.isscalar(bounds):
        return np.full(n_features, bounds, dtype=dtype)

    bounds = check_array(
        bounds,
        accept_sparse=False,
        ensure_2d=False,
        dtype=dtype,
        **{keyword_finiteness: False},
    )

    bounds = cast(np.ndarray, bounds)

    if bounds.ndim > 1:  # type: ignore
        raise ValueError("Bounds must be 1D array or scalar.")
    if bounds.shape[0] != n_features:  # type: ignore
        raise ValueError("Bounds must be the same length as X.shape[1].")

    return bounds


def check_inequality_constraints(
    A_ineq: Optional[np.ndarray],
    b_ineq: Optional[np.ndarray],
    n_features: int,
    dtype,
) -> tuple[Union[None, np.ndarray], Union[None, np.ndarray]]:
    """Check that the inequality constraints are well-defined."""
    if A_ineq is None or b_ineq is None:
        return None, None
    else:
        A_ineq = check_array(
            A_ineq,
            accept_sparse=False,
            ensure_2d=True,
            dtype=dtype,
            copy=True,
            **{keyword_finiteness: False},
        )
        b_ineq = check_array(
            b_ineq,
            accept_sparse=False,
            ensure_2d=False,
            dtype=dtype,
            copy=True,
            **{keyword_finiteness: False},
        )
        if A_ineq.shape[1] != n_features:  # type: ignore
            raise ValueError("A_ineq must have same number of columns as X.")
        if A_ineq.shape[0] != b_ineq.shape[0]:  # type: ignore
            raise ValueError("A_ineq and b_ineq must have same number of rows.")
        if b_ineq.ndim > 1:  # type: ignore
            raise ValueError("b_ineq must be 1D array.")
    return A_ineq, b_ineq


def _standardize(
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


def _unstandardize(
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


def _standardize_warm_start(
    coef: np.ndarray, col_means: np.ndarray, col_stds: Optional[np.ndarray]
) -> None:
    if col_stds is None:
        coef[0] += np.squeeze(col_means).dot(coef[1:])
    else:
        coef[1:] *= col_stds
        coef[0] += np.squeeze(col_means * _one_over_var_inf_to_val(col_stds, 1)).dot(
            coef[1:]
        )


def get_family(
    family: Union[str, ExponentialDispersionModel],
) -> ExponentialDispersionModel:
    if isinstance(family, ExponentialDispersionModel):
        return family

    name_to_dist = {
        "binomial": BinomialDistribution(),
        "gamma": GammaDistribution(),
        "gaussian": NormalDistribution(),
        "inverse.gaussian": InverseGaussianDistribution(),
        "normal": NormalDistribution(),
        "poisson": PoissonDistribution(),
        "tweedie": TweedieDistribution(1.5),
        "negative.binomial": NegativeBinomialDistribution(1.0),
    }

    if family in name_to_dist:
        return name_to_dist[family]

    custom_tweedie = re.search(r"tweedie\s?\((.+)\)", family)

    if custom_tweedie:
        return TweedieDistribution(float(custom_tweedie.group(1)))

    custom_negative_binomial = re.search(r"negative.binomial\s?\((.+)\)", family)

    if custom_negative_binomial:
        return NegativeBinomialDistribution(float(custom_negative_binomial.group(1)))

    raise ValueError(
        "The family must be an instance of class ExponentialDispersionModel or an "
        f"element of {sorted(name_to_dist.keys())}; got (family={family})."
    )


def get_link(link: Union[str, Link], family: ExponentialDispersionModel) -> Link:
    """
    For the Tweedie distribution, this code follows actuarial best practices
    regarding link functions. Note that these links are sometimes not canonical:
        - identity for normal (``p = 0``);
        - no convention for ``p < 0``, so let's leave it as identity;
        - log otherwise.
    """
    if isinstance(link, Link):
        return link

    if (link is None) or (link == "auto"):
        if tweedie_representation := family.to_tweedie(safe=False):
            if tweedie_representation.power <= 0:
                return IdentityLink()
            return LogLink()
        if isinstance(family, GeneralizedHyperbolicSecant):
            return IdentityLink()
        if isinstance(family, BinomialDistribution):
            return LogitLink()
        if isinstance(family, NegativeBinomialDistribution):
            return LogLink()
        raise ValueError(
            "No default link known for the specified distribution family. "
            "Please set link manually, i.e. not to 'auto'. "
            f"Got (link='auto', family={family.__class__.__name__})."
        )

    mapping = {
        "cloglog": CloglogLink(),
        "identity": IdentityLink(),
        "log": LogLink(),
        "logit": LogitLink(),
        "tweedie": TweedieLink(1.5),
    }

    if link in mapping:
        return mapping[link]
    if custom_tweedie := re.search(r"tweedie\s?\((.+)\)", link):
        return TweedieLink(float(custom_tweedie.group(1)))

    raise ValueError(
        "The link must be an instance of class Link or an element of "
        "['auto', 'identity', 'log', 'logit', 'cloglog', 'tweedie']; "
        f"got (link={link})."
    )


def setup_p1(
    P1: Optional[Union[str, np.ndarray]],
    X: Union[tm.MatrixBase, tm.StandardizedMatrix],
    dtype,
    alpha: float,
    l1_ratio: float,
) -> np.ndarray:
    if not isinstance(X, (tm.MatrixBase, tm.StandardizedMatrix)):
        raise TypeError

    n_features = X.shape[1]

    if isinstance(P1, str):
        if P1 != "identity":
            raise ValueError(f"P1 must be either 'identity' or an array; got {P1}.")
        P1 = np.ones(n_features, dtype=dtype)
    elif P1 is None:
        P1 = np.ones(n_features, dtype=dtype)
    else:
        P1 = np.atleast_1d(P1)
        try:
            P1 = P1.astype(dtype, casting="safe", copy=False)  # type: ignore
        except TypeError as e:
            raise TypeError(
                "The given P1 cannot be converted to a numeric array; "
                f"got (P1.dtype={P1.dtype})."  # type: ignore
            ) from e
        if (P1.ndim != 1) or (P1.shape[0] != n_features):  # type: ignore
            raise ValueError(
                "P1 must be either 'identity' or a 1d array with the length of "
                "X.shape[1] (either before or after categorical expansion); "
                f"got (P1.shape[0]={P1.shape[0]})."  # type: ignore
            )

    # P1 and P2 are now for sure copies
    P1 = alpha * l1_ratio * P1  # type: ignore
    return cast(np.ndarray, P1).astype(dtype)


def setup_p2(
    P2: Optional[Union[str, np.ndarray, sparse.spmatrix]],
    X: Union[tm.MatrixBase, tm.StandardizedMatrix],
    stype,
    dtype,
    alpha: float,
    l1_ratio: float,
) -> Union[np.ndarray, sparse.spmatrix]:
    if not isinstance(X, (tm.MatrixBase, tm.StandardizedMatrix)):
        raise TypeError

    n_features = X.shape[1]

    def _setup_sparse_p2(P2):
        return (sparse.dia_matrix((P2, 0), shape=(n_features, n_features))).tocsc()

    if isinstance(P2, str):
        if P2 != "identity":
            raise ValueError(f"P2 must be either 'identity' or an array. Got {P2}.")
        if sparse.issparse(X):  # if X is sparse, make P2 sparse, too
            P2 = _setup_sparse_p2(np.ones(n_features, dtype=dtype))
        else:
            P2 = np.ones(n_features, dtype=dtype)
    elif P2 is None:
        if sparse.issparse(X):  # if X is sparse, make P2 sparse, too
            P2 = _setup_sparse_p2(np.ones(n_features, dtype=dtype))
        else:
            P2 = np.ones(n_features, dtype=dtype)
    else:
        P2 = check_array(
            P2, copy=True, accept_sparse=stype, dtype=dtype, ensure_2d=False
        )
        P2 = cast(np.ndarray, P2)
        if P2.ndim == 1:
            P2 = np.asarray(P2)
            if P2.shape[0] != n_features:
                raise ValueError(
                    "P2 should be a 1d array of shape X.shape[1] (either before or "
                    "after categorical expansion); "
                    f"got (P2.shape={P2.shape})."
                )
            if sparse.issparse(X):
                P2 = _setup_sparse_p2(P2)
        elif P2.ndim == 2 and P2.shape[0] == P2.shape[1] and P2.shape[0] == n_features:
            if sparse.issparse(X):
                P2 = sparse.csc_matrix(P2)
        else:
            raise ValueError(
                "P2 must be either None or an array of shape (n_features, n_features) "
                f"with n_features=X.shape[1]; got (P2.shape={P2.shape}); "
                f"needed ({n_features}, {n_features})."
            )

    # P1 and P2 are now for sure copies
    P2 = alpha * (1 - l1_ratio) * P2
    # one only ever needs the symmetrized L2 penalty matrix 1/2 (P2 + P2')
    # reason: w' P2 w = (w' P2 w)', i.e. it is symmetric
    if P2.ndim == 2:
        if sparse.issparse(P2):
            if sparse.isspmatrix_csc(P2):
                P2 = 0.5 * (P2 + P2.transpose()).tocsc()
            else:
                P2 = 0.5 * (P2 + P2.transpose()).tocsr()
        else:
            P2 = 0.5 * (P2 + P2.T)
    return P2


def is_pos_semidef(p: Union[sparse.spmatrix, np.ndarray]) -> Union[bool, np.bool_]:
    """
    Checks for positive semidefiniteness of ``p`` if ``p`` is a matrix, or
    ``diag(p)`` if a vector.

    ``np.linalg.cholesky(P2)`` 'only' asserts positive definiteness; due to
    numerical precision, we allow eigenvalues to be a tiny bit negative.
    """
    # 1d case
    if p.ndim == 1 or p.shape[0] == 1:
        any_negative = (p < 0).max() if sparse.isspmatrix(p) else (p < 0).any()
        return not any_negative

    # 2d case
    # About -6e-7 for 32-bit, -1e-15 for 64-bit
    epsneg = -10 * np.finfo(np.result_type(float, p.dtype)).epsneg

    if sparse.issparse(p):
        # Computing eigenvalues for sparse matrices is inefficient. If the matrix is
        # not huge, convert to dense. Otherwise, calculate 10% of its eigenvalues.
        p = cast(sparse.spmatrix, p)
        if p.shape[0] < 2000:
            eigenvalues = linalg.eigvalsh(p.toarray())
        else:
            n_evals_to_compuate = p.shape[0] // 10 + 1
            sigma = -1000 * epsneg  # start searching near this value
            which = "SA"  # find smallest algebraic eigenvalues first
            eigenvalues = splinalg.eigsh(
                p,
                k=n_evals_to_compuate,
                sigma=sigma,
                which=which,
                return_eigenvectors=False,
            )
    else:  # dense
        eigenvalues = linalg.eigvalsh(p)

    return np.all(eigenvalues >= epsneg)


def _group_sum(groups: np.ndarray, data: tm.MatrixBase):
    """Sum over groups."""
    ngroups = len(np.unique(groups))
    out = np.empty((ngroups, data.shape[1]))
    eye_n = sps.eye(ngroups, format="csc")[:, groups]
    for i in range(data.shape[1]):
        out[:, i] = _safe_toarray(eye_n @ data.getcol(i).unpack()).ravel()
    return out


# TODO: abc
class GeneralizedLinearRegressorBase(RegressorMixin, BaseEstimator):
    """
    Base class for :class:`GeneralizedLinearRegressor` and
    :class:`GeneralizedLinearRegressorCV`.
    """

    def __init__(
        self,
        *,
        l1_ratio: float = 0,
        P1: Optional[Union[str, np.ndarray]] = "identity",
        P2: Optional[Union[str, np.ndarray, sparse.spmatrix]] = "identity",
        fit_intercept=True,
        family: Union[str, ExponentialDispersionModel] = "normal",
        link: Union[str, Link] = "auto",
        solver: str = "auto",
        max_iter=100,
        max_inner_iter=100000,
        gradient_tol: Optional[float] = None,
        step_size_tol: Optional[float] = None,
        hessian_approx: float = 0.0,
        warm_start=False,
        alpha_search: bool = False,
        n_alphas: int = 100,
        min_alpha_ratio: Optional[float] = None,
        min_alpha: Optional[float] = None,
        start_params: Optional[np.ndarray] = None,
        selection="cyclic",
        random_state=None,
        copy_X: Optional[bool] = None,
        check_input=True,
        verbose=0,
        scale_predictors: bool = False,
        lower_bounds: Optional[np.ndarray] = None,
        upper_bounds: Optional[np.ndarray] = None,
        A_ineq: Optional[np.ndarray] = None,
        b_ineq: Optional[np.ndarray] = None,
        force_all_finite: bool = True,
        drop_first: bool = False,
        robust: bool = True,
        expected_information: bool = False,
        formula: Optional[FormulaSpec] = None,
        interaction_separator: str = ":",
        categorical_format: str = "{name}[{category}]",
        cat_missing_method: str = "fail",
        cat_missing_name: str = "(MISSING)",
    ):
        self.l1_ratio = l1_ratio
        self.P1 = P1
        self.P2 = P2
        self.fit_intercept = fit_intercept
        self.family = family
        self.link = link
        self.solver = solver
        self.max_iter = max_iter
        self.max_inner_iter = max_inner_iter
        self.gradient_tol = gradient_tol
        self.step_size_tol = step_size_tol
        self.hessian_approx = hessian_approx
        self.warm_start = warm_start
        self.alpha_search = alpha_search
        self.n_alphas = n_alphas
        self.min_alpha_ratio = min_alpha_ratio
        self.min_alpha = min_alpha
        self.start_params = start_params
        self.selection = selection
        self.random_state = random_state
        self.copy_X = copy_X
        self.check_input = check_input
        self.verbose = verbose
        self.scale_predictors = scale_predictors
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.A_ineq = A_ineq
        self.b_ineq = b_ineq
        self.force_all_finite = force_all_finite
        self.drop_first = drop_first
        self.robust = robust
        self.expected_information = expected_information
        self.formula = formula
        self.interaction_separator = interaction_separator
        self.categorical_format = categorical_format
        self.cat_missing_method = cat_missing_method
        self.cat_missing_name = cat_missing_name

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = True
        return tags

    @property
    def family_instance(self) -> ExponentialDispersionModel:
        """Return an :class:`~glum._distribution.ExponentialDispersionModel`."""
        if hasattr(self, "_family_instance"):
            return self._family_instance
        else:
            return get_family(self.family)

    @property
    def link_instance(self) -> Link:
        """Return a :class:`~glum._link.Link`."""
        if hasattr(self, "_link_instance"):
            return self._link_instance
        else:
            return get_link(self.link, self.family_instance)

    def _get_start_coef(
        self,
        X: Union[tm.MatrixBase, tm.StandardizedMatrix],
        y: np.ndarray,
        sample_weight: np.ndarray,
        offset: Optional[np.ndarray],
        col_means: np.ndarray,
        col_stds: Optional[np.ndarray],
        dtype,
    ) -> np.ndarray:
        if self.warm_start and hasattr(self, "coef_"):
            coef = self.coef_  # type: ignore
            intercept = self.intercept_  # type: ignore
            if self.fit_intercept:
                coef = np.concatenate((np.array([intercept]), coef))
            if self._center_predictors:
                _standardize_warm_start(coef, col_means, col_stds)  # type: ignore

        elif self.start_params is None:
            if self.fit_intercept:
                coef = np.zeros(
                    X.shape[1] + 1, dtype=_float_itemsize_to_dtype[X.dtype.itemsize]
                )
                coef[0] = guess_intercept(
                    y, sample_weight, self._link_instance, self._family_instance, offset
                )
            else:
                coef = np.zeros(
                    X.shape[1], dtype=_float_itemsize_to_dtype[X.dtype.itemsize]
                )

        else:  # assign given array as start values
            coef = check_array(
                self.start_params,
                accept_sparse=False,
                ensure_2d=False,
                dtype=dtype,
                copy=True,
                **{keyword_finiteness: True},
            )

            if coef.shape != (len(col_means) + self.fit_intercept,):
                raise ValueError(
                    "Start values for parameters must have the right length "
                    f"and dimension; got {coef.shape}, needed "
                    f"({len(col_means) + self.fit_intercept},)."
                )

            if self._center_predictors:
                _standardize_warm_start(coef, col_means, col_stds)  # type: ignore

        # If starting values are outside the specified bounds (if set),
        # bring the starting value exactly at the bound.
        idx = int(self.fit_intercept)
        if self.lower_bounds is not None:
            if np.any(coef[idx:] < self.lower_bounds):
                warnings.warn(
                    "lower_bounds above starting value. Setting the starting values "
                    "to max(start_params, lower_bounds)."
                )
                coef[idx:] = np.maximum(coef[idx:], self.lower_bounds)
        if self.upper_bounds is not None:
            if np.any(coef[idx:] > self.upper_bounds):
                warnings.warn(
                    "upper_bounds below starting value. Setting the starting values "
                    "to min(start_params, upper_bounds)."
                )
                coef[idx:] = np.minimum(coef[idx:], self.upper_bounds)

        return coef

    def _convert_from_pandas(
        self, df: pd.DataFrame, context: Optional[Mapping[str, Any]] = None
    ) -> tm.MatrixBase:
        """Convert a pandas data frame to a tabmat matrix."""
        if hasattr(self, "X_model_spec_"):
            return self.X_model_spec_.get_model_matrix(df, context=context)

        cat_missing_method_after_alignment = getattr(self, "cat_missing_method", "fail")

        if hasattr(self, "feature_dtypes_"):
            df = _align_df_categories(
                df,
                self.feature_dtypes_,
                getattr(self, "has_missing_category_", {}),
                cat_missing_method_after_alignment,
            )
            if cat_missing_method_after_alignment == "convert":
                df = _add_missing_categories(
                    df=df,
                    dtypes=self.feature_dtypes_,
                    feature_names=self.feature_names_,
                    cat_missing_name=self.cat_missing_name,
                    categorical_format=self.categorical_format,
                )
                # there should be no missing categories after this
                cat_missing_method_after_alignment = "fail"

        X = tm.from_pandas(
            df,
            drop_first=self.drop_first,
            categorical_format=getattr(  # convention prior to v3
                self, "categorical_format", "{name}__{category}"
            ),
            cat_missing_method=cat_missing_method_after_alignment,
        )

        return X

    def _set_up_for_fit(self, y: np.ndarray) -> None:
        #######################################################################
        # 1. input validation                                                 #
        #######################################################################
        # self.family and self.link are user-provided inputs and may be strings or
        #  ExponentialDispersonModel/Link objects
        # self.family_instance_ and self.link_instance_ are cleaned by 'fit' to be
        # ExponentialDispersionModel and Link arguments
        self._family_instance: ExponentialDispersionModel = get_family(self.family)
        # Guarantee that self._link_instance is set to an instance of class Link
        self._link_instance: Link = get_link(self.link, self._family_instance)

        # when fit_intercept is False, we can't center because that would
        # substantially change estimates
        self._center_predictors: bool = self.fit_intercept

        # require number of observations in the training data for later
        # computation of information criteria
        self._num_obs: int = y.shape[0]

        if self.solver == "auto":
            if (self.A_ineq is not None) and (self.b_ineq is not None):
                self._solver = "trust-constr"
            elif (self.lower_bounds is None) and (self.upper_bounds is None):
                if np.all(np.asarray(self.l1_ratio) == 0):
                    self._solver = "irls-ls"
                elif (
                    hasattr(self, "alpha") and self.alpha == 0 and not self.alpha_search
                ):
                    self._solver = "irls-ls"
                else:
                    self._solver = "irls-cd"
            else:
                self._solver = "irls-cd"
        else:
            self._solver = self.solver

        if self.gradient_tol is None:
            if self._solver == "trust-constr":
                self._gradient_tol = 1e-8
            else:
                self._gradient_tol = 1e-4
        else:
            self._gradient_tol = self.gradient_tol

        # 1.4 additional validations ##########################################
        if self.check_input:
            if not np.all(self._family_instance.in_y_range(y)):
                raise ValueError(
                    "Some value(s) of y are out of the valid range for family"
                    f"{self._family_instance.__class__.__name__}."
                )

    def _get_alpha_path(
        self,
        P1_no_alpha: np.ndarray,
        X,
        y: np.ndarray,
        w: np.ndarray,
        offset: np.ndarray = None,
    ) -> np.ndarray:
        """
        Get the regularization path.

        If some features have L1 regularization, the maximum alpha is the lowest
        alpha such that no l1-regularized coefficients are nonzero.

        If all features do not have L1 regularization, use the
        :class:`sklearn.linear_model.RidgeCV` default path ``[10, 1, 0.1]`` or
        whatever is specified by the input parameters ``min_alpha_ratio`` and
        ``n_alphas``.

        ``min_alpha_ratio`` governs the length of the path, with ``1e-6`` as the
        default. Smaller values will lead to a longer path.
        """

        def _make_grid(max_alpha: float) -> np.ndarray:
            if self.min_alpha is None:
                if self.min_alpha_ratio is None:
                    min_alpha = max_alpha * 1e-6
                else:
                    min_alpha = max_alpha * self.min_alpha_ratio
            else:
                if self.min_alpha >= max_alpha:
                    raise ValueError(
                        "Current value of min_alpha would generate all zeros. "
                        "Consider reducing this value."
                    )
                if self.min_alpha_ratio is not None:
                    warnings.warn("`min_alpha` is set. Ignoring `min_alpha_ratio`.")
                min_alpha = self.min_alpha
            return np.logspace(
                np.log(max_alpha),
                np.log(min_alpha),
                self.n_alphas,
                base=np.e,
                dtype=X.dtype,
            )

        if np.all(P1_no_alpha == 0):
            alpha_max = 10
            return _make_grid(alpha_max)

        if self.fit_intercept:
            intercept_offset = 1
            coef = np.zeros(X.shape[1] + 1, dtype=X.dtype)
            coef[0] = guess_intercept(
                y=y,
                sample_weight=w,
                link=self._link_instance,
                distribution=self._family_instance,
            )
        else:
            intercept_offset = 0
            coef = np.zeros(X.shape[1], dtype=X.dtype)

        _, dev_der = self._family_instance._mu_deviance_derivative(
            coef=coef,
            X=X,
            y=y,
            sample_weight=w,
            link=self._link_instance,
            offset=offset,
        )

        l1_regularized_mask = P1_no_alpha > 0
        alpha_max = np.max(
            np.abs(
                -0.5
                * dev_der[intercept_offset:][l1_regularized_mask]
                / P1_no_alpha[l1_regularized_mask]
            )
        )
        return _make_grid(alpha_max)

    def _solve(
        self,
        X: Union[tm.MatrixBase, tm.StandardizedMatrix],
        y: np.ndarray,
        sample_weight: np.ndarray,
        P2,
        P1: np.ndarray,
        coef: np.ndarray,
        offset: Optional[np.ndarray],
        lower_bounds: Optional[np.ndarray],
        upper_bounds: Optional[np.ndarray],
        A_ineq: Optional[np.ndarray],
        b_ineq: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Must be run after running :func:`_set_up_for_fit`. Sets
        ``self.coef_`` and ``self.intercept_``.
        """
        fixed_inner_tol = None
        if (
            isinstance(self._family_instance, NormalDistribution)
            and isinstance(self._link_instance, IdentityLink)
            and "irls" in self._solver
        ):
            # IRLS-CD and IRLS-LS should converge in one iteration for any
            # normal distribution problem with identity link.
            fixed_inner_tol = (self._gradient_tol, self.step_size_tol)
            max_iter = 1
        else:
            max_iter = self.max_iter

        # 4.1 IRLS ############################################################
        if "irls" in self._solver:
            # Note: we already set P1 = l1*P1, see above
            # Note: we already set P2 = l2*P2, see above
            # Note: we already symmetrized P2 = 1/2 (P2 + P2')
            irls_data = IRLSData(
                X=X,
                y=y,
                sample_weight=sample_weight,
                P1=P1,
                P2=P2,
                fit_intercept=self.fit_intercept,
                family=self._family_instance,
                link=self._link_instance,
                max_iter=max_iter,
                max_inner_iter=getattr(self, "max_inner_iter", 100_000),
                gradient_tol=self._gradient_tol,
                step_size_tol=self.step_size_tol,
                fixed_inner_tol=fixed_inner_tol,
                hessian_approx=self.hessian_approx,
                selection=self.selection,
                random_state=self.random_state,
                offset=offset,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
                verbose=self.verbose > 0,
            )
            if self._solver == "irls-ls":
                coef, self.n_iter_, self._n_cycles, self.diagnostics_ = _irls_solver(
                    _least_squares_solver, coef, irls_data
                )
            # 4.2 coordinate descent ##############################################
            elif self._solver == "irls-cd":
                coef, self.n_iter_, self._n_cycles, self.diagnostics_ = _irls_solver(
                    _cd_solver, coef, irls_data
                )
        # 4.3 L-BFGS ##########################################################
        elif self._solver == "lbfgs":
            coef, self.n_iter_, self._n_cycles, self.diagnostics_ = _lbfgs_solver(
                coef=coef,
                X=X,
                y=y,
                sample_weight=sample_weight,
                P2=P2,
                verbose=self.verbose,
                family=self._family_instance,
                link=self._link_instance,
                max_iter=max_iter,
                # TODO: support step_size_tol?
                tol=self._gradient_tol,  # type: ignore
                offset=offset,
            )
        # 4.4 trust-constr ####################################################
        elif self._solver == "trust-constr":
            (
                coef,
                self.n_iter_,
                self._n_cycles,
                self.diagnostics_,
            ) = _trust_constr_solver(
                coef=coef,
                X=X,
                y=y,
                sample_weight=sample_weight,
                P2=P2,
                fit_intercept=self.fit_intercept,
                verbose=self.verbose > 0,
                family=self._family_instance,
                link=self._link_instance,
                max_iter=max_iter,
                gtol=self._gradient_tol,
                offset=offset,
                A_ineq=A_ineq,
                b_ineq=b_ineq,
            )
        return coef

    def _solve_regularization_path(
        self,
        X: Union[tm.MatrixBase, tm.StandardizedMatrix],
        y: np.ndarray,
        sample_weight: np.ndarray,
        alphas: np.ndarray,
        P2_no_alpha,
        P1_no_alpha: np.ndarray,
        coef: np.ndarray,
        offset: Optional[np.ndarray],
        lower_bounds: Optional[np.ndarray],
        upper_bounds: Optional[np.ndarray],
        A_ineq: Optional[np.ndarray],
        b_ineq: Optional[np.ndarray],
    ) -> np.ndarray:
        self.coef_path_ = np.empty((len(alphas), len(coef)), dtype=X.dtype)

        for k, alpha in enumerate(alphas):
            P1 = P1_no_alpha * alpha
            P2 = P2_no_alpha * alpha

            tic = perf_counter()
            coef = self._solve(
                X=X,
                y=y,
                sample_weight=sample_weight,
                P2=P2,
                P1=P1,
                coef=coef,
                offset=offset,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
                A_ineq=A_ineq,
                b_ineq=b_ineq,
            )
            toc = perf_counter()

            if self.verbose > 0:
                print(
                    f"alpha={alpha:.3e}, time={toc - tic:.2f}s, n_iter={self.n_iter_}"
                )

            self.coef_path_[k, :] = coef

        return self.coef_path_

    def report_diagnostics(
        self, *, full_report: bool = False, custom_columns: Optional[Iterable] = None
    ) -> None:
        """Print diagnostics to ``stdout``.

        Parameters
        ----------
        full_report : bool, optional (default=False)
            Print all available information. When ``False`` and
            ``custom_columns`` is ``None``, a restricted set of columns is
            printed out.

        custom_columns : iterable, optional (default=None)
            Print only the specified columns.
        """
        diagnostics = self.get_formatted_diagnostics(
            full_report=full_report, custom_columns=custom_columns
        )
        if isinstance(diagnostics, str):
            print(diagnostics)
            return

        import pandas as pd

        print("Diagnostics:")
        with pd.option_context("display.max_rows", None, "display.max_columns", None):
            print(diagnostics)

    def get_formatted_diagnostics(
        self, *, full_report: bool = False, custom_columns: Optional[Iterable] = None
    ) -> Union[str, pd.DataFrame]:
        """Get formatted diagnostics which can be printed with report_diagnostics.

        Parameters
        ----------
        full_report : bool, optional (default=False)
            Print all available information. When ``False`` and
            ``custom_columns`` is ``None``, a restricted set of columns is
            printed out.

        custom_columns : iterable, optional (default=None)
            Print only the specified columns.
        """
        if not hasattr(self, "diagnostics_"):
            to_print = "Model has not been fit, so no diagnostics exist."
            return to_print
        if self.diagnostics_ is None:
            to_print = "solver does not report diagnostics"
            return to_print

        import pandas as pd

        df = pd.DataFrame(data=self.diagnostics_).set_index("n_iter", drop=True)
        if self.fit_intercept:
            df["intercept"] = df["first_coef"]
        else:
            df["intercept"] = np.nan

        if custom_columns is not None:
            keep_cols = custom_columns
        elif full_report:
            keep_cols = df.columns
        else:
            keep_cols = [
                "convergence",
                "n_cycles",
                "iteration_runtime",
                "intercept",
            ]
        return df[keep_cols]

    def _find_alpha_index(self, alpha):
        if alpha is None:
            return None
        if not self.alpha_search:
            raise ValueError
        # `np.isclose` because comparing floats is difficult
        isclose = np.isclose(self._alphas, alpha)
        if np.sum(isclose) == 1:
            return np.argmax(isclose)  # cf. stackoverflow.com/a/61117770
        raise IndexError(
            f"Could not determine a unique index for alpha {alpha}. Available values: "
            f"{self._alphas}. Consider specifying the index directly via 'alpha_index'."
        )

    def linear_predictor(
        self,
        X: ArrayLike,
        offset: Optional[ArrayLike] = None,
        *,
        alpha_index: Optional[Union[int, Sequence[int]]] = None,
        alpha: Optional[Union[float, Sequence[float]]] = None,
        context: Optional[Union[int, Mapping[str, Any]]] = None,
    ):
        """Compute the linear predictor, ``X * coef_ + intercept_``.

        If ``alpha_search`` is ``True``, but ``alpha_index`` and ``alpha`` are
        both ``None``, we use the last alpha value ``self._alphas[-1]``.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Observations. ``X`` may be a pandas data frame with categorical
            types. If ``X`` was also a data frame with categorical types during
            fitting and a category wasn't observed at that point, the
            corresponding prediction will be ``numpy.nan``.

        offset : array-like, shape (n_samples,), optional (default=None)

        alpha_index : int or list[int], optional (default=None)
            Sets the index of the alpha(s) to use in case ``alpha_search`` is
            ``True``. Incompatible with ``alpha`` (see below).

        alpha : float or list[float], optional (default=None)
            Sets the alpha(s) to use in case ``alpha_search`` is ``True``.
            Incompatible with ``alpha_index`` (see above).

        context : Optional[Union[int, Mapping[str, Any]]], default=None
            The context to add to the evaluation context of the formula with,
            e.g., custom transforms. If an integer, the context is taken from
            the stack frame of the caller at the given depth. Otherwise, a
            mapping from variable names to values is expected. By default,
            no context is added. Set ``context=0`` to make the calling scope
            available.

        Returns
        -------
        array, shape (n_samples, n_alphas)
            The linear predictor.
        """
        check_is_fitted(self, "coef_")

        if (alpha is not None) and (alpha_index is not None):
            raise ValueError("Please specify only one of {alpha_index, alpha}.")
        elif np.isscalar(alpha):  # `None` doesn't qualify
            alpha_index = self._find_alpha_index(alpha)
        elif alpha is not None:
            alpha_index = [self._find_alpha_index(a) for a in alpha]  # type: ignore

        if isinstance(X, pd.DataFrame):
            captured_context = capture_context(
                context + 1 if isinstance(context, int) else context
            )
            X = self._convert_from_pandas(X, context=captured_context)

        X = check_array_tabmat_compliant(
            X,
            accept_sparse=["csr", "csc", "coo"],
            dtype="numeric",
            copy=True,
            ensure_2d=True,
            allow_nd=False,
            drop_first=getattr(self, "drop_first", False),
        )

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but {self.__class__.__name__} "
                f"is expecting {self.n_features_in_} features as input."
            )

        if alpha_index is None:
            xb = X @ self.coef_ + self.intercept_
            if offset is not None:
                xb += offset
        elif np.isscalar(alpha_index):  # `None` doesn't qualify
            xb = X @ self.coef_path_[alpha_index] + self.intercept_path_[alpha_index]  # type: ignore
            if offset is not None:
                xb += offset
        else:  # hopefully a list or some such
            xb = np.stack(
                [
                    X @ self.coef_path_[idx] + self.intercept_path_[idx]
                    for idx in alpha_index  # type: ignore
                ],
                axis=1,
            )
            if offset is not None:
                xb += np.asanyarray(offset)[:, np.newaxis]

        return xb

    def predict(
        self,
        X: ShapedArrayLike,
        sample_weight: Optional[ArrayLike] = None,
        offset: Optional[ArrayLike] = None,
        *,
        alpha_index: Optional[Union[int, Sequence[int]]] = None,
        alpha: Optional[Union[float, Sequence[float]]] = None,
        context: Optional[Union[int, Mapping[str, Any]]] = None,
    ):
        """Predict using GLM with feature matrix ``X``.

        If ``alpha_search`` is ``True``, but ``alpha_index`` and ``alpha`` are
        both ``None``, we use the last alpha value ``self._alphas[-1]``.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Observations. ``X`` may be a pandas data frame with categorical
            types. If ``X`` was also a data frame with categorical types during
            fitting and a category wasn't observed at that point, the
            corresponding prediction will be ``numpy.nan``.

        sample_weight : array-like, shape (n_samples,), optional (default=None)
            Sample weights to multiply predictions by.

        offset : array-like, shape (n_samples,), optional (default=None)

        alpha_index : int or list[int], optional (default=None)
            Sets the index of the alpha(s) to use in case ``alpha_search`` is
            ``True``. Incompatible with ``alpha`` (see below).

        alpha : float or list[float], optional (default=None)
            Sets the alpha(s) to use in case ``alpha_search`` is ``True``.
            Incompatible with ``alpha_index`` (see above).

        context : Optional[Union[int, Mapping[str, Any]]], default=None
            The context to add to the evaluation context of the formula with,
            e.g., custom transforms. If an integer, the context is taken from
            the stack frame of the caller at the given depth. Otherwise, a
            mapping from variable names to values is expected. By default,
            no context is added. Set ``context=0`` to make the calling scope
            available.

        Returns
        -------
        array, shape (n_samples, n_alphas)
            Predicted values times ``sample_weight``.
        """
        if isinstance(X, pd.DataFrame):
            captured_context = capture_context(
                context + 1 if isinstance(context, int) else context
            )
            X = self._convert_from_pandas(X, context=captured_context)

        eta = self.linear_predictor(
            X, offset=offset, alpha_index=alpha_index, alpha=alpha
        )
        mu = get_link(self.link, get_family(self.family)).inverse(eta)

        if sample_weight is None:
            return mu

        sample_weight = _check_weights(sample_weight, X.shape[0], X.dtype)
        return mu * sample_weight

    def coef_table(
        self,
        X=None,
        y=None,
        sample_weight=None,
        offset=None,
        *,
        confidence_level=0.95,
        mu=None,
        dispersion=None,
        robust=None,
        clusters: np.ndarray = None,
        expected_information=None,
        context: Optional[Union[int, Mapping[str, Any]]] = None,
    ):
        """Get a table of of the regression coefficients.

        Includes coefficient estimates, standard errors, t-values, p-values
        and confidence intervals.

        Parameters
        ----------
        confidence_level : float, optional, default=0.95
            The confidence level for the confidence intervals.
        X : {array-like, sparse matrix}, shape (n_samples, n_features), optional
            Training data. Can be omitted if a covariance matrix has already
            been computed or if standard errors, etc. are not desired.
        y : array-like, shape (n_samples,), optional
            Target values. Can be omitted if a covariance matrix has already
            been computed.
        mu : array-like, optional, default=None
            Array with predictions. Estimated if absent.
        offset : array-like, optional, default=None
            Array with additive offsets.
        sample_weight : array-like, shape (n_samples,), optional, default=None
            Individual weights for each sample.
        dispersion : float, optional, default=None
            The dispersion parameter. Estimated if absent.
        robust : boolean, optional, default=None
            Whether to compute robust standard errors instead of normal ones.
            If not specified, the model's ``robust`` attribute is used.
        clusters : array-like, optional, default=None
            Array with cluster membership. Clustered standard errors are
            computed if clusters is not None.
        expected_information : boolean, optional, default=None
            Whether to use the expected or observed information matrix.
            Only relevant when computing robust standard errors.
            If not specified, the model's ``expected_information`` attribute is used.
        context : Optional[Union[int, Mapping[str, Any]]], default=None
            The context to add to the evaluation context of the formula with,
            e.g., custom transforms. If an integer, the context is taken from
            the stack frame of the caller at the given depth. Otherwise, a
            mapping from variable names to values is expected. By default,
            no context is added. Set ``context=0`` to make the calling scope
            available.

        Returns
        -------
        pandas.DataFrame
            A table of the regression results.
        """
        if self.fit_intercept:
            names = ["intercept"] + list(self.feature_names_)
            beta = np.concatenate([[self.intercept_], self.coef_])
        else:
            names = self.feature_names_
            beta = self.coef_

        captured_context = capture_context(
            context + 1 if isinstance(context, int) else context
        )
        if (X is None) and (getattr(self, "covariance_matrix_", None) is None):
            return pd.Series(beta, index=names, name="coef")

        covariance_matrix = self.covariance_matrix(
            X=X,
            y=y,
            mu=mu,
            offset=offset,
            sample_weight=sample_weight,
            dispersion=dispersion,
            robust=robust,
            clusters=clusters,
            expected_information=expected_information,
            context=captured_context,
        )

        significance_level = 1 - confidence_level

        std_errors = np.sqrt(np.diag(covariance_matrix))
        ci_lower = beta + stats.norm.ppf(significance_level / 2) * std_errors
        ci_upper = beta + stats.norm.ppf(1 - significance_level / 2) * std_errors
        t_values = beta / std_errors
        p_values = 2.0 * (1.0 - stats.norm.cdf(np.abs(t_values)))

        return pd.DataFrame(
            {
                "coef": beta,
                "se": std_errors,
                "t_value": t_values,
                "p_value": p_values,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
            },
            index=names,
        )

    def wald_test(
        self,
        X=None,
        y=None,
        sample_weight=None,
        offset=None,
        *,
        R: Optional[np.ndarray] = None,
        features: Optional[Union[str, list[str]]] = None,
        terms: Optional[Union[str, list[str]]] = None,
        formula: Optional[str] = None,
        r: Optional[Sequence] = None,
        mu=None,
        dispersion=None,
        robust=None,
        clusters: np.ndarray = None,
        expected_information=None,
        context: Optional[Union[int, Mapping[str, Any]]] = None,
    ) -> WaldTestResult:
        """Compute the Wald test statistic and p-value for a linear hypothesis.

        The left hand side of the hypothesis may be specified in the following ways:

        - ``R``: The restriction matrix representing the linear combination of
          coefficients to test.
        - ``features``: The name of a feature or a list of features to test.
        - ``terms``: The name of a term or a list of terms to test.
        - ``formula``: A formula string specifying the hypothesis to test.

        The right hand side of the tested hypothesis is specified by ``r``. In the
        case of a ``terms``-based test, the null hypothesis is that each coefficient
        relating to a term equals the corresponding value in ``r``.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features), optional
            Training data. Can be omitted if a covariance matrix has already
            been computed.
        y : array-like, shape (n_samples,), optional
            Target values. Can be omitted if a covariance matrix has already
            been computed.
        sample_weight : array-like, shape (n_samples,), optional, default=None
            Individual weights for each sample.
        offset : array-like, optional, default=None
            Array with additive offsets.
        R : np.ndarray, optional, default=None
            The restriction matrix representing the linear combination of coefficients
            to test.
        features : Union[str, list[str]], optional, default=None
            The name of a feature or a list of features to test.
        terms : Union[str, list[str]], optional, default=None
            The name of a term or a list of terms to test. It can cover one or more
            coefficients. In the case of a model based on a formula, a term is one
            of the expressions separated by ``+`` signs. Otherwise, a term is one column
            in the input data. As categorical variables need not be one-hot encoded in
            glum, in their case, the hypothesis to be tested is that the coefficients
            of all categories are equal to ``r``.
        r : Sequence, optional, default=None
            The vector representing the values of the linear combination.
            If None, the test is for whether the linear combinations of the coefficients
            are zero.
        mu : array-like, optional, default=None
            Array with predictions. Estimated if absent.
        dispersion : float, optional, default=None
            The dispersion parameter. Estimated if absent.
        robust : boolean, optional, default=None
            Whether to compute robust standard errors instead of normal ones.
            If not specified, the model's ``robust`` attribute is used.
        clusters : array-like, optional, default=None
            Array with cluster membership. Clustered standard errors are
            computed if clusters is not None.
        expected_information : boolean, optional, default=None
            Whether to use the expected or observed information matrix.
            Only relevant when computing robust standard errors.
            If not specified, the model's ``expected_information`` attribute is used.
        context : Optional[Union[int, Mapping[str, Any]]], default=None
            The context to add to the evaluation context of the formula with,
            e.g., custom transforms. If an integer, the context is taken from
            the stack frame of the caller at the given depth. Otherwise, a
            mapping from variable names to values is expected. By default,
            no context is added. Set ``context=0`` to make the calling scope
            available.

        Returns
        -------
        WaldTestResult
            NamedTuple with test statistic, p-value, and degrees of freedom.
        """

        num_lhs_specs = sum(
            [
                R is not None,
                features is not None,
                terms is not None,
                formula is not None,
            ]
        )
        if num_lhs_specs != 1:
            raise ValueError(
                "Exactly one of R, features, terms or formula must be specified. "
                f"Received {num_lhs_specs} specifications."
            )

        captured_context = capture_context(
            context + 1 if isinstance(context, int) else context
        )

        kwargs = {
            "X": X,
            "y": y,
            "sample_weight": sample_weight,
            "offset": offset,
            "mu": mu,
            "dispersion": dispersion,
            "robust": robust,
            "clusters": clusters,
            "expected_information": expected_information,
            "context": captured_context,
        }

        if R is not None:
            return self._wald_test_matrix(R=R, r=np.asarray(r), **kwargs)
        if features is not None:
            return self._wald_test_feature_names(features=features, values=r, **kwargs)
        if terms is not None:
            return self._wald_test_term_names(terms=terms, values=r, **kwargs)
        if formula is not None:
            if r is not None:
                raise ValueError("Cannot specify both formula and r")
            return self._wald_test_formula(formula=formula, **kwargs)

        raise RuntimeError("This should never happen")

    def _wald_test_matrix(
        self, R: np.ndarray, r: Optional[np.ndarray] = None, **kwargs
    ) -> WaldTestResult:
        """
        Perform a Wald test statistic for a hypothesis specified by constraints
        given as ``R @ coef_ = r``. Under the null hypothesis, the test statistic
        follows a chi-squared distribution with ``R.shape[0]`` degrees of freedom.
        """
        covariance_matrix = self.covariance_matrix(**kwargs)

        if self.fit_intercept:
            beta = np.concatenate([[self.intercept_], self.coef_])
        else:
            beta = self.coef_

        if r is None:
            r = np.zeros(R.shape[0])

        if R.shape[0] != r.shape[0]:
            raise ValueError("R and r must have the same number of rows")
        if R.shape[1] != beta.shape[0]:
            raise ValueError("R must have one column for each coefficient")
        # There is no point in checking that R is full rank. If it is not, then
        # solve(RVR, Rb_r) will raise an exception.

        beta = beta[:, np.newaxis]
        r = r[:, np.newaxis]
        Q = R.shape[0]

        Rb_r = R @ beta - r  # R \beta - r
        RVR = R @ covariance_matrix @ R.T  # R V R^T

        # We want to calculate Rb_r^T (RVR)^{-1} Rb_r.
        # We can do it in a more numerically stable way by using `scipy.linalg.solve`:
        try:
            test_stat = (Rb_r.T @ linalg.solve(RVR, Rb_r))[0]
        except linalg.LinAlgError as err:
            raise linalg.LinAlgError("The restriction matrix is not full rank") from err
        p_value = 1 - stats.chi2.cdf(test_stat, Q)

        return WaldTestResult(test_stat, p_value, Q)

    def _wald_test_feature_names(
        self,
        features: Union[str, list[str]],
        values: Optional[Sequence] = None,
        **kwargs,
    ) -> WaldTestResult:
        """
        Perform a Wald test for the hypothesis that the coefficients of the
        features in ``features`` are equal to the values in ``values``.
        """

        if isinstance(features, str):
            features = [features]

        if values is not None:
            r = np.array(values)
            if len(features) != len(values):
                raise ValueError("features and values must have the same length")
        else:
            r = None

        if self.fit_intercept:
            names = ["intercept"] + list(self.feature_names_)
            beta = np.concatenate([[self.intercept_], self.coef_])
        else:
            names = self.feature_names_
            beta = self.coef_

        R = np.zeros((len(features), len(beta)))
        for i, feature in enumerate(features):
            try:
                j = names.index(feature)
            except ValueError:
                raise ValueError(f"feature {feature} is not in the model") from None
            R[i, j] = 1

        return self._wald_test_matrix(R=R, r=r, **kwargs)

    def _wald_test_formula(self, formula: str, **kwargs) -> WaldTestResult:
        """
        Perform a Wald test for the hypothesis described in ``formula``.
        """

        if self.fit_intercept:
            names = ["intercept"] + list(self.feature_names_)
        else:
            names = self.feature_names_

        parser = LinearConstraintParser(names)

        R, r = parser.get_matrix(formula)

        return self._wald_test_matrix(R=R, r=r, **kwargs)

    def _wald_test_term_names(
        self, terms: Union[str, list[str]], values: Optional[Sequence] = None, **kwargs
    ) -> WaldTestResult:
        """
        Perform a Wald test for the hypothesis that the coefficients of the
        features in ``terms`` are equal to the values in ``values``.
        """

        if isinstance(terms, str):
            terms = [terms]

        if values is not None:
            rhs = True
            if len(terms) != len(values):
                raise ValueError("terms and values must have the same length")
        else:
            rhs = False
            values = [None] * len(terms)

        if self.fit_intercept:
            names = np.array(["intercept"] + list(self.term_names_))
            beta = np.concatenate([[self.intercept_], self.coef_])
        else:
            names = np.array(self.term_names_)
            beta = self.coef_

        R_list = []
        r_list = []
        for term, value in zip(terms, values):
            R_indices, *_ = np.where(names == term)
            num_restrictions = len(R_indices)
            if num_restrictions == 0:
                raise ValueError(f"term {term} is not in the model")
            R_current = np.zeros((num_restrictions, len(beta)), dtype=np.float64)
            R_current[np.arange(num_restrictions), R_indices] = 1.0
            R_list.append(R_current)

            if rhs:
                r_list.append(np.full(num_restrictions, fill_value=value))

        R = np.vstack(R_list)
        r = np.concatenate(r_list) if rhs else None

        return self._wald_test_matrix(R=R, r=r, **kwargs)

    def std_errors(
        self,
        X=None,
        y=None,
        sample_weight=None,
        offset=None,
        *,
        mu=None,
        dispersion=None,
        robust=None,
        clusters: np.ndarray = None,
        expected_information=None,
        store_covariance_matrix=False,
        context: Optional[Union[int, Mapping[str, Any]]] = None,
    ):
        """Calculate standard errors for generalized linear models.

        See `covariance_matrix` for an in-depth explanation of how the
        standard errors are computed.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features), optional
            Training data. Can be omitted if a covariance matrix has already
            been computed.
        y : array-like, shape (n_samples,), optional
            Target values. Can be omitted if a covariance matrix has already
            been computed.
        sample_weight : array-like, shape (n_samples,), optional, default=None
            Individual weights for each sample.
        offset : array-like, optional, default=None
            Array with additive offsets.
        mu : array-like, optional, default=None
            Array with predictions. Estimated if absent.
        dispersion : float, optional, default=None
            The dispersion parameter. Estimated if absent.
        robust : boolean, optional, default=None
            Whether to compute robust standard errors instead of normal ones.
            If not specified, the model's ``robust`` attribute is used.
        clusters : array-like, optional, default=None
            Array with cluster membership. Clustered standard errors are
            computed if clusters is not None.
        expected_information : boolean, optional, default=None
            Whether to use the expected or observed information matrix.
            Only relevant when computing robust standard errors.
            If not specified, the model's ``expected_information`` attribute is used.
        store_covariance_matrix : boolean, optional, default=False
            Whether to store the covariance matrix in the model instance.
            If a covariance matrix has already been stored, it will be overwritten.
        context : Optional[Union[int, Mapping[str, Any]]], default=None
            The context to add to the evaluation context of the formula with,
            e.g., custom transforms. If an integer, the context is taken from
            the stack frame of the caller at the given depth. Otherwise, a
            mapping from variable names to values is expected. By default,
            no context is added. Set ``context=0`` to make the calling scope
            available.
        """
        captured_context = capture_context(
            context + 1 if isinstance(context, int) else context
        )

        return np.sqrt(
            self.covariance_matrix(
                X=X,
                y=y,
                sample_weight=sample_weight,
                offset=offset,
                mu=mu,
                dispersion=dispersion,
                robust=robust,
                clusters=clusters,
                expected_information=expected_information,
                store_covariance_matrix=store_covariance_matrix,
                context=captured_context,
            ).diagonal()
        )

    def covariance_matrix(
        self,
        X=None,
        y=None,
        sample_weight=None,
        offset=None,
        *,
        mu=None,
        dispersion=None,
        robust=None,
        clusters: Optional[np.ndarray] = None,
        expected_information=None,
        store_covariance_matrix=False,
        skip_checks=False,
        context: Optional[Union[int, Mapping[str, Any]]] = None,
    ):
        """Calculate the covariance matrix for generalized linear models.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features), optional
            Training data. Can be omitted if a covariance matrix has already
            been computed.

        y : array-like, shape (n_samples,), optional
            Target values. Can be omitted if a covariance matrix has already
            been computed.

        mu : array-like, optional, default=None
            Array with predictions. Estimated if absent.

        offset : array-like, optional, default=None
            Array with additive offsets.

        sample_weight : array-like, shape (n_samples,), optional, default=None
            Individual weights for each sample.

        dispersion : float, optional, default=None
            The dispersion parameter. Estimated if absent.

        robust : boolean, optional, default=None
            Whether to compute robust standard errors instead of normal ones.
            If not specified, the model's ``robust`` attribute is used.

        clusters : array-like, optional, default=None
            Array with cluster membership. Clustered standard errors are
            computed if clusters is not None.

        expected_information : boolean, optional, default=None
            Whether to use the expected or observed information matrix.
            Only relevant when computing robust standard errors.
            If not specified, the model's ``expected_information`` attribute is used.

        store_covariance_matrix : boolean, optional, default=False
            Whether to store the covariance matrix in the model instance.
            If a covariance matrix has already been stored, it will be overwritten.

        skip_checks : boolean, optional, default=False
            Whether to skip input validation. For internal use only.

        context : Optional[Union[int, Mapping[str, Any]]], default=None
            The context to add to the evaluation context of the formula with,
            e.g., custom transforms. If an integer, the context is taken from
            the stack frame of the caller at the given depth. Otherwise, a
            mapping from variable names to values is expected. By default,
            no context is added. Set ``context=0`` to make the calling scope
            available.

        Notes
        -----
        We support three types of covariance matrices:

        - non-robust
        - robust (HC-1)
        - clustered

        For maximum-likelihood estimator, the covariance matrix takes the form
        :math:`\\mathcal{H}^{-1}(\\theta_0)\\mathcal{I}(\\theta_0)
        \\mathcal{H}^{-1}(\\theta_0)` where :math:`\\mathcal{H}^{-1}` is the
        inverse Hessian and :math:`\\mathcal{I}` is the Information matrix.
        The different types of covariance matrices use different approximation
        of these quantities.

        The non-robust covariance matrix is computed as the inverse of the Fisher
        information matrix. This assumes that the information matrix equality holds.

        The robust (HC-1) covariance matrix takes the form :math:`\\mathbf{H}^{−1}
        (\\hat{\\theta})\\mathbf{G}^{T}(\\hat{\\theta})\\mathbf{G}(\\hat{\\theta})
        \\mathbf{H}^{−1}(\\hat{\\theta})` where :math:`\\mathbf{H}` is the empirical
        Hessian and :math:`\\mathbf{G}` is the gradient. We apply a finite-sample
        correction of :math:`\\frac{N}{N-p}`.

        The clustered covariance matrix uses a similar approach to the robust (HC-1)
        covariance matrix. However, instead of using :math:`\\mathbf{G}^{T}(
        \\hat{\\theta}\\mathbf{G}(\\hat{\\theta})` directly, we first sum over
        all the groups first. The finite-sample correction is affected as well,
        becoming :math:`\\frac{M}{M-1}\\frac{N}{N-p}` where :math:`M` is the number
        of groups.

        References
        ----------
        .. Davidson, Russell & MacKinnon, James G. (1993).
           "Estimation and Inference in Econometrics," OUP Catalogue,
           Oxford University Press

        .. Cameron, A. C., & Trivedi, P. K. (2005).
           "Microeconometrics: methods and applications,"
           Cambridge university press

        """
        self.covariance_matrix_: Union[np.ndarray, None]

        captured_context = capture_context(
            context + 1 if isinstance(context, int) else context
        )

        if robust is None:
            _robust = getattr(self, "robust", True)
        else:
            _robust = robust

        if expected_information is None:
            _expected_information = getattr(self, "expected_information", False)
        else:
            _expected_information = expected_information

        if (
            (
                hasattr(self, "alpha")
                and isinstance(self.alpha, (int, float))
                and self.alpha > 0
            )
            or (hasattr(self, "alpha_") and self.alpha_ > 0)  # glm_cv
            or (hasattr(self, "_alphas") and self._alphas[-1] > 0)  # alpha_search
        ):
            warnings.warn(
                "Covariance matrix estimation assumes that the model is not "
                "penalized. You are estimating a penalized model. The covariance "
                "matrix will be incorrect."
            )

        cannot_estimate_cov = (y is None) and not hasattr(self, "y_model_spec_")
        cannot_estimate_cov |= X is None

        if not skip_checks:
            if cannot_estimate_cov and self.covariance_matrix_ is None:
                raise ValueError(
                    "Either X and y must be provided or the covariance matrix "
                    "must have been previously computed."
                )

            if cannot_estimate_cov and store_covariance_matrix:
                raise ValueError(
                    "X and y must be provided if 'store_covariance_matrix' is True."
                )

            if store_covariance_matrix and self.covariance_matrix_ is not None:
                warnings.warn(
                    "A covariance matrix has already been computed. "
                    "It will be overwritten."
                )

            if X is None and y is None:
                if (
                    offset is not None
                    or mu is not None
                    or offset is not None
                    or sample_weight is not None
                    or dispersion is not None
                    or robust is not None
                    or clusters is not None
                    or expected_information is not None
                ):
                    raise ValueError(
                        "Cannot reestimate the covariance matrix with different "
                        "parameters if X and y are not provided."
                    )
                return self.covariance_matrix_

            if hasattr(self, "y_model_spec_"):
                y = self.y_model_spec_.get_model_matrix(X).toarray().ravel()
                # This has to go first because X is modified in the next line

            if isinstance(X, pd.DataFrame):
                X = self._convert_from_pandas(X, context=captured_context)

            X, y = check_X_y_tabmat_compliant(
                X,
                y,
                accept_sparse=["csr", "csc", "coo"],
                dtype="numeric",
                copy=self._should_copy_X(),
                ensure_2d=True,
                allow_nd=False,
                drop_first=getattr(self, "drop_first", False),
            )

            if isinstance(X, np.ndarray):
                X = tm.DenseMatrix(X)
            if sparse.issparse(X) and not isinstance(X, tm.SparseMatrix):
                X = tm.SparseMatrix(X)

            sample_weight = _check_weights(
                sample_weight,
                y.shape[0],
                X.dtype,
                force_all_finite=self.force_all_finite,
            )
            offset = _check_offset(offset, y.shape[0], X.dtype)

        sum_weights = np.sum(sample_weight)  # type: ignore

        mu = self.predict(X, offset=offset) if mu is None else np.asanyarray(mu)

        if dispersion is None:
            # sample_weight here need to be non-normalized to count the number
            # of observations.
            dispersion = self._family_instance.dispersion(
                y,
                mu,
                sample_weight=sample_weight,
                ddof=X.shape[1] + self.fit_intercept,
                method="pearson",
            )

        if (
            np.linalg.cond(_safe_toarray(X.sandwich(np.ones(X.shape[0]))))
            > 1 / sys.float_info.epsilon**2
        ):
            raise np.linalg.LinAlgError(
                "Matrix is singular. Cannot estimate standard errors."
            )

        if _robust or clusters is not None:
            if _expected_information:
                oim_fct = self._family_instance._fisher_information
            else:
                oim_fct = self._family_instance._observed_information
            oim = oim_fct(
                self._link_instance,
                X,
                y,
                mu,
                sample_weight,
                dispersion,
                self.fit_intercept,
            )
            gradient = self._family_instance._score_matrix(
                self._link_instance,
                X,
                y,
                mu,
                sample_weight,
                dispersion,
                self.fit_intercept,
            )
            if clusters is not None:
                n_groups = len(np.unique(clusters))
                grouped_gradient = _group_sum(clusters, gradient)
                inner_part = grouped_gradient.T @ grouped_gradient
                correction = (n_groups / (n_groups - 1)) * (
                    (sum_weights - 1)
                    / (sum_weights - self.n_features_in_ - int(self.fit_intercept))
                )
            else:
                inner_part = gradient.sandwich(np.ones_like(y, dtype=X.dtype))
                correction = sum_weights / (
                    sum_weights - self.n_features_in_ - int(self.fit_intercept)
                )
            vcov = linalg.solve(oim, linalg.solve(oim, _safe_toarray(inner_part)).T)
            vcov *= correction
        else:
            fisher = self._family_instance._fisher_information(
                self._link_instance,
                X,
                y,
                mu,
                sample_weight,
                dispersion,
                self.fit_intercept,
            )
            vcov = linalg.inv(_safe_toarray(fisher))
            vcov *= sum_weights / (
                sum_weights - self.n_features_in_ - int(self.fit_intercept)
            )

        if store_covariance_matrix:
            self.covariance_matrix_ = vcov

        return vcov

    # Note: check_estimator(GeneralizedLinearRegressor) might raise
    # "AssertionError: -0.28014056555724598 not greater than 0.5"
    # unless GeneralizedLinearRegressor has a score which passes the test.
    def score(
        self,
        X: ShapedArrayLike,
        y: ShapedArrayLike,
        sample_weight: Optional[ArrayLike] = None,
        offset: Optional[ArrayLike] = None,
        *,
        context: Optional[Union[int, Mapping[str, Any]]] = None,
    ):
        """Compute :math:`D^2`, the percentage of deviance explained.

        :math:`D^2` is a generalization of the coefficient of determination
        :math:`R^2`. The :math:`R^2` uses the squared error and the :math:`D^2`,
        the deviance. Note that those two are equal for ``family='normal'``.

        :math:`D^2` is defined as
        :math:`D^2 = 1 - \\frac{D(y_{\\mathrm{true}}, y_{\\mathrm{pred}})}
        {D_{\\mathrm{null}}}`,
        :math:`D_{\\mathrm{null}}` is the null deviance, i.e. the deviance of a
        model with intercept alone. The best possible score is one and it can be
        negative.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Test samples.

        y : array-like, shape (n_samples,)
            True values of target.

        sample_weight : array-like, shape (n_samples,), optional (default=None)
            Sample weights.

        offset : array-like, shape (n_samples,), optional (default=None)

        context : Optional[Union[int, Mapping[str, Any]]], default=None
            The context to add to the evaluation context of the formula with,
            e.g., custom transforms. If an integer, the context is taken from
            the stack frame of the caller at the given depth. Otherwise, a
            mapping from variable names to values is expected. By default,
            no context is added. Set ``context=0`` to make the calling scope
            available.

        Returns
        -------
        float
            D^2 of self.predict(X) w.r.t. y.
        """
        # Note, default score defined in RegressorMixin is R^2 score.
        # TODO: make D^2 a score function in module metrics (and thereby get
        #       input validation and so on)
        captured_context = capture_context(
            context + 1 if isinstance(context, int) else context
        )

        sample_weight = _check_weights(sample_weight, y.shape[0], y.dtype)
        mu = self.predict(X, offset=offset, context=captured_context)
        family = get_family(self.family)
        dev = family.deviance(y, mu, sample_weight=sample_weight)
        y_mean = np.average(y, weights=sample_weight)
        dev_null = family.deviance(y, y_mean, sample_weight=sample_weight)
        return 1.0 - dev / dev_null

    def _validate_hyperparameters(self) -> None:
        if not isinstance(self.fit_intercept, bool):
            raise TypeError(
                f"The argument fit_intercept must be bool; got {self.fit_intercept}."
            )
        if self.solver == "newton-cg":
            raise ValueError(
                """
                newton-cg solver is no longer supported because
                sklearn.utils.optimize.newton_cg has been deprecated. If you need this
                functionality, please use
                https://github.com/scikit-learn/scikit-learn/pull/9405.
                """
            )
        if self.solver not in ["auto", "irls-ls", "lbfgs", "irls-cd", "trust-constr"]:
            raise ValueError(
                "GeneralizedLinearRegressor supports only solvers"
                " 'auto', 'irls-ls', 'lbfgs', 'irls-cd' and 'trust-constr'; "
                f"got (solver={self.solver})."
            )
        if not isinstance(self.max_iter, int) or self.max_iter <= 0:
            raise ValueError(
                "Maximum number of iteration must be a positive integer; "
                f"got (max_iter={self.max_iter})."
            )
        if self.gradient_tol is not None:
            if (
                not isinstance(self.gradient_tol, (float, int))
                or self.gradient_tol <= 0
            ):
                raise ValueError(
                    "Tolerance for the gradient stopping criteria must be positive; "
                    f"got (gradient_tol={self.gradient_tol})."
                )
        if self.step_size_tol is not None and (
            not isinstance(self.step_size_tol, (float, int)) or self.step_size_tol <= 0
        ):
            raise ValueError(
                "Tolerance for the step-size stopping criteria must be positive; "
                f"got (step_size_tol={self.step_size_tol})."
            )
        if not isinstance(self.warm_start, bool):
            raise TypeError(
                f"The argument warm_start must be bool; got {self.warm_start}."
            )
        if self.selection not in ["cyclic", "random"]:
            raise ValueError(
                "The argument selection must be 'cyclic' or 'random'; "
                f"got {self.selection}."
            )
        if self.copy_X is not None and not isinstance(self.copy_X, bool):
            raise TypeError(
                f"The argument copy_X must be None or bool; got {self.copy_X}."
            )
        if not isinstance(self.check_input, bool):
            raise TypeError(
                f"The argument check_input must be bool; got {self.check_input}."
            )
        if self.scale_predictors and not self.fit_intercept:
            raise ValueError(
                "scale_predictors=True is not supported when fit_intercept=False."
            )
        if ((self.lower_bounds is not None) or (self.upper_bounds is not None)) and (
            self.solver not in ["auto", "irls-cd"]
        ):
            raise ValueError(
                "Only the 'cd' solver is supported when bounds are set; "
                f"got {self.solver}."
            )
        if ((self.A_ineq is not None) or (self.b_ineq is not None)) and (
            self.solver not in [None, "auto", "trust-constr"]
        ):
            raise ValueError(
                "Only the 'trust-constr' solver supports inequality constraints; "
                f"got {self.solver}."
            )
        if ((self.A_ineq is not None) or (self.b_ineq is not None)) and (
            (self.lower_bounds is not None) or (self.upper_bounds is not None)
        ):
            raise NotImplementedError(
                "Only either bound or inequality constraints are supported."
            )
        if ((self.A_ineq is not None) and (self.b_ineq is None)) or (
            (self.A_ineq is None) and (self.b_ineq is not None)
        ):
            raise ValueError("Must provide both A_ineq and b_ineq.")
        if self.check_input:
            # check if P1 has only non-negative values, negative values might
            # indicate group lasso in the future.
            if not isinstance(self.P1, str):  # if self.P1 != 'identity':
                if not np.all(np.asarray(self.P1) >= 0):
                    raise ValueError("P1 must not have negative values.")

    def _should_copy_X(self):
        # If self.copy_X is True, copy_X is True
        # If self.copy_X is None, copy_X is False. Check for data of wrong dtype and
        # fix if necessary.
        # If self.copy_X is False, check for data of wrong dtype and error if it exists.
        return self.copy_X or False

    def _set_up_and_check_fit_args(
        self,
        X: ArrayLike,
        y: Optional[ArrayLike],
        sample_weight: Optional[VectorLike],
        offset: Optional[VectorLike],
        force_all_finite,
        context: Optional[Mapping[str, Any]] = None,
    ) -> tuple[
        tm.MatrixBase,
        np.ndarray,
        np.ndarray,
        Optional[np.ndarray],
        float,
        Union[str, np.ndarray, Any],
        Union[str, np.ndarray, Any],
    ]:
        dtype = [np.float64, np.float32]
        stype = ["csc"] if self.solver == "irls-cd" else ["csc", "csr"]

        P1 = self.P1
        P2 = self.P2

        copy_X = self._should_copy_X()
        drop_first = getattr(self, "drop_first", False)

        if isinstance(X, pd.DataFrame):
            if hasattr(self, "formula") and self.formula is not None:
                lhs, rhs = _parse_formula(
                    self.formula, include_intercept=self.fit_intercept
                )

                if lhs is not None:
                    if y is not None:
                        raise ValueError(
                            "`y` is not allowed when using a two-sided formula. "
                            "Either set `y=None` or use a one-sided formula."
                        )

                    y = tm.from_formula(
                        formula=lhs,
                        data=X,
                        include_intercept=False,
                        context=context,
                    )

                    self.y_model_spec_ = y.model_spec  # type: ignore
                    y = y.toarray().ravel()  # type: ignore

                X = tm.from_formula(
                    formula=rhs,
                    data=X,
                    include_intercept=False,
                    ensure_full_rank=self.drop_first,
                    categorical_format=self.categorical_format,
                    cat_missing_method=self.cat_missing_method,
                    interaction_separator=self.interaction_separator,
                    add_column_for_intercept=False,
                    context=context,
                )

                intercept = "1" in X.model_spec.terms
                if intercept != self.fit_intercept:
                    raise ValueError(
                        f"The formula sets the intercept to {intercept}, "
                        f"contradicting fit_intercept={self.fit_intercept}. "
                        "You should use fit_intercept to specify the intercept."
                    )

                self.X_model_spec_ = X.model_spec
                self.feature_names_ = list(X.model_spec.column_names)

                self.term_names_ = [
                    term
                    for term, _, cols in X.model_spec.structure
                    for _ in range(len(cols))
                ]

            else:
                # Maybe TODO: expand categorical penalties with formulas

                self.feature_dtypes_ = X.dtypes.to_dict()

                self.has_missing_category_ = {
                    col: (getattr(self, "cat_missing_method", "fail") == "convert")
                    and X[col].isna().any()
                    for col, dtype in self.feature_dtypes_.items()
                    if isinstance(dtype, pd.CategoricalDtype)
                }

                if any(X.dtypes == "category"):
                    P1 = _expand_categorical_penalties(
                        self.P1, X, drop_first, self.has_missing_category_
                    )
                    P2 = _expand_categorical_penalties(
                        self.P2, X, drop_first, self.has_missing_category_
                    )

                X = tm.from_pandas(
                    X,
                    drop_first=drop_first,
                    categorical_format=getattr(  # convention prior to v3
                        self, "categorical_format", "{name}__{category}"
                    ),
                    cat_missing_method=getattr(self, "cat_missing_method", "fail"),
                    cat_missing_name=getattr(self, "cat_missing_name", "(MISSING)"),
                )

        if y is None:
            raise ValueError(
                f"Unless using a two-sided formula, {self.__class__.__name__} "
                "requires y to be passed, but the target y is None."
            )

        if not _is_contiguous(X):
            if self.copy_X is not None and not self.copy_X:
                raise ValueError(
                    "The X matrix is noncontiguous and copy_X = False."
                    "To fix this, either set copy_X = None or pass a contiguous matrix."
                )
            X = X.copy()

        if (
            not isinstance(X, tm.CategoricalMatrix)
            and hasattr(X, "dtype")
            and np.issubdtype(X.dtype, np.integer)  # type: ignore
        ):
            if self.copy_X is not None and not self.copy_X:
                raise ValueError(
                    "Integer data needs to be converted to float, but you specified "
                    "copy_X = False. To fix this, set copy_X = None or convert to "
                    "float yourself."
                )
            # check_X_y will convert to float32 if we don't do this, which causes
            # precision issues with the new handling of single precision. The new
            # behavior is to give everything the precision of X, but we don't want to
            # do that if X was initially int64.
            X = X.astype(np.float64)  # type: ignore

        if isinstance(X, tm.MatrixBase):
            X, y = check_X_y_tabmat_compliant(
                X,
                y,
                accept_sparse=stype,
                dtype=dtype,
                copy=copy_X,
                drop_first=getattr(self, "drop_first", False),
                **{keyword_finiteness: force_all_finite},
            )
            self.n_features_in_ = X.shape[1]
        else:
            X, y = validate_data(
                self,
                X,
                y,
                ensure_2d=True,
                accept_sparse=stype,
                dtype=dtype,
                copy=copy_X,
                **{keyword_finiteness: force_all_finite},
            )

        # Without converting y to float, deviance might raise
        # ValueError: Integers to negative integer powers are not allowed.
        # Also, y must not be sparse.
        # Make sure everything has the same precision as X
        # This will prevent accidental upcasting later and slow operations on
        # mixed-precision numbers
        y = np.asarray(y, dtype=X.dtype)

        sample_weight = _check_weights(
            sample_weight,
            y.shape[0],  # type: ignore
            X.dtype,
            force_all_finite=force_all_finite,
        )

        offset = _check_offset(offset, y.shape[0], X.dtype)  # type: ignore

        # IMPORTANT NOTE: Since we want to minimize
        # 1/(2*sum(sample_weight)) * deviance + L1 + L2,
        # deviance = sum(sample_weight * unit_deviance),
        # we rescale weights such that sum(weights) = 1 and this becomes
        # 1/2*deviance + L1 + L2 with deviance=sum(weights * unit_deviance)
        weights_sum: float = np.sum(sample_weight)  # type: ignore
        sample_weight = sample_weight / weights_sum

        # Convert to wrapper matrix types
        X = tm.as_tabmat(X)

        self.feature_names_ = X.get_names(type="column", missing_prefix="_col_")  # type: ignore
        self.term_names_ = X.get_names(type="term", missing_prefix="_col_")

        return X, y, sample_weight, offset, weights_sum, P1, P2  # type: ignore


class GeneralizedLinearRegressor(GeneralizedLinearRegressorBase):
    """Regression via a Generalized Linear Model (GLM) with penalties.

    GLMs based on a reproductive Exponential Dispersion Model (EDM) aimed at
    fitting and predicting the mean of the target ``y`` as ``mu=h(X*w)``.
    Therefore, the fit minimizes the following objective function with combined
    L1 and L2 priors as regularizer::

            1/(2*sum(s)) * deviance(y, h(X*w); s)
            + alpha * l1_ratio * ||P1*w||_1
            + 1/2 * alpha * (1 - l1_ratio) * w*P2*w

    with inverse link function ``h`` and ``s=sample_weight``.
    Note that, for ``alpha=0`` the unregularized GLM is recovered.
    This is not the default behavior (see ``alpha`` parameter description for details).
    Additionally, for ``sample_weight=None``, one has ``s_i=1`` and
    ``sum(s)=n_samples``. For ``P1=P2='identity'``, the penalty is the elastic net::

            alpha * l1_ratio * ||w||_1 + 1/2 * alpha * (1 - l1_ratio) * ||w||_2^2.

    If you are interested in controlling the L1 and L2 penalties separately,
    keep in mind that this is equivalent to::

            a * L1 + b * L2,

    where::

            alpha = a + b and l1_ratio = a / (a + b).

    The parameter ``l1_ratio`` corresponds to alpha in the R package glmnet,
    while ``alpha`` corresponds to the lambda parameter in glmnet.
    Specifically, ``l1_ratio = 1`` is the lasso penalty.

    Read more in :doc:`background<background>`.

    Parameters
    ----------
    alpha : {float, array-like}, optional (default=None)
        Constant that multiplies the penalty terms and thus determines the
        regularization strength. If ``alpha_search`` is ``False`` (the default),
        then ``alpha`` must be a scalar or None (equivalent to ``alpha=0``).
        If ``alpha_search`` is ``True``, then ``alpha`` must be an iterable or
        ``None``. See ``alpha_search`` to find how the regularization path is
        set if ``alpha`` is ``None``. See the notes for the exact mathematical
        meaning of this parameter. ``alpha=0`` is equivalent to unpenalized
        GLMs. In this case, the design matrix ``X`` must have full column rank
        (no collinearities).

    l1_ratio : float, optional (default=0)
        The elastic net mixing parameter, with ``0 <= l1_ratio <= 1``. For
        ``l1_ratio = 0``, the penalty is an L2 penalty. ``For l1_ratio = 1``, it
        is an L1 penalty.  For ``0 < l1_ratio < 1``, the penalty is a
        combination of L1 and L2.

    P1 : {'identity', array-like, None}, shape (n_features,), optional
         (default='identity')
        This array controls the strength of the regularization for each coefficient
        independently. A high value will lead to higher regularization while a value of
        zero will remove the regularization on this parameter.
        Note that ``n_features = X.shape[1]``. If ``X`` is a pandas DataFrame
        with a categorical dtype and P1 has the same size as the number of columns,
        the penalty of the categorical column will be applied to all the levels of
        the categorical.

    P2 : {'identity', array-like, sparse matrix, None}, shape (n_features,) \
            or (n_features, n_features), optional (default='identity')
        With this option, you can set the P2 matrix in the L2 penalty
        ``w*P2*w``. This gives a fine control over this penalty (Tikhonov
        regularization). A 2d array is directly used as the square matrix P2. A
        1d array is interpreted as diagonal (square) matrix. The default
        ``'identity'`` and ``None`` set the identity matrix, which gives the usual
        squared L2-norm. If you just want to exclude certain coefficients, pass a 1d
        array filled with 1 and 0 for the coefficients to be excluded. Note that P2 must
        be positive semi-definite. If ``X`` is a pandas DataFrame with a categorical
        dtype and P2 has the same size as the number of columns, the penalty of the
        categorical column will be applied to all the levels of the categorical. Note
        that if P2 is two-dimensional, its size needs to be of the same length as the
        expanded ``X`` matrix.

    fit_intercept : bool, optional (default=True)
        Specifies if a constant (a.k.a. bias or intercept) should be
        added to the linear predictor (``X * coef + intercept``).

    family : str or ExponentialDispersionModel, optional (default='normal')
        The distributional assumption of the GLM, i.e. the loss function to
        minimize. If a string, one of: ``'binomial'``, ``'gamma'``,
        ``'gaussian'``, ``'inverse.gaussian'``, ``'normal'``, ``'poisson'``,
        ``'tweedie'`` or ``'negative.binomial'``. Note that ``'tweedie'`` sets
        the power of the Tweedie distribution to 1.5; to use another value,
        specify it in parentheses (e.g., ``'tweedie (1.5)'``). The same applies
        for ``'negative.binomial'`` and theta parameter.

    link : {'auto', 'identity', 'log', 'logit', 'cloglog'} oe Link, \
            optional (default='auto')
        The link function of the GLM, i.e. mapping from linear
        predictor (``X * coef``) to expectation (``mu``). Option ``'auto'`` sets
        the link depending on the chosen family as follows:

        - ``'identity'`` for family ``'normal'``
        - ``'log'`` for families ``'poisson'``, ``'gamma'``,
          ``'inverse.gaussian'`` and ``'negative.binomial'``.
        - ``'logit'`` for family ``'binomial'``

    solver : {'auto', 'irls-cd', 'irls-ls', 'lbfgs', 'trust-constr'}, \
            optional (default='auto')
        Algorithm to use in the optimization problem:

        - ``'auto'``: ``'irls-ls'`` if ``l1_ratio`` is zero and ``'irls-cd'`` otherwise.
        - ``'irls-cd'``: Iteratively reweighted least squares with a coordinate
          descent inner solver. This can deal with L1 as well as L2 penalties.
          Note that in order to avoid unnecessary memory duplication of X in the
          ``fit`` method, ``X`` should be directly passed as a
          Fortran-contiguous Numpy array or sparse CSC matrix.
        - ``'irls-ls'``: Iteratively reweighted least squares with a least
          squares inner solver. This algorithm cannot deal with L1 penalties.
        - ``'lbfgs'``: Scipy's L-BFGS-B optimizer. It cannot deal with L1
          penalties.
        - ``'trust-constr'``: Calls
          ``scipy.optimize.minimize(method='trust-constr')``. It cannot deal
          with L1 penalties. This solver can optimize problems with inequality
          constraints, passed via ``A_ineq`` and ``b_ineq``. It will be selected
          automatically when inequality constraints are set and
          ``solver='auto'``. Note that using this method can lead to
          significantly increased runtimes by a factor of ten or higher.

    max_iter : int, optional (default=100)
        The maximal number of iterations for solver algorithms.

    max_inner_iter: int, optional (default=100000)
        The maximal number of iterations for the inner solver in the IRLS-CD
        algorithm. This parameter is only used when ``solver='irls-cd'``.

    gradient_tol : float, optional (default=None)
        Stopping criterion. If ``None``, solver-specific defaults will be used.
        The default value for most solvers is ``1e-4``, except for
        ``'trust-constr'``, which requires more conservative convergence
        settings and has a default value of ``1e-8``.

        For the IRLS-LS, L-BFGS and trust-constr solvers, the iteration
        will stop when ``max{|g_i|, i = 1, ..., n} <= tol``, where ``g_i`` is
        the ``i``-th component of the gradient (derivative) of the objective
        function. For the CD solver, convergence is reached when
        ``sum_i(|minimum norm of g_i|)``, where ``g_i`` is the subgradient of
        the objective and the minimum norm of ``g_i`` is the element of the
        subgradient with the smallest L2 norm.

        If you wish to only use a step-size tolerance, set ``gradient_tol``
        to a very small number.

    step_size_tol: float, optional (default=None)
        Alternative stopping criterion. For the IRLS-LS and IRLS-CD solvers, the
        iteration will stop when the L2 norm of the step size is less than
        ``step_size_tol``. This stopping criterion is disabled when
        ``step_size_tol`` is ``None``.

    hessian_approx: float, optional (default=0.0)
        The threshold below which data matrix rows will be ignored for updating
        the Hessian. See the algorithm documentation for the IRLS algorithm
        for further details.

    warm_start : bool, optional (default=False)
        Whether to reuse the solution of the previous call to ``fit``
        as initialization for ``coef_`` and ``intercept_`` (supersedes
        ``start_params``). If ``False`` or if the attribute ``coef_`` does not
        exist (first call to ``fit``), ``start_params`` sets the start values
        for ``coef_`` and ``intercept_``.

    alpha_search : bool, optional (default=False)
        Whether to search along the regularization path for the best alpha.
        When set to ``True``, ``alpha`` should either be ``None`` or an
        iterable. To determine the regularization path, the following sequence
        is used:

        1. If ``alpha`` is an iterable, use it directly. All other parameters
           governing the regularization path are ignored.
        2. If ``min_alpha`` is set, create a path from ``min_alpha`` to the
           lowest alpha such that all coefficients are zero.
        3. If ``min_alpha_ratio`` is set, create a path where the ratio of
           ``min_alpha / max_alpha = min_alpha_ratio``.
        4. If none of the above parameters are set, use a ``min_alpha_ratio`` of
           ``1e-6``.

    alphas : DEPRECATED. Use ``alpha`` instead.

    n_alphas : int, optional (default=100)
        Number of alphas along the regularization path

    min_alpha_ratio : float, optional (default=None)
        Length of the path. ``min_alpha_ratio=1e-6`` means that
        ``min_alpha / max_alpha = 1e-6``. If ``None``, ``1e-6`` is used.

    min_alpha : float, optional (default=None)
        Minimum alpha to estimate the model with. The grid will then be created
        over ``[max_alpha, min_alpha]``.

    start_params : array-like, shape (n_features*,), optional (default=None)
        Relevant only if ``warm_start`` is ``False`` or if ``fit`` is called
        for the first time (so that ``self.coef_`` does not exist yet). If
        ``None``, all coefficients are set to zero and the start value for the
        intercept is the weighted average of ``y`` (If ``fit_intercept`` is
        ``True``). If an array, used directly as start values; if
        ``fit_intercept`` is ``True``, its first element is assumed to be the
        start value for the ``intercept_``. Note that
        ``n_features* = X.shape[1] + fit_intercept``, i.e. it includes the
        intercept.

    selection : str, optional (default='cyclic')
        For the CD solver 'cd', the coordinates (features) can be updated in
        either cyclic or random order. If set to ``'random'``, a random
        coefficient is updated every iteration rather than looping over features
        sequentially in the same order, which often leads to significantly
        faster convergence, especially when ``gradient_tol`` is higher than
        ``1e-4``.

    random_state : int or RandomState, optional (default=None)
        The seed of the pseudo random number generator that selects a random
        feature to be updated for the CD solver. If an integer, ``random_state``
        is the seed used by the random number generator; if a
        :class:`RandomState` instance, ``random_state`` is the random number
        generator; if ``None``, the random number generator is the
        :class:`RandomState` instance used by ``np.random``. Used when
        ``selection`` is ``'random'``.

    copy_X : bool, optional (default=None)
        Whether to copy ``X``. Since ``X`` is never modified by
        :class:`GeneralizedLinearRegressor`, this is unlikely to be needed; this
        option exists mainly for compatibility with other scikit-learn
        estimators. If ``False``, ``X`` will not be copied and there will be an
        error if you pass an ``X`` in the wrong format, such as providing
        integer ``X`` and float ``y``. If ``None``, ``X`` will not be copied
        unless it is in the wrong format.

    check_input : bool, optional (default=True)
        Whether to bypass several checks on input: ``y`` values in range of
        ``family``, ``sample_weight`` non-negative, ``P2`` positive
        semi-definite. Don't use this parameter unless you know what you are
        doing.

    verbose : int, optional (default=0)
        For the IRLS solver, any positive number will result in a pretty
        progress bar showing convergence. This features requires having the
        tqdm package installed. For the L-BFGS and ``'trust-constr'`` solvers,
        set ``verbose`` to any positive number for verbosity.

    scale_predictors: bool, optional (default=False)
        If ``True``, scale all predictors to have standard deviation one.
        Should be set to ``True`` if ``alpha > 0`` and if you want coefficients
        to be penalized equally.

        Reported coefficient estimates are always at the original scale.

        Advanced developer note: Internally, predictors are always rescaled for
        computational reasons, but this only affects results if
        ``scale_predictors`` is ``True``.

    lower_bounds : array-like, shape (n_features,), optional (default=None)
        Set a lower bound for the coefficients. Setting bounds forces the use
        of the coordinate descent solver (``'irls-cd'``).

    upper_bounds : array-like, shape=(n_features,), optional (default=None)
        See ``lower_bounds``.

    A_ineq : array-like, shape=(n_constraints, n_features), optional (default=None)
        Constraint matrix for linear inequality constraints of the form
        ``A_ineq w <= b_ineq``. Setting inequality constraints forces the use
        of the local gradient-based solver ``'trust-constr'``, which may
        increase runtime significantly. Note that the constraints only apply
        to coefficients related to features in ``X``. If you want to constrain
        the intercept, add it to the feature matrix ``X`` manually and set
        ``fit_intercept==False``.

    b_ineq : array-like, shape=(n_constraints,), optional (default=None)
        Constraint vector for linear inequality constraints of the form
        ``A_ineq w <= b_ineq``. Refer to the documentation of ``A_ineq`` for
        details.

    drop_first : bool, optional (default = False)
        If ``True``, drop the first column when encoding categorical variables.
        Set this to True when ``alpha=0`` and ``solver='auto'`` to prevent an error
        due to a singular feature matrix. In the case of using a formula with
        interactions, setting this argument to ``True`` ensures structural
        full-rankness (it is equivalent to ``ensure_full_rank`` in formulaic and
        tabmat).

    robust : bool, optional (default = False)
        If true, then robust standard errors are computed by default.

    expected_information : bool, optional (default = False)
        If true, then the expected information matrix is computed by default.
        Only relevant when computing robust standard errors.

    formula : formulaic.FormulaSpec
        A formula accepted by formulaic. It can either be a one-sided formula, in
        which case ``y`` must be specified in ``fit``, or a two-sided formula, in
        which case ``y`` must be ``None``.

    interaction_separator: str, default=":"
        The separator between the names of interacted variables.

    categorical_format : str, optional, default='{name}[{category}]'
        Format string for categorical features. The format string should
        contain the placeholder ``{name}`` for the feature name and
        ``{category}`` for the category name. Only used if ``X`` is a pandas
        DataFrame.

    cat_missing_method: str {'fail'|'zero'|'convert'}, default='fail'
        How to handle missing values in categorical columns. Only used if ``X``
        is a pandas data frame.
        - if 'fail', raise an error if there are missing values
        - if 'zero', missing values will represent all-zero indicator columns.
        - if 'convert', missing values will be converted to the ``cat_missing_name``
          category.

    cat_missing_name: str, default='(MISSING)'
        Name of the category to which missing values will be converted if
        ``cat_missing_method='convert'``.  Only used if ``X`` is a pandas data frame.

    Attributes
    ----------
    coef_ : numpy.array, shape (n_features,)
        Estimated coefficients for the linear predictor (X*coef_+intercept_) in
        the GLM.

    intercept_ : float
        Intercept (a.k.a. bias) added to linear predictor.

    n_iter_ : int
        Actual number of iterations used in solver.

    col_means_: array, shape (n_features,)
        The means of the columns of the design matrix ``X``.

    col_stds_: array, shape (n_features,)
        The standard deviations of the columns of the design matrix ``X``.

    Notes
    -----
    The fit itself does not need outcomes to be from an EDM, but only assumes
    the first two moments to be
    :math:`\\mu_i \\equiv \\mathrm{E}(y_i) = h(x_i' w)` and
    :math:`\\mathrm{var}(y_i) = (\\phi / s_i) v(\\mu_i)`. The unit
    variance function :math:`v(\\mu_i)` is a property of and given by the
    specific EDM; see :doc:`background<background>`.

    The parameters :math:`w` (``coef_`` and ``intercept_``) are estimated by
    minimizing the deviance plus penalty term, which is equivalent to
    (penalized) maximum likelihood estimation.

    If the target ``y`` is a ratio, appropriate sample weights ``s`` should be
    provided. As an example, consider Poisson distributed counts ``z``
    (integers) and weights ``s = exposure`` (time, money, persons years, ...).
    Then you fit ``y ≡ z/s``, i.e.
    ``GeneralizedLinearModel(family='poisson').fit(X, y, sample_weight=s)``. The
    weights are necessary for the right (finite sample) mean. Consider
    :math:`\\bar{y} = \\sum_i s_i y_i / \\sum_i s_i`: in this case, one might
    say that :math:`y` follows a 'scaled' Poisson distribution. The same holds
    for other distributions.

    References
    ----------
    For the coordinate descent implementation:
        * Guo-Xun Yuan, Chia-Hua Ho, Chih-Jen Lin
          An Improved GLMNET for L1-regularized Logistic Regression,
          Journal of Machine Learning Research 13 (2012) 1999-2030
          https://www.csie.ntu.edu.tw/~cjlin/papers/l1_glmnet/long-glmnet.pdf
    """

    def __init__(
        self,
        *,
        alpha=None,
        l1_ratio=0,
        P1: Optional[Union[str, np.ndarray]] = "identity",
        P2: Optional[Union[str, np.ndarray, sparse.spmatrix]] = "identity",
        fit_intercept=True,
        family: Union[str, ExponentialDispersionModel] = "normal",
        link: Union[str, Link] = "auto",
        solver: str = "auto",
        max_iter=100,
        max_inner_iter=100000,
        gradient_tol: Optional[float] = None,
        step_size_tol: Optional[float] = None,
        hessian_approx: float = 0.0,
        warm_start: bool = False,
        alpha_search: bool = False,
        alphas: Optional[np.ndarray] = None,
        n_alphas: int = 100,
        min_alpha_ratio: Optional[float] = None,
        min_alpha: Optional[float] = None,
        start_params: Optional[np.ndarray] = None,
        selection: str = "cyclic",
        random_state=None,
        copy_X: Optional[bool] = None,
        check_input: bool = True,
        verbose=0,
        scale_predictors: bool = False,
        lower_bounds: Optional[np.ndarray] = None,
        upper_bounds: Optional[np.ndarray] = None,
        A_ineq: Optional[np.ndarray] = None,
        b_ineq: Optional[np.ndarray] = None,
        force_all_finite: bool = True,
        drop_first: bool = False,
        robust: bool = True,
        expected_information: bool = False,
        formula: Optional[FormulaSpec] = None,
        interaction_separator: str = ":",
        categorical_format: str = "{name}[{category}]",
        cat_missing_method: str = "fail",
        cat_missing_name: str = "(MISSING)",
    ):
        self.alphas = alphas
        self.alpha = alpha
        super().__init__(
            l1_ratio=l1_ratio,
            P1=P1,
            P2=P2,
            fit_intercept=fit_intercept,
            family=family,
            link=link,
            solver=solver,
            max_iter=max_iter,
            max_inner_iter=max_inner_iter,
            gradient_tol=gradient_tol,
            step_size_tol=step_size_tol,
            hessian_approx=hessian_approx,
            warm_start=warm_start,
            alpha_search=alpha_search,
            n_alphas=n_alphas,
            min_alpha=min_alpha,
            min_alpha_ratio=min_alpha_ratio,
            start_params=start_params,
            selection=selection,
            random_state=random_state,
            copy_X=copy_X,
            check_input=check_input,
            verbose=verbose,
            scale_predictors=scale_predictors,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            A_ineq=A_ineq,
            b_ineq=b_ineq,
            force_all_finite=force_all_finite,
            drop_first=drop_first,
            robust=robust,
            expected_information=expected_information,
            formula=formula,
            interaction_separator=interaction_separator,
            categorical_format=categorical_format,
            cat_missing_method=cat_missing_method,
            cat_missing_name=cat_missing_name,
        )

    def _validate_hyperparameters(self) -> None:
        if self.alpha_search:
            if not isinstance(self.alpha, Iterable) and self.alpha is not None:
                raise ValueError(
                    "`alpha` should be an Iterable or None when `alpha_search`"
                    " is True"
                )
            if self.alpha is not None and (
                (np.asarray(self.alpha) < 0).any()
                or not np.issubdtype(np.asarray(self.alpha).dtype, np.number)
            ):
                raise ValueError("`alpha` must contain only non-negative numbers")
        if not self.alpha_search:
            if not np.isscalar(self.alpha) and self.alpha is not None:
                raise ValueError(
                    "`alpha` should be a scalar or None when `alpha_search`" " is False"
                )
            if self.alpha is not None and (
                not isinstance(self.alpha, (int, float)) or self.alpha < 0
            ):
                raise ValueError(
                    "Penalty term must be a non-negative number;"
                    f" got (alpha={self.alpha})"  # type: ignore
                )

        if (
            not np.isscalar(self.l1_ratio)
            # check for numeric, i.e. not a string
            or not np.issubdtype(np.asarray(self.l1_ratio).dtype, np.number)
            or self.l1_ratio < 0  # type: ignore
            or self.l1_ratio > 1  # type: ignore
        ):
            raise ValueError(
                "l1_ratio must be a number in interval [0, 1];"
                f" got (l1_ratio={self.l1_ratio})"
            )
        super()._validate_hyperparameters()

    def fit(
        self,
        X: ArrayLike,
        y: Optional[ArrayLike] = None,
        sample_weight: Optional[ArrayLike] = None,
        offset: Optional[ArrayLike] = None,
        *,
        store_covariance_matrix: bool = False,
        clusters: Optional[np.ndarray] = None,
        # TODO: take out weights_sum (or use it properly)
        weights_sum: Optional[float] = None,
        context: Optional[Union[int, Mapping[str, Any]]] = None,
    ):
        """Fit a Generalized Linear Model.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training data. Note that a ``float32`` matrix is acceptable and will
            result in the entire algorithm being run in 32-bit precision.
            However, for problems that are poorly conditioned, this might result
            in poor convergence or flawed parameter estimates. If a Pandas data
            frame is provided, it may contain categorical columns. In that case,
            a separate coefficient will be estimated for each category. No
            category is omitted. This means that some regularization is required
            to fit models with an intercept or models with several categorical
            columns.

        y : array-like, shape (n_samples,)
            Target values.

        sample_weight : array-like, shape (n_samples,), optional (default=None)
            Individual weights w_i for each sample. Note that, for an
            Exponential Dispersion Model (EDM), one has
            :math:`\\mathrm{var}(y_i) = \\phi \\times v(mu) / w_i`.
            If :math:`y_i \\sim EDM(\\mu, \\phi / w_i)`, then
            :math:`\\sum w_i y_i / \\sum w_i \\sim EDM(\\mu, \\phi / \\sum w_i)`,
            i.e. the mean of :math:`y` is a weighted average with weights equal
            to ``sample_weight``.

        offset: array-like, shape (n_samples,), optional (default=None)
            Added to linear predictor. An offset of 3 will increase expected
            ``y`` by 3 if the link is linear and will multiply expected ``y`` by
            3 if the link is logarithmic.

        store_covariance_matrix : bool, optional (default=False)
            Whether to estimate and store the covariance matrix of the parameter
            estimates. If ``True``, the covariance matrix will be available in the
            ``covariance_matrix_`` attribute after fitting.

        clusters : array-like, optional, default=None
            Array with cluster membership. Clustered standard errors are
            computed if clusters is not None.

        context : Optional[Union[int, Mapping[str, Any]]], default=None
            The context to add to the evaluation context of the formula with,
            e.g., custom transforms. If an integer, the context is taken from
            the stack frame of the caller at the given depth. Otherwise, a
            mapping from variable names to values is expected. By default,
            no context is added. Set ``context=0`` to make the calling scope
            available.

        weights_sum: float, optional (default=None)

        Returns
        -------
        self
        """

        self._validate_hyperparameters()

        captured_context = capture_context(
            context + 1 if isinstance(context, int) else context
        )

        # NOTE: This function checks if all the entries in X and y are
        # finite. That can be expensive. But probably worthwhile.
        (
            X,
            y,
            sample_weight,
            offset,
            weights_sum,
            P1,
            P2,
        ) = self._set_up_and_check_fit_args(
            X,
            y,
            sample_weight,
            offset,
            force_all_finite=self.force_all_finite,
            context=captured_context,
        )
        assert isinstance(X, tm.MatrixBase)
        assert isinstance(y, np.ndarray)

        self._set_up_for_fit(y)

        # 1.3 arguments to take special care ##################################
        # P1, P2, start_params
        stype = ["csc"] if self._solver == "irls-cd" else ["csc", "csr"]
        P1_no_alpha = setup_p1(P1, X, X.dtype, 1, self.l1_ratio)
        P2_no_alpha = setup_p2(P2, X, stype, X.dtype, 1, self.l1_ratio)

        lower_bounds = check_bounds(self.lower_bounds, X.shape[1], X.dtype)
        upper_bounds = check_bounds(self.upper_bounds, X.shape[1], X.dtype)

        A_ineq, b_ineq = check_inequality_constraints(
            self.A_ineq, self.b_ineq, n_features=X.shape[1], dtype=X.dtype
        )

        if (lower_bounds is not None) and (upper_bounds is not None):
            if np.any(lower_bounds > upper_bounds):
                raise ValueError("Upper bounds must be higher than lower bounds.")

        # 1.4 additional validations ##########################################
        if self.check_input:
            # check if P2 is positive semidefinite
            if not isinstance(self.P2, str):  # self.P2 != 'identity'
                if not is_pos_semidef(P2_no_alpha):
                    if P2_no_alpha.ndim == 1 or P2_no_alpha.shape[0] == 1:
                        error = "1d array P2 must not have negative values."
                    else:
                        error = "P2 must be positive semi-definite."
                    raise ValueError(error)
            # TODO: if alpha=0 check that X is not rank deficient
            # TODO: what else to check?

        #######################################################################
        # 2c. potentially rescale predictors
        #######################################################################

        (
            X,
            self.col_means_,
            self.col_stds_,
            lower_bounds,
            upper_bounds,
            A_ineq,
            P1_no_alpha,
            P2_no_alpha,
        ) = _standardize(
            X,
            sample_weight,
            self._center_predictors,
            self.scale_predictors,
            lower_bounds,
            upper_bounds,
            A_ineq,
            P1_no_alpha,
            P2_no_alpha,
        )

        #######################################################################
        # 3. initialization of coef = (intercept_, coef_)                     #
        #######################################################################

        coef = self._get_start_coef(
            X,
            y,
            sample_weight,
            offset,
            self.col_means_,
            self.col_stds_,
            dtype=[np.float64, np.float32],
        )

        #######################################################################
        # 4. fit                                                              #
        #######################################################################
        if self.alpha_search:
            if self.alphas is not None:
                warnings.warn(
                    "alphas is deprecated. Use alpha instead.", DeprecationWarning
                )
                self._alphas = self.alphas
            elif self.alpha is None:
                self._alphas = self._get_alpha_path(
                    P1_no_alpha=P1_no_alpha, X=X, y=y, w=sample_weight, offset=offset
                )
            else:
                self._alphas = self.alpha
                if self.min_alpha is not None or self.min_alpha_ratio is not None:
                    warnings.warn(
                        "`alpha` is set. Ignoring `min_alpha` and `min_alpha_ratio`."
                    )

            coef = self._solve_regularization_path(
                X=X,
                y=y,
                sample_weight=sample_weight,
                P2_no_alpha=P2_no_alpha,
                P1_no_alpha=P1_no_alpha,
                alphas=self._alphas,
                coef=coef,
                offset=offset,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
                A_ineq=A_ineq,
                b_ineq=b_ineq,
            )

            # intercept_ and coef_ return the last estimated alpha
            if self.fit_intercept:
                self.intercept_path_, self.coef_path_ = _unstandardize(
                    self.col_means_, self.col_stds_, coef[:, 0], coef[:, 1:]
                )
                self.intercept_ = self.intercept_path_[-1]  # type: ignore
                self.coef_ = self.coef_path_[-1]
            else:
                # set intercept to zero as the other linear models do
                self.intercept_path_, self.coef_path_ = _unstandardize(
                    self.col_means_, self.col_stds_, np.zeros(coef.shape[0]), coef
                )
                self.intercept_ = 0.0
                self.coef_ = self.coef_path_[-1]
        else:
            if self.alpha is None:
                _alpha = 0.0
            else:
                _alpha = self.alpha
            if _alpha > 0 and self.l1_ratio > 0 and self._solver != "irls-cd":
                raise ValueError(
                    f"The chosen solver (solver={self._solver}) can't deal "
                    "with L1 penalties, which are included with "
                    f"(alpha={_alpha}) and (l1_ratio={self.l1_ratio})."
                )
            coef = self._solve(
                X=X,
                y=y,
                sample_weight=sample_weight,
                P2=P2_no_alpha * _alpha,
                P1=P1_no_alpha * _alpha,
                coef=coef,
                offset=offset,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
                A_ineq=A_ineq,
                b_ineq=b_ineq,
            )

            if self.fit_intercept:
                self.intercept_, self.coef_ = _unstandardize(
                    self.col_means_, self.col_stds_, coef[0], coef[1:]
                )
            else:
                # set intercept to zero as the other linear models do
                self.intercept_, self.coef_ = _unstandardize(
                    self.col_means_, self.col_stds_, 0.0, coef
                )

        self.covariance_matrix_ = None
        if store_covariance_matrix:
            self.covariance_matrix(
                X=X.unstandardize(),
                y=y,
                offset=offset,
                sample_weight=sample_weight * weights_sum,
                robust=getattr(self, "robust", True),
                clusters=clusters,
                expected_information=getattr(self, "expected_information", False),
                store_covariance_matrix=True,
                skip_checks=True,
            )

        return self

    def _compute_information_criteria(
        self,
        X: ShapedArrayLike,
        y: ShapedArrayLike,
        sample_weight: Optional[ArrayLike] = None,
        context: Optional[Mapping[str, Any]] = None,
    ):
        """
        Computes and stores the model's degrees of freedom, the 'aic', 'aicc'
        and 'bic' information criteria.

        The model's degrees of freedom are used to calculate the effective
        number of parameters. This uses the claim by [2] and [3] that, for
        L1 regularisation, the number of non-zero parameters in the trained model
        is an unbiased approximator of the degrees of freedom of the model. Note
        that this might not hold true for L2 regularisation and thus we raise a
        warning for this case.

        References
        ----------
        [1] Burnham KP, Anderson KR (2002). Model Selection and Multimodel
        Inference; Springer New York.
        [2] Zou, H., Hastie, T. and Tibshirani, R., (2007). On the “degrees of
        freedom” of the lasso; The Annals of Statistics.
        [3] Park, M.Y., 2006. Generalized linear models with regularization;
        Stanford Universty.
        """
        if not hasattr(self.family_instance, "log_likelihood"):
            raise NotImplementedError(
                "The family instance does not define a `log_likelihood` method, so "
                "information criteria cannot be computed. Compatible families include "
                "the binomial, negative binomial and Tweedie (power<=2 or power=3)."
            )

        ddof = np.sum(np.abs(self.coef_) > np.finfo(self.coef_.dtype).eps)  # type: ignore
        k_params = ddof + self.fit_intercept
        nobs = X.shape[0]

        if nobs != self._num_obs:
            raise ValueError(
                "The same dataset that was used for training should also be used for "
                "the computation of information criteria."
            )

        mu = self.predict(X, context=context)
        ll = self.family_instance.log_likelihood(y, mu, sample_weight=sample_weight)

        aic = -2 * ll + 2 * k_params
        bic = -2 * ll + np.log(nobs) * k_params

        if nobs > k_params + 1:
            aicc = aic + 2 * k_params * (k_params + 1) / (nobs - k_params - 1)
        else:
            aicc = None

        self._info_criteria = {"aic": aic, "aicc": aicc, "bic": bic}

        return True

    def aic(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
        *,
        context: Optional[Union[int, Mapping[str, Any]]] = None,
    ):
        """
        Akaike's information criteria. Computed as:
        :math:`-2\\log\\hat{\\mathcal{L}} + 2\\hat{k}` where
        :math:`\\hat{\\mathcal{L}}` is the maximum likelihood estimate of the
        model, and :math:`\\hat{k}` is the effective number of parameters. See
        `_compute_information_criteria` for more information on the computation
        of :math:`\\hat{k}`.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Same data as used in 'fit'

        y : array-like, shape (n_samples,)
            Same data as used in 'fit'

        sample_weight : array-like, shape (n_samples,), optional (default=None)
            Same data as used in 'fit'

        context : Optional[Union[int, Mapping[str, Any]]], default=None
            The context to add to the evaluation context of the formula with,
            e.g., custom transforms. If an integer, the context is taken from
            the stack frame of the caller at the given depth. Otherwise, a
            mapping from variable names to values is expected. By default,
            no context is added. Set ``context=0`` to make the calling scope
            available.
        """
        captured_context = capture_context(
            context + 1 if isinstance(context, int) else context
        )
        return self._get_info_criteria(
            "aic", X, y, sample_weight, context=captured_context
        )

    def aicc(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
        *,
        context: Optional[Union[int, Mapping[str, Any]]] = None,
    ):
        """
        Second-order Akaike's information criteria (or small sample AIC).
        Computed as:
        :math:`-2\\log\\hat{\\mathcal{L}} + 2\\hat{k} + \\frac{2k(k+1)}{n-k-1}`
        where :math:`\\hat{\\mathcal{L}}` is the maximum likelihood estimate of
        the model, :math:`n` is the number of training instances, and
        :math:`\\hat{k}` is the effective number of parameters. See
        `_compute_information_criteria` for more information on the computation
        of :math:`\\hat{k}`.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Same data as used in 'fit'

        y : array-like, shape (n_samples,)
            Same data as used in 'fit'

        sample_weight : array-like, shape (n_samples,), optional (default=None)
             Same data as used in 'fit'

        context : Optional[Union[int, Mapping[str, Any]]], default=None
            The context to add to the evaluation context of the formula with,
            e.g., custom transforms. If an integer, the context is taken from
            the stack frame of the caller at the given depth. Otherwise, a
            mapping from variable names to values is expected. By default,
            no context is added. Set ``context=0`` to make the calling scope
            available.
        """
        captured_context = capture_context(
            context + 1 if isinstance(context, int) else context
        )
        aicc = self._get_info_criteria(
            "aicc", X, y, sample_weight, context=captured_context
        )
        if not aicc:
            raise ValueError(
                "Model degrees of freedom should be more than training datapoints."
            )
        return aicc

    def bic(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
        *,
        context: Optional[Union[int, Mapping[str, Any]]] = None,
    ):
        """
        Bayesian information criterion. Computed as:
        :math:`-2\\log\\hat{\\mathcal{L}} + k\\log(n)` where
        :math:`\\hat{\\mathcal{L}}` is the maximum likelihood estimate of the
        model, :math:`n` is the number of training instances, and
        :math:`\\hat{k}` is the effective number of parameters. See
        `_compute_information_criteria` for more information on the computation
        of :math:`\\hat{k}`.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Same data as used in 'fit'

        y : array-like, shape (n_samples,)
            Same data as used in 'fit'

        sample_weight : array-like, shape (n_samples,), optional (default=None)
             Same data as used in 'fit'

        context : Optional[Union[int, Mapping[str, Any]]], default=None
            The context to add to the evaluation context of the formula with,
            e.g., custom transforms. If an integer, the context is taken from
            the stack frame of the caller at the given depth. Otherwise, a
            mapping from variable names to values is expected. By default,
            no context is added. Set ``context=0`` to make the calling scope
            available.
        """
        captured_context = capture_context(
            context + 1 if isinstance(context, int) else context
        )
        return self._get_info_criteria(
            "bic", X, y, sample_weight, context=captured_context
        )

    def _get_info_criteria(
        self,
        crit: str,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
        context: Optional[Mapping[str, Any]] = None,
    ):
        check_is_fitted(self, "coef_")

        if not hasattr(self, "_info_criteria"):
            self._compute_information_criteria(X, y, sample_weight, context=context)

        if (
            self.alpha is None or (self.alpha is not None and self.alpha > 0)
        ) and self.l1_ratio < 1.0:
            warnings.warn(
                "There is no general definition for the model's degrees of "
                + f"freedom under L2 (ridge) regularisation. The {crit} "
                + "might not be well defined in these cases."
            )

        return self._info_criteria[crit]
