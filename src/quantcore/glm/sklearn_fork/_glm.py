"""
Generalized Linear Models with Exponential Dispersion Family
"""

# Author: Christian Lorentzen <lorentzen.ch@googlemail.com>
# some parts and tricks stolen from other sklearn files.
# License: BSD 3 clause

# TODO: Add cross validation support, e.g. GCV?
# TODO: Should GeneralizedLinearRegressor inherit from LinearModel?
#       So far, it does not.
# TODO: Include further classes in class.rst? ExponentialDispersionModel?
#       TweedieDistribution?
# TODO: Negative values in P1 are not allowed so far. They could be used
#       for group lasso.

# Design Decisions:
# - Which name? GeneralizedLinearModel vs GeneralizedLinearRegressor.
#   Estimators in sklearn are either regressors or classifiers. A GLM can do
#   both depending on the distr (Normal => regressor, Binomial => classifier).
#   Solution: GeneralizedLinearRegressor since this is the focus.
# - Allow for finer control of penalty terms:
#   L1: ||P1*w||_1 with P1*w as element-wise product, this allows to exclude
#       factors from the L1 penalty.
#   L2: w*P2*w with P2 a positive (semi-) definite matrix, e.g. P2 could be
#   a 1st or 2nd order difference matrix (compare B-spline penalties and
#   Tikhonov regularization).
# - The link function (instance of class Link) is necessary for the evaluation
#   of deviance, score, Hessian matrix as functions of the
#   coefficients, which is needed by optimizers.
#   Solution: link as argument in those functions
# - Which name/symbol for sample_weight in docu?
#   sklearn.linear_models uses w for coefficients, standard literature on
#   GLMs use beta for coefficients and w for (sample) weights.
#   So far, coefficients=w and sample weights=s.
# - The intercept term is the first index, i.e. coef[0]


from __future__ import division

import warnings
from typing import Iterable, List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd
import scipy.sparse.linalg as splinalg
from scipy import linalg, sparse
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_array
from sklearn.utils.validation import (
    _assert_all_finite,
    check_consistent_length,
    check_is_fitted,
    check_random_state,
    check_X_y,
    column_or_1d,
)

from .. import matrix as mx
from ._distribution import (
    BinomialDistribution,
    ExponentialDispersionModel,
    GammaDistribution,
    GeneralizedHyperbolicSecant,
    InverseGaussianDistribution,
    NormalDistribution,
    PoissonDistribution,
    TweedieDistribution,
    guess_intercept,
)
from ._link import IdentityLink, Link, LogitLink, LogLink
from ._solvers import (
    IRLSData,
    _cd_solver,
    _irls_solver,
    _lbfgs_solver,
    _least_squares_solver,
)

_float_itemsize_to_dtype = {8: np.float64, 4: np.float32, 2: np.float16}

ArrayLike = Union[
    pd.DataFrame,
    pd.Series,
    List,
    np.ndarray,
    sparse.spmatrix,
    mx.MatrixBase,
    mx.ColScaledMat,
]
ShapedArrayLike = Union[
    pd.DataFrame,
    pd.Series,
    np.ndarray,
    sparse.spmatrix,
    mx.MatrixBase,
    mx.ColScaledMat,
]


def check_X_y_matrix(
    X: mx.MatrixBase,
    y: Union[np.ndarray, List, sparse.spmatrix],
    *,
    accept_sparse: Union[str, bool, List[str]] = False,
    dtype: Union[str, Type, List[Type], None] = "numeric",
    order: Optional[str] = None,
    copy: bool = False,
    ensure_min_samples: int = 1,
    estimator: Optional[str] = None,
) -> Tuple[Union[mx.MatrixBase, sparse.spmatrix, np.ndarray], np.ndarray]:
    """
    See documentation for sklearn.utils.check_X_y. This function behaves identically
    for inputs that are not from the Matrix package, and has some parameters,
    such as "force_all_finite", fixed to match the needs of GLMs..

    Returns
    -------
    X_converted : object
        The converted and validated X.
    y_converted : object
        The converted and validated y.
    """
    assert isinstance(X, mx.MatrixBase)

    def _check_array(mat, ensure_min_features: int):

        return check_array(
            mat,
            accept_sparse=accept_sparse,
            accept_large_sparse=True,
            dtype=dtype,
            order=order,
            copy=copy,
            force_all_finite=True,
            ensure_2d=True,
            allow_nd=False,
            ensure_min_samples=ensure_min_samples,
            ensure_min_features=ensure_min_features,
            estimator=estimator,
        )

    if y is None:
        raise ValueError("y cannot be None")

    y = column_or_1d(y, warn=True)
    _assert_all_finite(y)
    if y.dtype.kind == "O":
        y = y.astype(np.float64)

    check_consistent_length(X, y)

    if isinstance(X, mx.SplitMatrix):
        X.X_sparse = _check_array(X.X_sparse, ensure_min_features=0)
        X.X_dense_F = mx.DenseGLMDataMatrix(
            _check_array(X.X_dense_F, ensure_min_features=0)
        )

    else:
        original_type = type(X)
        X = _check_array(X, ensure_min_features=1)
        if original_type is mx.DenseGLMDataMatrix:
            X = mx.DenseGLMDataMatrix(X)

    return X, y


def _check_weights(
    sample_weight: Union[float, np.ndarray, None], n_samples: int, dtype
) -> np.ndarray:
    """Check that sample weights are non-negative and have the right shape."""
    if sample_weight is None:
        weights = np.ones(n_samples, dtype=dtype)
    elif np.isscalar(sample_weight):
        if sample_weight <= 0:
            raise ValueError("Sample weights must be non-negative.")
        weights = sample_weight * np.ones(n_samples, dtype=dtype)
    else:
        _dtype = [np.float64, np.float32]
        weights = check_array(
            sample_weight,
            accept_sparse=False,
            force_all_finite=True,
            ensure_2d=False,
            dtype=_dtype,
        )
        if weights.ndim > 1:
            raise ValueError("Sample weight must be 1D array or scalar")
        elif weights.shape[0] != n_samples:
            raise ValueError("Sample weights must have the same length as y")
        if not np.all(weights >= 0):
            raise ValueError("Sample weights must be non-negative.")
        elif not np.sum(weights) > 0:
            raise ValueError(
                "Sample weights must have at least one positive " "element."
            )

    return weights


def _check_offset(
    offset: Union[np.ndarray, float, None], n_rows: int, dtype
) -> Optional[np.ndarray]:
    """
    Unlike weights, if the offset is given as None, it can stay None. So we only need
    to validate it when it is not none.
    """
    if offset is None:
        return None
    if not np.isscalar(offset):
        offset = check_array(
            offset,
            accept_sparse=False,
            force_all_finite=True,
            ensure_2d=False,
            dtype=dtype,
        )
        if offset.ndim > 1:
            raise ValueError("Offset must be 1D array or scalar.")
        elif offset.shape[0] != n_rows:
            raise ValueError("offset must have the same length as y.")
    return np.full(n_rows, offset)


def check_bounds(
    bounds: Union[Iterable, float, np.ndarray, None], n_features: int, dtype
) -> Union[None, np.ndarray]:
    """Check that the bounds have the right shape."""
    if bounds is None:
        return None
    if np.isscalar(bounds):
        bounds = np.full(n_features, bounds, dtype=dtype)
    else:  # assume it's an array
        bounds = check_array(
            bounds,
            accept_sparse=False,
            force_all_finite=True,
            ensure_2d=False,
            dtype=dtype,
        )
        if bounds.ndim > 1:
            raise ValueError("Bounds must be 1D array or scalar.")
        if bounds.shape[0] != n_features:
            raise ValueError("Bounds must be the same length as X.shape[1].")
    return bounds


def _unstandardize(
    X: mx.ColScaledMat,
    col_means: np.ndarray,
    col_stds: Optional[np.ndarray],
    intercept: float,
    coef: np.ndarray,
) -> Tuple[mx.MatrixBase, float, np.ndarray]:
    assert isinstance(X, mx.ColScaledMat)
    X_mat: mx.MatrixBase = X.unstandardize(col_stds)
    if col_stds is None:
        intercept -= np.squeeze(np.squeeze(col_means).dot(np.atleast_1d(coef).T))
        # intercept -= float(np.squeeze(col_means).dot(coef))
    else:
        intercept -= np.squeeze(
            np.squeeze(col_means / col_stds).dot(np.atleast_1d(coef).T)
        )
        # intercept -= float(np.squeeze(col_means / col_stds).dot(coef))
        coef /= col_stds
    return X_mat, intercept, coef


def _standardize_warm_start(
    coef: np.ndarray, col_means: np.ndarray, col_stds: Optional[np.ndarray]
) -> None:
    if col_stds is None:
        coef[0] += np.squeeze(col_means).dot(coef[1:])
    else:
        coef[1:] *= col_stds
        coef[0] += np.squeeze(col_means / col_stds).dot(coef[1:])


def get_family(
    family: Union[str, ExponentialDispersionModel]
) -> ExponentialDispersionModel:
    if isinstance(family, ExponentialDispersionModel):
        return family
    name_to_dist = {
        "normal": NormalDistribution,
        "poisson": PoissonDistribution,
        "gamma": GammaDistribution,
        "inverse.gaussian": InverseGaussianDistribution,
        "binomial": BinomialDistribution,
    }
    try:
        return name_to_dist[family]()
    except KeyError:
        raise ValueError(
            "The family must be an instance of class"
            " ExponentialDispersionModel or an element of"
            " ['normal', 'poisson', 'gamma', 'inverse.gaussian', "
            "'binomial']; got (family={})".format(family)
        )


def get_link(link: Union[str, Link], family: ExponentialDispersionModel) -> Link:
    """
    For the Tweedie distribution, this code follows actuarial best practices regarding
    link functions. Note that these links are sometimes non-canonical:
        - Identity for normal (p=0)
        - No convention for p < 0, so let's leave it as identity
        - Log otherwise
    """
    if isinstance(link, Link):
        return link
    if link == "auto":
        if isinstance(family, TweedieDistribution):
            if family.power <= 0:
                return IdentityLink()
            if family.power < 1:
                # TODO: move more detailed error here
                raise ValueError("No distribution")
            return LogLink()
        if isinstance(family, GeneralizedHyperbolicSecant):
            return IdentityLink()
        if isinstance(family, BinomialDistribution):
            return LogitLink()
        raise ValueError(
            """No default link known for the specified distribution family. Please
            set link manually, i.e. not to 'auto';
            got (link='auto', family={})""".format(
                family.__class__.__name__
            )
        )
    if link == "identity":
        return IdentityLink()
    if link == "log":
        return LogLink()
    if link == "logit":
        return LogitLink()
    raise ValueError(
        """The link must be an instance of class Link or an element of
        ['auto', 'identity', 'log', 'logit']; got (link={})""".format(
            link
        )
    )


def setup_p1(
    P1: Union[str, np.ndarray],
    X: Union[mx.MatrixBase, mx.ColScaledMat],
    _dtype,
    alpha: float,
    l1_ratio: float,
) -> np.ndarray:
    n_features = X.shape[1]
    assert isinstance(X, (mx.MatrixBase, mx.ColScaledMat))
    if isinstance(P1, str) and P1 == "identity":
        P1 = np.ones(n_features, dtype=_dtype)
    else:
        P1 = np.atleast_1d(P1)
        try:
            P1 = P1.astype(_dtype, casting="safe", copy=False)
        except TypeError:
            raise TypeError(
                "The given P1 cannot be converted to a numeric"
                "array; got (P1.dtype={}).".format(P1.dtype)
            )
        if (P1.ndim != 1) or (P1.shape[0] != n_features):
            raise ValueError(
                "P1 must be either 'identity' or a 1d array "
                "with the length of X.shape[1]; "
                "got (P1.shape[0]={}), "
                "needed (X.shape[1]={}).".format(P1.shape[0], n_features)
            )

    # P1 and P2 are now for sure copies
    P1 = alpha * l1_ratio * P1
    return P1.astype(_dtype)


def setup_p2(
    P2: Union[str, np.ndarray],
    X: Union[mx.MatrixBase, mx.ColScaledMat],
    _stype,
    _dtype,
    alpha: float,
    l1_ratio: float,
) -> Union[np.ndarray, sparse.spmatrix]:
    assert isinstance(X, (mx.MatrixBase, mx.ColScaledMat))
    n_features = X.shape[1]

    # If X is sparse, make P2 sparse, too.
    if isinstance(P2, str) and P2 == "identity":
        if sparse.issparse(X):
            P2 = (
                sparse.dia_matrix(
                    (np.ones(n_features, dtype=_dtype), 0),
                    shape=(n_features, n_features),
                )
            ).tocsc()
        else:
            P2 = np.ones(n_features, dtype=_dtype)
    else:
        P2 = check_array(
            P2, copy=True, accept_sparse=_stype, dtype=_dtype, ensure_2d=False
        )
        if P2.ndim == 1:
            P2 = np.asarray(P2)
            if P2.shape[0] != n_features:
                raise ValueError(
                    "P2 should be a 1d array of shape "
                    "(n_features,) with "
                    "n_features=X.shape[1]; "
                    "got (P2.shape=({},)), needed ({},)".format(P2.shape[0], X.shape[1])
                )
            if sparse.issparse(X):
                P2 = (
                    sparse.dia_matrix((P2, 0), shape=(n_features, n_features))
                ).tocsc()
        elif P2.ndim == 2 and P2.shape[0] == P2.shape[1] and P2.shape[0] == X.shape[1]:
            if sparse.issparse(X):
                P2 = sparse.csc_matrix(P2)
        else:
            raise ValueError(
                "P2 must be either None or an array of shape "
                "(n_features, n_features) with "
                "n_features=X.shape[1]; "
                "got (P2.shape=({0}, {1})), needed ({2}, {2})".format(
                    P2.shape[0], P2.shape[1], X.shape[1]
                )
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


def initialize_start_params(
    start_params: Optional[np.ndarray], n_cols: int, fit_intercept: bool, _dtype
) -> Optional[np.ndarray]:
    if start_params is not None:
        start_params = check_array(
            start_params,
            accept_sparse=False,
            force_all_finite=True,
            ensure_2d=False,
            dtype=_dtype,
            copy=True,
        )
        if (start_params.shape[0] != n_cols + fit_intercept) or (
            start_params.ndim != 1
        ):
            raise ValueError(
                "Start values for parameters must have the"
                "right length and dimension; required (length"
                "={}, ndim=1); got (length={}, ndim={}).".format(
                    n_cols + fit_intercept, start_params.shape[0], start_params.ndim,
                )
            )
    return start_params


def is_pos_semidef(p: Union[sparse.spmatrix, np.ndarray]) -> bool:
    """
    Checks for positive semidefiniteness of p if p is a matrix, or diag(p) if p is a
    vector.

    np.linalg.cholesky(P2) 'only' asserts positive definite due to numerical precision,
    we allow eigenvalues to be a tiny bit negative
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
    else:
        # dense
        eigenvalues = linalg.eigvalsh(p)
    pos_semidef = np.all(eigenvalues >= epsneg)
    return pos_semidef


# TODO: abc
class GeneralizedLinearRegressorBase(BaseEstimator, RegressorMixin):
    """
    Base class for GeneralizedLinearRegressor and GeneralizedLinearRegressorCV.
    """

    def __init__(
        self,
        l1_ratio: Union[int, float] = 0,
        P1="identity",
        P2: Union[int, float, str, np.ndarray, sparse.spmatrix] = "identity",
        fit_intercept=True,
        family: Union[str, ExponentialDispersionModel] = "normal",
        link: Union[str, Link] = "auto",
        fit_dispersion=None,
        solver="auto",
        max_iter=100,
        gradient_tol: Optional[float] = 1e-4,
        step_size_tol: Optional[float] = None,
        warm_start=False,
        alpha_search: bool = False,
        n_alphas: int = 100,
        alphas: Optional[np.ndarray] = None,
        min_alpha_ratio: Optional[float] = None,
        min_alpha: Optional[float] = None,
        start_params: Optional[np.ndarray] = None,
        selection="cyclic",
        random_state=None,
        copy_X=True,
        check_input=True,
        verbose=0,
        scale_predictors=False,
        lower_bounds: Optional[np.ndarray] = None,
        upper_bounds: Optional[np.ndarray] = None,
    ):
        self.l1_ratio = l1_ratio
        self.P1 = P1
        self.P2 = P2
        self.fit_intercept = fit_intercept
        self.family = family
        self.link = link
        self.fit_dispersion = fit_dispersion
        self.solver = solver
        self.max_iter = max_iter
        self.gradient_tol = gradient_tol
        self.step_size_tol = step_size_tol
        self.warm_start = warm_start
        self.alpha_search = alpha_search
        self.n_alphas = n_alphas
        self.alphas = alphas
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

    def get_start_coef(
        self,
        start_params,
        X: Union[mx.MatrixBase, mx.ColScaledMat],
        y: np.ndarray,
        weights: np.ndarray,
        offset: Optional[np.ndarray],
        col_means: Optional[np.ndarray],
        col_stds: Optional[np.ndarray],
    ) -> np.ndarray:
        if self.warm_start and hasattr(self, "coef_"):
            coef = self.coef_  # type: ignore
            intercept = self.intercept_  # type: ignore
            if self.fit_intercept:
                coef = np.concatenate((np.array([intercept]), coef))
            if self._center_predictors:
                _standardize_warm_start(coef, col_means, col_stds)
        elif start_params is None:
            if self.fit_intercept:
                coef = np.zeros(
                    X.shape[1] + 1, dtype=_float_itemsize_to_dtype[X.dtype.itemsize]
                )
                coef[0] = guess_intercept(
                    y, weights, self._link_instance, self._family_instance, offset
                )
            else:
                coef = np.zeros(
                    X.shape[1], dtype=_float_itemsize_to_dtype[X.dtype.itemsize]
                )
        else:  # assign given array as start values
            coef = start_params
            if self._center_predictors:
                _standardize_warm_start(coef, col_means, col_stds)

        # If starting values are outside the specified bounds (if set),
        # bring the starting value exactly at the bound.
        idx = 1 if self.fit_intercept else 0
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

    def set_up_for_fit(self, y: np.ndarray) -> None:
        #######################################################################
        # 1. input validation                                                 #
        #######################################################################
        # 1.1
        self._validate_hyperparameters()
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

        if self.solver == "auto":
            if (
                (self.l1_ratio == 0)
                and (self.lower_bounds is None)
                and (self.upper_bounds is None)
            ):
                self._solver = "irls-ls"
            else:
                self._solver = "irls-cd"
        else:
            self._solver = self.solver

        self._random_state = check_random_state(self.random_state)

        # 1.4 additional validations ##########################################
        if self.check_input:
            if not np.all(self._family_instance.in_y_range(y)):
                raise ValueError(
                    "Some value(s) of y are out of the valid "
                    "range for family {}".format(
                        self._family_instance.__class__.__name__
                    )
                )

    def tear_down_from_fit(
        self,
        X: Union[mx.MatrixBase, mx.ColScaledMat],
        y: np.ndarray,
        col_means: Optional[np.ndarray],
        col_stds: Optional[np.ndarray],
        weights: np.ndarray,
        weights_sum: Optional[float],
    ):
        """
        Delete attributes that were only needed for the fit method.
        """
        #######################################################################
        # 5a. undo standardization
        #######################################################################
        if self._center_predictors:
            assert isinstance(X, mx.ColScaledMat)
            X, self.intercept_, self.coef_ = _unstandardize(
                X, col_means, col_stds, self.intercept_, self.coef_,  # type: ignore
            )

        if self.fit_dispersion in ["chisqr", "deviance"]:
            # attention because of rescaling of weights
            self.dispersion_ = self.estimate_phi(X, y, weights) * weights_sum

        del self._center_predictors
        del self._solver
        del self._random_state
        return X

    def _get_alpha_path(
        self,
        l1_ratio: float,
        X,
        y: np.ndarray,
        w: np.ndarray,
        offset: np.ndarray = None,
    ) -> np.ndarray:
        """
        If l1_ratio is positive, the highest alpha is the lowest alpha such that no
        coefficients (other than the intercept) are nonzero.

        If l1_ratio is zero, use the sklearn RidgeCV default path [10, 1, 0.1] or
        whatever is specified by the input parameters min_alpha_ratio and n_alphas..

        min_alpha_ratio governs the length of the path, with 1e-6 as the default.
        Smaller values will lead to a longer path.
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
                min_alpha = self.min_alpha
            return np.logspace(
                np.log(max_alpha), np.log(min_alpha), self.n_alphas, base=np.e
            )

        def _get_normal_identity_grad_at_zeros_with_optimal_intercept() -> np.ndarray:
            if self.fit_intercept:
                if offset is None:
                    mu = y.dot(w)
                else:
                    mu = offset + (y - offset).dot(w)
            else:
                mu = 0
            return X.T.dot(w * (y - mu))

        def _get_tweedie_log_grad_at_zeros_with_optimal_intercept() -> np.ndarray:
            if self.fit_intercept:
                # if all non-intercept coefficients are zero and there is no offset,
                # the best intercept makes the predicted mean the sample mean
                mu = y.dot(w)
            elif offset is not None:
                mu = offset
            else:
                mu = 1

            family = get_family(self.family)
            if isinstance(family, TweedieDistribution):
                p = family.power
            else:
                p = 0

            # tweedie grad
            return mu ** (1 - p) * X.T.dot(w * (y - mu))

        if l1_ratio == 0:
            alpha_max = 10
            return _make_grid(alpha_max)

        if self.fit_intercept:
            coef = np.zeros(X.shape[1] + 1)
            coef[0] = guess_intercept(
                y=y,
                weights=w,
                link=self._link_instance,
                distribution=self._family_instance,
            )
        else:
            coef = np.zeros(X.shape[1])

        _, dev_der = self._family_instance._mu_deviance_derivative(
            coef=coef, X=X, y=y, weights=w, link=self._link_instance, offset=offset,
        )

        alpha_max = np.max(np.abs(-0.5 * dev_der)) / l1_ratio
        return _make_grid(alpha_max)

    def solve(
        self,
        X: Union[mx.MatrixBase, mx.ColScaledMat],
        y: np.ndarray,
        weights: np.ndarray,
        P2,
        P1: np.ndarray,
        coef: np.ndarray,
        offset: Optional[np.ndarray],
        lower_bounds: Optional[np.ndarray],
        upper_bounds: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Must be run after running set_up_for_fit and before running tear_down_from_fit.
        Sets self.coef_ and self.intercept_.
        """
        fixed_inner_tol = None
        if (
            isinstance(self._family_instance, NormalDistribution)
            and isinstance(self._link_instance, IdentityLink)
            and "irls" in self._solver
        ):
            # IRLS-CD and IRLS-LS should converge in one iteration for any
            # normal distribution problem with identity link.
            fixed_inner_tol = (self.gradient_tol, self.step_size_tol)
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
                weights=weights,
                P1=P1,
                P2=P2,
                fit_intercept=self.fit_intercept,
                family=self._family_instance,
                link=self._link_instance,
                max_iter=max_iter,
                gradient_tol=self.gradient_tol,
                step_size_tol=self.step_size_tol,
                fixed_inner_tol=fixed_inner_tol,
                selection=self.selection,
                random_state=self.random_state,
                offset=offset,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
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
                weights=weights,
                P2=P2,
                verbose=self.verbose,
                family=self._family_instance,
                link=self._link_instance,
                max_iter=max_iter,
                # TODO: support step_size_tol?
                tol=self.gradient_tol,  # type: ignore
                offset=offset,
            )
        return coef

    def solve_regularization_path(
        self,
        X: Union[mx.MatrixBase, mx.ColScaledMat],
        y: np.ndarray,
        weights: np.ndarray,
        alphas: np.ndarray,
        P2_no_alpha,
        P1_no_alpha: np.ndarray,
        coef: np.ndarray,
        offset: Optional[np.ndarray],
        lower_bounds: Optional[np.ndarray],
        upper_bounds: Optional[np.ndarray],
    ) -> np.ndarray:

        self.coef_path_ = np.empty((len(alphas), len(coef)))

        for k, alpha in enumerate(alphas):
            P1 = P1_no_alpha * alpha
            P2 = P2_no_alpha * alpha

            coef = self.solve(
                X=X,
                y=y,
                weights=weights,
                P2=P2,
                P1=P1,
                coef=coef,
                offset=offset,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
            )

            self.coef_path_[k, :] = coef

        return self.coef_path_

    def report_diagnostics(self) -> None:
        if hasattr(self, "diagnostics_"):
            print("diagnostics:")
            import pandas as pd

            with pd.option_context("max_rows", None):
                print(
                    pd.DataFrame(data=self.diagnostics_).set_index("n_iter", drop=True)
                )
        else:
            print("solver does not report diagnostics")

    def linear_predictor(self, X: ArrayLike, offset: Optional[ArrayLike] = None):
        """Compute the linear_predictor = X*coef_ + intercept_.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Samples.

        Returns
        -------
        C : array, shape (n_samples,)
            Returns predicted values of linear predictor.
        """
        check_is_fitted(self, "coef_")
        X = check_array(
            X,
            accept_sparse=["csr", "csc", "coo"],
            dtype="numeric",
            copy=True,
            ensure_2d=True,
            allow_nd=False,
        )
        xb = X @ self.coef_ + self.intercept_
        if offset is None:
            return xb
        return xb + offset

    def predict(
        self,
        X: ShapedArrayLike,
        sample_weight: Optional[ArrayLike] = None,
        offset: Optional[ArrayLike] = None,
    ):
        """Predict using GLM with feature matrix X.

        If sample_weight is given, returns prediction*sample_weight.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Samples.

        sample_weight : {None, array-like}, shape (n_samples,), optional \
                (default=None)

        offset: {None, array-like}, shape (n_samples,), optional \
                (default=None)

        Returns
        -------
        C : array, shape (n_samples,)
            Returns predicted values times sample_weight.
        """
        # TODO: Is copy=True necessary?
        X = check_array(
            X,
            accept_sparse=["csr", "csc", "coo"],
            dtype="numeric",
            copy=True,
            ensure_2d=True,
            allow_nd=False,
        )
        eta = self.linear_predictor(X, offset=offset)
        mu = get_link(self.link, get_family(self.family)).inverse(eta)
        weights = _check_weights(sample_weight, X.shape[0], X.dtype)

        return mu * weights

    def estimate_phi(
        self, X: ArrayLike, y: ArrayLike, sample_weight: Optional[ArrayLike] = None
    ):
        """Estimate/fit the dispersion parameter phi.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training data.

        y : array-like, shape (n_samples,)
            Target values.

        sample_weight : {None, array-like}, shape (n_samples,), optional \
                (default=None)
            Sample weights.

        Returns
        -------
        phi : float
            Dispersion parameter.
        """
        check_is_fitted(self, "coef_")
        _dtype = [np.float64, np.float32]
        if isinstance(X, mx.MatrixBase):
            X, y = check_X_y_matrix(
                X, y, accept_sparse=["csr", "csc", "coo"], dtype=_dtype
            )
        else:
            X, y = check_X_y(X, y, accept_sparse=["csr", "csc", "coo"], dtype=_dtype)

        n_samples, n_features = X.shape
        weights = _check_weights(sample_weight, n_samples, X.dtype)
        eta = X @ self.coef_
        if self.fit_intercept is True:
            eta += self.intercept_
            n_features += 1
        if n_samples <= n_features:
            raise ValueError(
                "Estimation of dispersion parameter phi requires"
                " more samples than features, got"
                " samples=X.shape[0]={} and"
                " n_features=X.shape[1]+fit_intercept={}.".format(n_samples, n_features)
            )
        mu = self._link_instance.inverse(eta)
        if self.fit_dispersion == "chisqr":
            chisq = np.sum(
                weights * (y - mu) ** 2 / self._family_instance.unit_variance(mu)
            )
            return float(chisq) / (n_samples - n_features)
        elif self.fit_dispersion == "deviance":
            dev = self._family_instance.deviance(y, mu, weights)
            return float(dev) / (n_samples - n_features)

    # Note: check_estimator(GeneralizedLinearRegressor) might raise
    # "AssertionError: -0.28014056555724598 not greater than 0.5"
    # unless GeneralizedLinearRegressor has a score which passes the test.
    def score(
        self,
        X: ShapedArrayLike,
        y: ShapedArrayLike,
        sample_weight: Optional[ArrayLike] = None,
    ):
        """Compute D^2, the percentage of deviance explained.

        D^2 is a generalization of the coefficient of determination R^2.
        R^2 uses squared error and D^2 deviance. Note that those two are equal
        for family='normal'.

        D^2 is defined as
        :math:`D^2 = 1-\\frac{D(y_{true},y_{pred})}{D_{null}}`,
        :math:`D_{null}` is the null deviance, i.e. the deviance of a model
        with intercept alone, which corresponds to :math:`y_{pred} = \\bar{y}`.
        The mean :math:`\\bar{y}` is averaged by sample_weight.
        Best possible score is 1.0 and it can be negative (because the model
        can be arbitrarily worse).

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Test samples.

        y : array-like, shape (n_samples,)
            True values of target.

        sample_weight : {None, array-like}, shape (n_samples,), optional \
                (default=None)
            Sample weights.

        Returns
        -------
        score : float
            D^2 of self.predict(X) w.r.t. y.
        """
        # Note, default score defined in RegressorMixin is R^2 score.
        # TODO: make D^2 a score function in module metrics (and thereby get
        #       input validation and so on)
        weights = _check_weights(sample_weight, y.shape[0], X.dtype)
        mu = self.predict(X)
        family = get_family(self.family)
        dev = family.deviance(y, mu, weights=weights)
        y_mean = np.average(y, weights=weights)
        dev_null = family.deviance(y, y_mean, weights=weights)
        return 1.0 - dev / dev_null

    def _validate_hyperparameters(self) -> None:

        if self.gradient_tol is None and self.step_size_tol is None:
            raise ValueError("gradient_tol and step_size_tol cannot both be None.")

        if self.gradient_tol is None and self.solver == "lbfgs":
            raise ValueError(
                """lbfgs solver uses only a gradient-based convergence criterion, so
                gradient_tol must not be None.
                """
            )

        if not isinstance(self.fit_intercept, bool):
            raise ValueError(
                "The argument fit_intercept must be bool;"
                " got {}".format(self.fit_intercept)
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

        if self.solver not in ["auto", "irls-ls", "lbfgs", "irls-cd"]:
            raise ValueError(
                "GeneralizedLinearRegressor supports only solvers"
                " 'auto', 'irls-ls', 'lbfgs', and 'irls-cd';"
                " got {}".format(self.solver)
            )
        if not isinstance(self.max_iter, int) or self.max_iter <= 0:
            raise ValueError(
                "Maximum number of iteration must be a positive "
                "integer;"
                " got (max_iter={!r})".format(self.max_iter)
            )

        if self.gradient_tol is not None and (
            not (
                isinstance(self.gradient_tol, float)
                or isinstance(self.gradient_tol, int)
            )
            or self.gradient_tol <= 0
        ):
            raise ValueError(
                "Tolerance for stopping criteria must be "
                "positive; got (tol={!r})".format(self.gradient_tol)
            )

        if self.step_size_tol is not None and (
            not (
                isinstance(self.step_size_tol, float)
                or isinstance(self.step_size_tol, int)
            )
            or self.step_size_tol <= 0
        ):
            raise ValueError(
                "Tolerance for stopping criteria must be "
                "positive; got (tol={!r})".format(self.step_size_tol)
            )

        if not isinstance(self.warm_start, bool):
            raise ValueError(
                "The argument warm_start must be bool;"
                " got {}".format(self.warm_start)
            )
        if self.selection not in ["cyclic", "random"]:
            raise ValueError(
                "The argument selection must be 'cyclic' or "
                "'random'; got (selection={})".format(self.selection)
            )
        if not isinstance(self.copy_X, bool):
            raise ValueError(
                "The argument copy_X must be bool;" " got {}".format(self.copy_X)
            )
        if not isinstance(self.check_input, bool):
            raise ValueError(
                "The argument check_input must be bool; got "
                "(check_input={})".format(self.check_input)
            )
        if self.scale_predictors and not self.fit_intercept:
            raise ValueError(
                "scale_predictors=True is not supported when fit_intercept=False"
            )
        if ((self.lower_bounds is not None) or (self.upper_bounds is not None)) and (
            self.solver not in ["irls-cd", "auto"]
        ):
            raise ValueError(
                "Only the 'cd' solver is supported when bounds are set; "
                "got {}".format(self.solver)
            )
        if self.check_input:
            # check if P1 has only non-negative values, negative values might
            # indicate group lasso in the future.
            if not isinstance(self.P1, str):  # if self.P1 != 'identity':
                if not np.all(self.P1 >= 0):
                    raise ValueError("P1 must not have negative values.")

    def set_up_and_check_fit_args(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: Union[np.ndarray, None],
        offset: Union[np.ndarray, None],
        solver: str,
        copy_X: bool,
    ) -> Tuple[
        mx.MatrixBase, np.ndarray, np.ndarray, Union[np.ndarray, None], float,
    ]:
        _dtype = [np.float64, np.float32]
        if solver == "irls-cd":
            _stype = ["csc"]
        else:
            _stype = ["csc", "csr"]

        if hasattr(X, "dtype") and X.dtype == np.int64:  # type: ignore
            # check_X_y will convert to float32 if we don't do this, which causes
            # precision issues with the new handling of single precision. The new
            # behavior is to give everything the precision of X, but we don't want to
            # do that if X was intially int64.
            X = X.astype(np.float64)  # type: ignore

        if isinstance(X, mx.MatrixBase):
            X, y = check_X_y_matrix(
                X, y, accept_sparse=_stype, dtype=_dtype, copy=copy_X
            )
            self._check_n_features(X, reset=True)
        else:
            X, y = self._validate_data(
                X, y, ensure_2d=True, accept_sparse=_stype, dtype=_dtype, copy=copy_X
            )

        # Without converting y to float, deviance might raise
        # ValueError: Integers to negative integer powers are not allowed.
        # Also, y must not be sparse.
        # Make sure everything has the same precision as X
        # This will prevent accidental upcasting later and slow operations on
        # mixed-precision numbers
        y = np.asarray(y, dtype=X.dtype)
        weights = _check_weights(sample_weight, y.shape[0], X.dtype)
        offset = _check_offset(offset, y.shape[0], X.dtype)

        # IMPORTANT NOTE: Since we want to minimize
        # 1/(2*sum(sample_weight)) * deviance + L1 + L2,
        # deviance = sum(sample_weight * unit_deviance),
        # we rescale weights such that sum(weights) = 1 and this becomes
        # 1/2*deviance + L1 + L2 with deviance=sum(weights * unit_deviance)
        weights_sum: float = np.sum(weights)
        weights /= weights_sum
        #######################################################################
        # 2b. convert to wrapper matrix types
        #######################################################################
        if sparse.issparse(X):
            X = mx.MKLSparseMatrix(X)
        elif isinstance(X, np.ndarray):
            X = mx.DenseGLMDataMatrix(X)

        return X, y, weights, offset, weights_sum


class GeneralizedLinearRegressor(GeneralizedLinearRegressorBase):
    """Regression via a Generalized Linear Model (GLM) with penalties.

    GLMs based on a reproductive Exponential Dispersion Model (EDM) aim at
    fitting and predicting the mean of the target y as mu=h(X*w). Therefore,
    the fit minimizes the following objective function with combined L1 and L2
    priors as regularizer::

            1/(2*sum(s)) * deviance(y, h(X*w); s)
            + alpha * l1_ratio * ||P1*w||_1
            + 1/2 * alpha * (1 - l1_ratio) * w*P2*w

    with inverse link function h and s=sample_weight. Note that for
    ``sample_weight=None``, one has s_i=1 and sum(s)=n_samples).
    For ``P1=P2='identity'``, the penalty is the elastic net::

            alpha * l1_ratio * ||w||_1
            + 1/2 * alpha * (1 - l1_ratio) * ||w||_2^2

    If you are interested in controlling the L1 and L2 penalties
    separately, keep in mind that this is equivalent to::

            a * L1 + b * L2

    where::

            alpha = a + b and l1_ratio = a / (a + b)

    The parameter ``l1_ratio`` corresponds to alpha in the R package glmnet,
    while ``alpha`` corresponds to the lambda parameter in glmnet.
    Specifically, l1_ratio = 1 is the lasso penalty.

    Read more in the :ref:`User Guide <Generalized_linear_regression>`.

    Parameters
    ----------
    alpha : float, optional (default=1)
        Constant that multiplies the penalty terms and thus determines the
        regularization strength.
        See the notes for the exact mathematical meaning of this
        parameter.``alpha = 0`` is equivalent to unpenalized GLMs. In this
        case, the design matrix X must have full column rank
        (no collinearities).

    l1_ratio : float, optional (default=0)
        The elastic net mixing parameter, with ``0 <= l1_ratio <= 1``. For
        ``l1_ratio = 0`` the penalty is an L2 penalty. ``For l1_ratio = 1`` it
        is an L1 penalty.  For ``0 < l1_ratio < 1``, the penalty is a
        combination of L1 and L2.

    P1 : {'identity', array-like}, shape (n_features,), optional \
            (default='identity')
        With this array, you can exclude coefficients from the L1 penalty.
        Set the corresponding value to 1 (include) or 0 (exclude). The
        default value ``'identity'`` is the same as a 1d array of ones.
        Note that n_features = X.shape[1].

    P2 : {'identity', array-like, sparse matrix}, shape \

            (n_features,) or (n_features, n_features), optional \
            (default='identity')
        With this option, you can set the P2 matrix in the L2 penalty `w*P2*w`.
        This gives a fine control over this penalty (Tikhonov regularization).
        A 2d array is directly used as the square matrix P2. A 1d array is
        interpreted as diagonal (square) matrix. The default 'identity' sets
        the identity matrix, which gives the usual squared L2-norm. If you just
        want to exclude certain coefficients, pass a 1d array filled with 1,
        and 0 for the coefficients to be excluded.
        Note that P2 must be positive semi-definite.

    fit_intercept : boolean, optional (default=True)
        Specifies if a constant (a.k.a. bias or intercept) should be
        added to the linear predictor (X*coef+intercept).

    family : {'normal', 'poisson', 'gamma', 'inverse.gaussian', 'binomial'} \
            or an instance of class ExponentialDispersionModel, \
            optional(default='normal')
        The distributional assumption of the GLM, i.e. which distribution from
        the EDM, specifies the loss function to be minimized.

    link : {'auto', 'identity', 'log', 'logit'} or an instance of class Link, \
            optional (default='auto')
        The link function of the GLM, i.e. mapping from linear predictor
        (X*coef) to expectation (mu). Option 'auto' sets the link depending on
        the chosen family as follows:

        - 'identity' for family 'normal'

        - 'log' for families 'poisson', 'gamma', 'inverse.gaussian'

        - 'logit' for family 'binomial'

    fit_dispersion : {None, 'chisqr', 'deviance'}, optional (default=None)
        Method for estimation of the dispersion parameter phi. Whether to use
        the chi squared statistic or the deviance statistic. If None, the
        dispersion is not estimated.

    solver : {'auto', 'irls-cd', 'irls-ls', 'lbfgs'}, \
            optional (default='auto')
        Algorithm to use in the optimization problem:

        'auto'
            Sets 'irls-ls' if l1_ratio equals 0, else 'irls-cd'.

        'irls-cd'
            Iteratively reweighted least squares with a coordinate descent
            inner solver. This can deal with L1 as well as L2 penalties. Note
            that in order to avoid unnecessary memory duplication of X in the
            ``fit`` method, X should be directly passed as a Fortran-contiguous
            numpy array or sparse csc matrix.

        'irls-ls'
            Iteratively reweighted least squares with a least squares inner
            solver. This algorithm cannot deal with L1 penalties.

        'lbfgs'
            Calls scipy's L-BFGS-B optimizer. It cannot deal with L1 penalties.

    max_iter : int, optional (default=100)
        The maximal number of iterations for solver algorithms.

    gradient_tol : float, optional (default=1e-4)
        Stopping criterion. For the irls-ls and lbfgs solvers,
        the iteration will stop when ``max{|g_i|, i = 1, ..., n} <= tol``
        where ``g_i`` is the i-th component of the gradient (derivative) of
        the objective function. For the cd solver, convergence is reached
        when ``sum_i(|minimum-norm of g_i|)``, where ``g_i`` is the
        subgradient of the objective and minimum-norm of ``g_i`` is the element
        of the subgradient ``g_i`` with the smallest L2-norm.

    step_size_tol: float, optional (default=None)

    warm_start : boolean, optional (default=False)
        If set to ``True``, reuse the solution of the previous call to ``fit``
        as initialization for ``coef_`` and ``intercept_`` (supersedes option
        ``start_params``). If set to ``True`` or if the attribute ``coef_``
        does not exit (first call to ``fit``), option ``start_params`` sets the
        start values for ``coef_`` and ``intercept_``.

    n_alphas : int, optional (default=100)
        Number of alphas along the regularization path

    alphas : numpy array, optional (default=None)
        List of alphas where to compute the models.
        If ``None`` alphas are set automatically. Setting 'None' is preferred.

    min_alpha_ratio : float, optional (default=None)
        Length of the path. ``min_alpha_ratio=1e-6`` means that
        ``min_alpha / max_alpha = 1e-6``. If None, 1e-6 is used.

    min_alpha : float, optional (default=None)
        Minimum alpha to estimate the model with. The grid will then be created
        over [max_alpha, min_alpha].


    start_params : array of shape (n_features*, ), optional (default=None)
        Relevant only if ``warm_start=False`` or if fit is called
        the first time (``self.coef_`` does not yet exist).
        All coefficients are set to zero. If ``fit_intercept=True``, the
        start value for the intercept is obtained by the weighted average of y.

        array
        The array of size n_features* is directly used as start values
        for ``coef_``. If ``fit_intercept=True``, the first element
        is assumed to be the start value for the ``intercept_``.
        Note that n_features* = X.shape[1] + fit_intercept, i.e. it includes
        the intercept in counting.

    selection : str, optional (default='cyclic')
        For the solver 'cd' (coordinate descent), the coordinates (features)
        can be updated in either cyclic or random order.
        If set to 'random', a random coefficient is updated every iteration
        rather than looping over features sequentially in the same order. This
        (setting to 'random') often leads to significantly faster convergence
        especially when tol is higher than 1e-4.

    random_state : {int, RandomState instance, None}, optional (default=None)
        The seed of the pseudo random number generator that selects a random
        feature to be updated for solver 'cd' (coordinate descent).
        If int, random_state is the seed used by the random
        number generator; if RandomState instance, random_state is the random
        number generator; if None, the random number generator is the
        RandomState instance used by `np.random`. Used when ``selection`` ==
        'random'.

    copy_X : boolean, optional, (default=True)
        If ``True``, X will be copied; else, it may be overwritten.

    check_input : boolean, optional (default=True)
        Allow to bypass several checks on input: y values in range of family,
        sample_weight non-negative, P2 positive semi-definite.
        Don't use this parameter unless you know what you do.

    verbose : int, optional (default=0)
        For the lbfgs solver set verbose to any positive number for verbosity.

    lower_bounds : np.ndarray, shape=(n_features), optional (default=None)
        Set a lower bound for the coefficients. Setting bounds forces the use
        of the coordinate descent solver (irls-cd).

    upper_bounds : np.ndarray, shape=(n_features), optional (default=None)
        See lower_bounds.

    Attributes
    ----------
    coef_ : array, shape (n_features,)
        Estimated coefficients for the linear predictor (X*coef_+intercept_) in
        the GLM.

    intercept_ : float
        Intercept (a.k.a. bias) added to linear predictor.

    dispersion_ : float
        The dispersion parameter :math:`\\phi` if ``fit_dispersion`` was set.

    n_iter_ : int
        Actual number of iterations used in solver.

    Notes
    -----
    The fit itself does not need Y to be from an EDM, but only assumes
    the first two moments to be :math:`E[Y_i]=\\mu_i=h((Xw)_i)` and
    :math:`Var[Y_i]=\\frac{\\phi}{s_i} v(\\mu_i)`. The unit variance function
    :math:`v(\\mu_i)` is a property of and given by the specific EDM, see
    :ref:`User Guide <Generalized_linear_regression>`.

    The parameters :math:`w` (`coef_` and `intercept_`) are estimated by
    minimizing the deviance plus penalty term, which is equivalent to
    (penalized) maximum likelihood estimation.

    For alpha > 0, the feature matrix X should be standardized in order to
    penalize features equally strong. Call
    :class:`sklearn.preprocessing.StandardScaler` before calling ``fit``.

    If the target y is a ratio, appropriate sample weights s should be
    provided.
    As an example, consider Poisson distributed counts z (integers) and
    weights s=exposure (time, money, persons years, ...). Then you fit
    y = z/s, i.e. ``GeneralizedLinearModel(family='poisson').fit(X, y,
    sample_weight=s)``. The weights are necessary for the right (finite
    sample) mean.
    Consider :math:`\\bar{y} = \\frac{\\sum_i s_i y_i}{\\sum_i s_i}`,
    in this case one might say that y has a 'scaled' Poisson distributions.
    The same holds for other distributions.

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
        alpha=1.0,
        l1_ratio=0,
        P1="identity",
        P2="identity",
        fit_intercept=True,
        family: Union[str, ExponentialDispersionModel] = "normal",
        link: Union[str, Link] = "auto",
        fit_dispersion=None,
        solver="auto",
        max_iter=100,
        gradient_tol: Optional[float] = 1e-4,
        step_size_tol: Optional[float] = None,
        warm_start: bool = False,
        alpha_search: bool = False,
        n_alphas: int = 100,
        alphas: Optional[np.ndarray] = None,
        min_alpha_ratio: Optional[float] = None,
        min_alpha: Optional[float] = None,
        start_params: Optional[np.ndarray] = None,
        selection: str = "cyclic",
        random_state=None,
        copy_X=True,
        check_input=True,
        verbose=0,
        scale_predictors=False,
        lower_bounds: Optional[np.ndarray] = None,
        upper_bounds: Optional[np.ndarray] = None,
        fit_args_reformat="safe",
    ):
        self.alpha = alpha
        self.fit_args_reformat = fit_args_reformat
        super().__init__(
            l1_ratio=l1_ratio,
            P1=P1,
            P2=P2,
            fit_intercept=fit_intercept,
            family=family,
            link=link,
            fit_dispersion=fit_dispersion,
            solver=solver,
            max_iter=max_iter,
            gradient_tol=gradient_tol,
            step_size_tol=step_size_tol,
            warm_start=warm_start,
            alpha_search=alpha_search,
            n_alphas=n_alphas,
            alphas=alphas,
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
        )

    def _validate_hyperparameters(self) -> None:

        if (
            not (isinstance(self.alpha, float) or isinstance(self.alpha, int))
            or self.alpha < 0
        ):
            raise ValueError(
                "Penalty term must be a non-negative number;"
                " got (alpha={})".format(self.alpha)
            )

        if (
            not np.isscalar(self.l1_ratio)
            # check for numeric, i.e. not a string
            or not np.issubdtype(np.asarray(self.l1_ratio).dtype, np.number)
            or self.l1_ratio < 0
            or self.l1_ratio > 1
        ):
            raise ValueError(
                "l1_ratio must be a number in interval [0, 1];"
                " got (l1_ratio={})".format(self.l1_ratio)
            )
        super()._validate_hyperparameters()

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
        offset: Optional[ArrayLike] = None,
        weights_sum: Optional[float] = None,
    ):
        """Fit a Generalized Linear Model.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training data.

        y : array-like, shape (n_samples,)
            Target values.

        sample_weight : {None, array-like}, shape (n_samples,),\
                optional (default=None)
            Individual weights w_i for each sample. Note that for an
            Exponential Dispersion Model (EDM), one has
            Var[Y_i]=phi/w_i * v(mu).
            If Y_i ~ EDM(mu, phi/w_i), then
            sum(w*Y)/sum(w) ~ EDM(mu, phi/sum(w)), i.e. the mean of y is a
            weighted average with weights=sample_weight.

        offset: {None, array-like}, shape (n_samples,), optional (default=None)
            Added to linear predictor "eta". An offset of 3 will increase expected
            y by 3 if the link is linear, and will multiply expected y by 3 if the
            link is log.

        weights_sum: {None, float}, optional (default=None)

        Returns
        -------
        self : returns an instance of self.
        """

        if self.fit_args_reformat == "safe":
            # NOTE: This function checks if all the entries in X and y are
            # finite. That can be expensive. But probably worthwhile.
            X, y, weights, offset, weights_sum = self.set_up_and_check_fit_args(
                X, y, sample_weight, offset, solver=self.solver, copy_X=self.copy_X
            )
        else:
            weights = sample_weight
        assert isinstance(X, mx.MatrixBase)
        assert isinstance(y, np.ndarray)

        self.set_up_for_fit(y)

        # TODO: deal with alpha for regularization path
        if self.alpha > 0 and self.l1_ratio > 0 and self._solver != "irls-cd":
            raise ValueError(
                "The chosen solver (solver={}) can't deal "
                "with L1 penalties, which are included with "
                "(alpha={}) and (l1_ratio={}).".format(
                    self._solver, self.alpha, self.l1_ratio
                )
            )

        _dtype = [np.float64, np.float32]
        if self._solver == "irls-cd":
            _stype = ["csc"]
        else:
            _stype = ["csc", "csr"]

        # 1.3 arguments to take special care ##################################
        # P1, P2, start_params
        P1_no_alpha = setup_p1(self.P1, X, X.dtype, 1, self.l1_ratio)
        P2_no_alpha = setup_p2(self.P2, X, _stype, X.dtype, 1, self.l1_ratio)

        lower_bounds = check_bounds(self.lower_bounds, X.shape[1], X.dtype)
        upper_bounds = check_bounds(self.upper_bounds, X.shape[1], X.dtype)

        if (lower_bounds is not None) and (upper_bounds is not None):
            if np.any(lower_bounds > upper_bounds):
                raise ValueError("Upper bounds must be higher than lower bounds.")

        start_params = initialize_start_params(
            self.start_params,
            n_cols=X.shape[1],
            fit_intercept=self.fit_intercept,
            _dtype=_dtype,
        )

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
        if self._center_predictors:
            X, col_means, col_stds = X.standardize(weights, self.scale_predictors)
        else:
            col_means, col_stds = None, None

        #######################################################################
        # 3. initialization of coef = (intercept_, coef_)                     #
        #######################################################################
        # Note: Since phi=self.dispersion_ does not enter the estimation
        #       of mu_i=E[y_i], set it to 1.

        # set start values for coef
        coef = self.get_start_coef(
            start_params, X, y, weights, offset, col_means, col_stds
        )

        #######################################################################
        # 4. fit                                                              #
        #######################################################################
        if self.alpha_search:
            if self.alphas is None:
                self._alphas = self._get_alpha_path(
                    l1_ratio=self.l1_ratio, X=X, y=y, w=weights, offset=offset
                )
            else:
                self._alphas = self.alphas

            coef = self.solve_regularization_path(
                X=X,
                y=y,
                weights=weights,
                P2_no_alpha=P2_no_alpha,
                P1_no_alpha=P1_no_alpha,
                alphas=self._alphas,
                coef=coef,
                offset=offset,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
            )

            # intercept_ and coef_ return the last estimated alpha
            if self.fit_intercept:
                self.intercept_ = coef[-1, 0]
                self.intercept_path_ = coef[:, 0]
                self.coef_ = coef[-1, 1:]
                self.coef_path_ = coef[:, 1:]
            else:
                # set intercept to zero as the other linear models do
                self.intercept_ = 0.0
                self.intercept_path_ = np.zeros(coef.shape[0])
                self.coef_ = coef[-1, :]
                self.coef_path_ = coef
        else:
            coef = self.solve(
                X=X,
                y=y,
                weights=weights,
                P2=P2_no_alpha * self.alpha,
                P1=P1_no_alpha * self.alpha,
                coef=coef,
                offset=offset,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
            )

            if self.fit_intercept:
                self.intercept_ = coef[0]
                self.coef_ = coef[1:]
            else:
                # set intercept to zero as the other linear models do
                self.intercept_ = 0.0
                self.coef_ = coef

        self.tear_down_from_fit(X, y, col_means, col_stds, weights, weights_sum)

        return self
