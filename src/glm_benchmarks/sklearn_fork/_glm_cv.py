from __future__ import division

import copy
from typing import Optional, Union

import numpy as np
from sklearn.model_selection._split import check_cv

from ._distribution import ExponentialDispersionModel, TweedieDistribution
from ._glm import (
    GeneralizedLinearRegressorBase,
    _unstandardize,
    check_bounds,
    get_family,
    get_link,
    initialize_start_params,
    is_pos_semidef,
    setup_p1,
    setup_p2,
)
from ._link import IdentityLink, Link, LogLink
from ._util import _safe_lin_pred


class GeneralizedLinearRegressorCV(GeneralizedLinearRegressorBase):
    # TODO: add n_jobs
    """
    Generalized linear model like GeneralizedLinearRegressor with iterative fitting
    along a regularization path. See glossary entry for
    :term:`cross-validation estimator`.

    The best model is selected by cross-validation.

    Cross-validated regression via a Generalized Linear Model (GLM) with penalties.
    For more on GLMs and on these parameters,
    see the documentation for GeneralizedLinearRegressor. CV conventions follow
    sklearn LassoCV.

    Parameters
    ----------
    eps : float, optional
        Length of the path. ``eps=1e-3`` means that
        ``alpha_min / alpha_max = 1e-3``.
    n_alphas : int, optional
        Number of alphas along the regularization path
    alphas : numpy array, optional
        List of alphas where to compute the models.
        If ``None`` alphas are set automatically. Setting 'None' is preferred.

    l1_ratio : float or array of floats, optional (default=0). If you pass l1_ratio
        as an array, the `fit` method will choose the best value of l1_ratio and store
        it as self.l1_ratio.

    P1 : {'identity', array-like}, shape (n_features,), optional (default='identity')

    P2 : {'identity', array-like, sparse matrix}, shape (n_features,)
        or (n_features, n_features), optional (default='identity')

    fit_intercept : boolean, optional (default=True)

    family : {'normal', 'poisson', 'gamma', 'inverse.gaussian', 'binomial'} \
            or an instance of class ExponentialDispersionModel, \
            optional(default='normal')

    link : {'auto', 'identity', 'log', 'logit'} or an instance of class Link, \
            optional (default='auto')

    fit_dispersion : {None, 'chisqr', 'deviance'}, optional (default=None)

    solver : {'auto', 'cd', 'irls', 'lbfgs'}, \
            optional (default='auto')

    max_iter : int, optional (default=100)
        The maximal number of iterations per value of alpha for solver algorithms.

    gradient_tol : float, optional (default=1e-4)
        Stopping criterion for each value of alpha.

    step_size_tol: float, optional (default=None)

    warm_start : boolean, optional (default=False)

    start_params : array of shape (n_features*,), optional (default=None)

    selection : str, optional (default='cyclic')
    random_state : {int, RandomState instance, None}, optional (default=None)
    diag_fisher : boolean, optional, (default=False)
    copy_X : boolean, optional, (default=True)
    check_input : boolean, optional (default=True)
    verbose : int, optional (default=0)

    lower_bounds : np.ndarray, shape=(n_features), optional (default=None)
        Set a lower bound for the coefficients. Setting bounds forces the use
        of the coordinate descent solver (irls-cd).

    upper_bounds : np.ndarray, shape=(n_features), optional (default=None)
        See lower_bounds.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross-validation,
        - integer, to specify the number of folds.
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, :class:`KFold` is used.
        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

        .. versionchanged:: 0.22
            ``cv`` default value if None changed from 3-fold to 5-fold.

     n_jobs : int or None, optional (default=None)
        Number of CPUs to use during the cross validation.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    Attributes
    ----------
    alpha_: float
        The amount of penalization chosen by cross validation

    l1_ratio_: float
        The compromise between l1 and l2 penalization chosen by
        cross validation

    coef_ : array, shape (n_features,)
        Estimated coefficients for the linear predictor (X*coef_+intercept_) in
        the GLM.

    intercept_ : float
        Intercept (a.k.a. bias) added to linear predictor.

    dispersion_ : float
        The dispersion parameter :math:`\\phi` if ``fit_dispersion`` was set.

    n_iter_ : int
        number of iterations run by the coordinate descent solver to reach
        the specified tolerance for the optimal alpha.

    deviance_path_: array, shape(n_alphas, n_folds)
        Deviance for the test set on each fold, varying alpha

    """

    def __init__(
        self,
        eps: float = 1e-3,
        n_alphas: int = 100,
        alphas: np.ndarray = None,
        l1_ratio=0,
        P1="identity",
        P2="identity",
        fit_intercept=True,
        family: Union[str, ExponentialDispersionModel] = "normal",
        link: Union[str, Link] = "auto",
        fit_dispersion: Optional[bool] = None,
        solver="auto",
        max_iter=100,
        gradient_tol: Optional[float] = 1e-4,
        step_size_tol: Optional[float] = None,
        warm_start: bool = False,
        start_params: Optional[np.ndarray] = None,
        selection: str = "cyclic",
        random_state=None,
        diag_fisher: bool = False,
        copy_X: bool = True,
        check_input: bool = True,
        verbose=0,
        scale_predictors: bool = False,
        lower_bounds: Optional[np.ndarray] = None,
        upper_bounds: Optional[np.ndarray] = None,
        cv=None,
        n_jobs: Optional[int] = None,
    ):
        self.eps = eps
        self.n_alphas = n_alphas
        self.alphas = alphas
        self.cv = cv
        self.n_jobs = n_jobs
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
            start_params=start_params,
            selection=selection,
            random_state=random_state,
            diag_fisher=diag_fisher,
            copy_X=copy_X,
            check_input=check_input,
            verbose=verbose,
            scale_predictors=scale_predictors,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
        )

    def _validate_hyperparameters(self) -> None:
        if self.alphas is not None and np.any(np.asarray(self.alphas) < 0):
            raise ValueError
        l1_ratio = np.asarray(self.l1_ratio)
        if (
            not np.issubdtype(l1_ratio.dtype, np.number)
            or np.any(l1_ratio < 0)
            or np.any(l1_ratio > 1)
        ):
            raise ValueError(
                "l1_ratio must be a number in interval [0, 1]; got l1_ratio={}".format(
                    self.l1_ratio
                )
            )
        super()._validate_hyperparameters()

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
        whatever is specified by the input parameters eps and n_alphas..

        eps is the length of the path, with 1e-3 as the default.
        """

        def _make_grid(max_alpha: float) -> np.ndarray:
            min_alpha = max_alpha * self.eps
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

        if isinstance(get_link(self.link, get_family(self.family)), IdentityLink):
            # assume normal distribution
            grad = _get_normal_identity_grad_at_zeros_with_optimal_intercept()
        else:
            # assume log link and tweedie distribution
            grad = _get_tweedie_log_grad_at_zeros_with_optimal_intercept()

        alpha_max = np.max(np.abs(grad)) / l1_ratio
        return _make_grid(alpha_max)

    def fit(self, X, y, sample_weight=None, offset=None):
        X, y, weights, offset, weights_sum = self.set_up_and_check_fit_args(
            X, y, sample_weight, offset, solver=self.solver, copy_X=self.copy_X
        )

        self.set_up_for_fit(y)
        if (
            hasattr(self._family_instance, "_power")
            and self._family_instance._power == 1.5
        ):
            assert isinstance(self._link_instance, LogLink)

        l1_ratio = np.atleast_1d(self.l1_ratio)

        if self.alphas is None:
            alphas = [self._get_alpha_path(l1, X, y, weights) for l1 in l1_ratio]
        else:
            alphas = np.tile(np.sort(self.alphas)[::-1], (len(l1_ratio), 1))

        if len(l1_ratio) == 1:
            self.alphas_ = alphas[0]
        else:
            self.alphas_ = np.asarray(alphas)

        lower_bounds = check_bounds(self.lower_bounds, X.shape[1])
        upper_bounds = check_bounds(self.upper_bounds, X.shape[1])

        cv = check_cv(self.cv)

        self.deviance_path_ = np.full(
            (len(l1_ratio), len(alphas[0]), cv.get_n_splits()), np.nan
        )
        self.mse_path_ = self.deviance_path_.copy()
        self.coef_path_ = np.full(
            (
                len(l1_ratio),
                len(alphas[0]),
                cv.get_n_splits(),
                X.shape[1] + int(self.fit_intercept),
            ),
            np.nan,
        )
        if self._solver == "cd":
            _stype = ["csc"]
        else:
            _stype = ["csc", "csr"]

        for i, l1 in enumerate(l1_ratio):
            for k, (train_idx, test_idx) in enumerate(cv.split(X, y)):
                x_train, y_train, w_train = (
                    X[train_idx, :],
                    y[train_idx],
                    weights[train_idx],
                )
                w_train /= w_train.sum()

                x_test, y_test, w_test = (
                    X[test_idx, :],
                    y[test_idx],
                    weights[test_idx],
                )

                if self._center_predictors:
                    x_train, col_means, col_stds = x_train.standardize(
                        w_train, self.scale_predictors
                    )
                else:
                    col_means, col_stds = None, None

                if offset is not None:
                    offset_train = offset[train_idx]
                    offset_test = offset[test_idx]
                else:
                    offset_train, offset_test = None, None

                def _get_deviance(coef):
                    mu = self._link_instance.inverse(
                        _safe_lin_pred(x_test, coef, offset_test)
                    )

                    return self._family_instance.deviance(y_test, mu, weights=w_test)

                def _get_mse(coef):
                    mu = self._link_instance.inverse(
                        _safe_lin_pred(x_test, coef, offset_test)
                    )

                    return w_test.dot((y_test - mu) ** 2) / w_test.sum()

                P2 = setup_p2(self.P2, X, _stype, X.dtype, alphas[i][0], l1)

                if (
                    hasattr(self._family_instance, "_power")
                    and self._family_instance._power == 1.5
                ):
                    assert isinstance(self._link_instance, LogLink)

                _dtype = [np.float64, np.float32]
                start_params = initialize_start_params(
                    self.start_params,
                    n_cols=X.shape[1],
                    fit_intercept=self.fit_intercept,
                    _dtype=_dtype,
                )
                coef = self.get_start_coef(
                    start_params,
                    x_train,
                    y_train,
                    w_train,
                    offset_train,
                    col_means,
                    col_stds,
                )

                if self.check_input:
                    # check if P2 is positive semidefinite
                    if not isinstance(self.P2, str):  # self.P2 != 'identity'
                        if not is_pos_semidef(P2):
                            if P2.ndim == 1 or P2.shape[0] == 1:
                                error = "1d array P2 must not have negative values."
                            else:
                                error = "P2 must be positive semi-definite."
                            raise ValueError(error)

                for j, alpha in enumerate(alphas[i]):
                    P1 = setup_p1(self.P1, X, X.dtype, alpha, l1)
                    P2 = setup_p2(self.P2, X, _stype, X.dtype, alpha, l1)

                    # TODO: see if we need to deal with centering or something
                    # use standardize_warm_start?
                    # TODO: write simpler tests against sklearn ridge
                    coef = self.solve(
                        X=x_train,
                        y=y_train,
                        weights=w_train,
                        P2=P2,
                        P1=P1,
                        coef=coef,
                        offset=offset_train,
                        lower_bounds=lower_bounds,
                        upper_bounds=upper_bounds,
                    )

                    if self._center_predictors:
                        _, intercept, coef_tmp = _unstandardize(
                            copy.copy(x_train),
                            col_means,
                            col_stds,
                            coef[0],
                            coef[1:].copy(),
                        )
                        coef_uncentered = np.concatenate([[intercept], coef_tmp])
                    else:
                        coef_uncentered = coef

                    self.deviance_path_[i, j, k] = _get_deviance(coef_uncentered)
                    self.mse_path_[i, j, k] = _get_mse(coef_uncentered)
                    self.coef_path_[i, j, k, :] = coef_uncentered

        avg_deviance = self.deviance_path_.mean(2)
        best_idx = np.argmin(avg_deviance)
        # # TODO: simplify
        l1_ratios = np.repeat(l1_ratio[:, None], len(alphas[0]), axis=1)
        self.l1_ratio_ = l1_ratios.flatten()[best_idx]
        self.alpha_ = np.asarray(alphas).flatten()[best_idx]

        P1 = setup_p1(self.P1, X, X.dtype, self.alpha_, self.l1_ratio_)
        P2 = setup_p2(self.P2, X, _stype, X.dtype, self.alpha_, self.l1_ratio_)

        # Refit with full data and best alpha and lambda
        if self._center_predictors:
            X, col_means, col_stds = X.standardize(weights, self.scale_predictors)
        else:
            col_means, col_stds = None, None

        start_params = initialize_start_params(
            self.start_params,
            n_cols=X.shape[1],
            fit_intercept=self.fit_intercept,
            _dtype=X.dtype,
        )

        coef = self.get_start_coef(
            start_params, X, y, weights, offset, col_means, col_stds
        )
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

        if self.fit_intercept:
            self.intercept_ = coef[0]
            self.coef_ = coef[1:]
        else:
            # set intercept to zero as the other linear models do
            self.intercept_ = 0.0
            self.coef_ = coef

        self.tear_down_from_fit(X, y, col_means, col_stds, weights, weights_sum)
        self.mse_path_ = np.squeeze(self.mse_path_)
        return self
