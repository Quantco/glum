from __future__ import division

from typing import Union

import numpy as np
from sklearn.linear_model._coordinate_descent import _alpha_grid
from sklearn.model_selection._split import check_cv

from ._distribution import ExponentialDispersionModel
from ._glm import (
    GeneralizedLinearRegressor,
    GeneralizedLinearRegressorBase,
    get_family,
    set_up_and_check_fit_args,
)
from ._link import Link


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

    l1_ratio : float or array of floats, optional (default=0)

    P1 : {'identity', array-like}, shape (n_features,), optional \
            (default='identity')

    P2 : {'identity', array-like, sparse matrix}, shape \

            (n_features,) or (n_features, n_features), optional \
            (default='identity')

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

    tol : float, optional (default=1e-4)
        Stopping criterion for each value of alpha.

    warm_start : boolean, optional (default=False)

    start_params : {'guess', 'zero', array of shape (n_features*, )}, \
            optional (default='guess')

    selection : str, optional (default='cyclic')
    random_state : {int, RandomState instance, None}, optional (default=None)
    diag_fisher : boolean, optional, (default=False)
    copy_X : boolean, optional, (default=True)
    check_input : boolean, optional (default=True)
    center_predictors : boolean, optional (default=True)
    verbose : int, optional (default=0)

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
        fit_dispersion=None,
        solver="auto",
        max_iter=100,
        tol=1e-4,
        warm_start=False,
        start_params="guess",
        selection="cyclic",
        random_state=None,
        diag_fisher=False,
        copy_X=True,
        check_input=True,
        verbose=0,
        scale_predictors=False,
        cv=None,
        n_jobs: int = None,
    ):
        self.eps = eps
        self.n_alphas = n_alphas
        self.alphas = alphas
        self.cv = cv
        self.n_jobs = n_jobs
        super().__init__(
            l1_ratio,
            P1,
            P2,
            fit_intercept,
            family,
            link,
            fit_dispersion,
            solver,
            max_iter,
            tol,
            warm_start,
            start_params,
            selection,
            random_state,
            diag_fisher,
            copy_X,
            check_input,
            verbose,
            scale_predictors,
        )

    def _validate_hyperparameters(self) -> None:
        if self.n_alphas is not None and self.alphas is not None:
            raise ValueError("You cannot specify both alphas and n_alphas.")
        if self.eps is not None and self.alphas is not None:
            raise ValueError("You cannot specify both eps and n_alphas.")
        if self.alphas is None and self.n_alphas is None:
            raise ValueError
        if self.alphas is None and self.eps is None:
            raise ValueError
        if self.alphas is not None and np.any(self.alphas < 0):
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

    def fit(self, X, y, sample_weight=None, offset=None):
        # TODO:
        # 1) stuff from other fit
        # 2) cv stuff
        self._validate_hyperparameters()
        X, y, sample_weight, offset = set_up_and_check_fit_args(
            X, y, sample_weight, offset, solver=self.solver, copy_X=self.copy_X
        )

        l1_ratio = np.atleast_1d(self.l1_ratio)

        # From sklearn.linear_model.LinearModelCV.fit
        if self.alphas is None:
            if self.l1_ratio == 0:
                alphas = [[10.0, 1.0, 0.1] for _ in l1_ratio]
            else:
                # TODO: this is only valid for Gaussian
                alphas = [
                    _alpha_grid(
                        X,
                        y,
                        l1_ratio=l1,
                        fit_intercept=self.fit_intercept,
                        eps=self.eps,
                        n_alphas=self.n_alphas,
                        copy_X=self.copy_X,
                    )
                    for l1 in l1_ratio
                ]

            self.alphas_ = np.asarray(alphas)
            if len(l1_ratio) == 1:
                self.alphas_ = self.alphas_[0]
        else:
            alphas = np.tile(np.sort(self.alphas)[::-1], (len(l1_ratio), 1))
            self.alphas_ = np.asarray(alphas[0])

        cv = check_cv(self.cv)

        self.deviance_path_ = np.full(
            (len(l1_ratio), len(alphas[0]), cv.get_n_splits()), np.nan
        )

        model = GeneralizedLinearRegressor(
            self.alphas_[0],
            l1_ratio[0],
            self.P1,
            self.P2,
            self.fit_intercept,
            self.family,
            self.link,
            self.fit_dispersion,
            self.solver,
            self.max_iter,
            self.tol,
            self.warm_start,
            self.start_params,
            self.selection,
            self.random_state,
            self.diag_fisher,
            self.copy_X,
            self.check_input,
            self.verbose,
            self.scale_predictors,
            fit_args_reformat="unsafe",
        )

        for i, l1 in enumerate(l1_ratio):
            for k, (train_idx, test_idx) in enumerate(cv.split(X)):
                # Reset model - don't warm start
                model.set_params(
                    warm_start=False, check_input=self.check_input, l1_ratio=l1
                )

                x_train, y_train, w_train = (
                    X[train_idx, :],
                    y[train_idx],
                    sample_weight[train_idx],
                )
                x_test, y_test, w_test = (
                    X[test_idx, :],
                    y[test_idx],
                    sample_weight[test_idx],
                )
                if offset is not None:
                    offset_train = offset[train_idx]
                    offset_test = offset[test_idx]
                else:
                    offset_train, offset_test = None, None

                def _get_deviance():
                    return get_family(self.family).deviance(
                        y_test,
                        model.predict(x_test, offset=offset_test),
                        weights=w_test,
                    )

                for j, alpha in enumerate(alphas[i]):
                    model.set_params(alpha=alpha)
                    if j > 0:
                        model.set_params(warm_start=True, check_input=False)
                    model.fit(x_train, y_train, w_train, offset_train)
                    self.deviance_path_[i, j, k] = _get_deviance()

        avg_deviance = self.deviance_path_.mean(2)
        best_idx = np.argmin(avg_deviance)
        # TODO: simplify
        l1_ratios = np.repeat(l1_ratio[:, None], len(alphas[0]), axis=1)
        assert l1_ratios.shape == avg_deviance.shape
        assert np.asarray(alphas).shape == avg_deviance.shape
        self.l1_ratio_ = l1_ratios.flatten()[best_idx]
        self.alpha_ = np.asarray(alphas).flatten()[best_idx]

        # Refit with full data and best alpha and lambda
        model.set_params(
            warm_start=self.warm_start,
            check_input=self.check_input,
            alpha=self.alpha_,
            l1_ratio=self.l1_ratio_,
        )
        model.fit(X, y, sample_weight, offset)
        self.intercept_ = model.intercept_
        self.coef_ = model.coef_
        self.n_iter_ = model.n_iter_
        return self
