from __future__ import division

from typing import Union

import numpy as np
from sklearn.linear_model._coordinate_descent import _alpha_grid
from sklearn.model_selection._split import check_cv

from ._distribution import ExponentialDispersionModel
from ._glm import GeneralizedLinearRegressorBase
from ._link import Link


class GeneralizedLinearRegressorCV(GeneralizedLinearRegressorBase):
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

    l1_ratio : float, optional (default=0)

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

    Attributes
    ----------
    alpha_: float
        The amount of penalization chosen by cross validation

    coef_ : array, shape (n_features,)
        Estimated coefficients for the linear predictor (X*coef_+intercept_) in
        the GLM.

    intercept_ : float
        Intercept (a.k.a. bias) added to linear predictor.

    dispersion_ : float
        The dispersion parameter :math:`\\phi` if ``fit_dispersion`` was set.

    n_iter_ : int
        Actual number of iterations used in solver.

    deviance_path_: array, shape(n_alphas, n_folds)
        Deviance for the test set on each fold, varying alpha

    alphas_: numpy array, shape (n_alphas,)
        The grid of alphas used for fitting
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
        store_cv_values=False,
    ):
        self.eps = eps
        self.n_alphas = n_alphas
        self.alphas = alphas
        self.cv = cv
        self.store_cv_values = store_cv_values
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
        if self.n_alphas is not None:
            raise ValueError("You cannot specify both alphas and n_alphas.")
        if self.eps is not None:
            raise ValueError("You cannot specify both eps and n_alphas.")
        if not (self.alphas > 0).all():
            raise ValueError
        super()._validate_hyperparameters()

    def fit(self, X, y, sample_weight=None, offset=None):
        # TODO:
        # 1) stuff from other fit
        # 2) cv stuff
        # understand this: https://github.com/scikit-learn/scikit-learn/blob/95d4f0841d57e8b5f6b2a570312e9d832e69debc/sklearn/linear_model/_coordinate_descent.py#L1164
        self._validate_hyperparameters()

        if self.alphas is None:
            alphas = _alpha_grid(
                X,
                y,
                l1_ratio=self.l1_ratio,
                fit_intercept=self.fit_intercept,
                eps=self.eps,
                n_alphas=self.n_alphas,
                normalize=self.fit_intercept,
                copy_X=self.copy_X,
            )
        else:
            alphas = np.sort(self.alphas)[::-1]

        cv = check_cv(self.cv)
        folds = list(cv.split(X, y, sample_weight, offset))

        best_mse = np.inf
        print(alphas, folds, best_mse)
        # For first value of alpha, fit with start_params = "guess"
        # for successive ones, warm start with average of previous params
