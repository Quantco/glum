from __future__ import division

import copy
from typing import List, Optional, Union

import numpy as np
from joblib import Parallel, delayed
from scipy import sparse as sparse
from sklearn.model_selection._split import check_cv

from .. import matrix as mx
from ._distribution import ExponentialDispersionModel
from ._glm import (
    GeneralizedLinearRegressorBase,
    _unstandardize,
    check_bounds,
    initialize_start_params,
    is_pos_semidef,
    setup_p1,
    setup_p2,
)
from ._link import Link, LogLink
from ._util import _safe_lin_pred

IndexableArrayLike = Union[
    List,
    np.ndarray,
    sparse.spmatrix,
    mx.DenseGLMDataMatrix,
    mx.MKLSparseMatrix,
    mx.ColScaledMat,
]


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

    n_alphas : int, optional (default=100)
        Number of alphas along the regularization path

    alphas : numpy array, optional (default=None)
        List of alphas where to compute the models.
        If ``None`` alphas are set automatically. Setting 'None' is preferred.

    min_alpha_ratio : float, optional (default=None)
        Length of the path. ``min_alpha_ratio=1e-6`` means that
        ``min_alpha / max_alpha = 1e-6``. None will default to 1e-6.

    min_alpha : float, optional (default=None)
        Minimum alpha to estimate the model with. The grid will then be created
        over [max_alpha, min_alpha].

    start_params : array of shape (n_features*,), optional (default=None)

    selection : str, optional (default='cyclic')
    random_state : {int, RandomState instance, None}, optional (default=None)
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

    alphas_: array, shape (n_l1_ratios, n_alphas)
        Alphas used by the model

    l1_ratio_: float
        The compromise between l1 and l2 penalization chosen by
        cross validation

    coef_ : array, shape (n_features,)
        Estimated coefficients for the linear predictor (X*coef_+intercept_) in
        the GLM at the optimal (l1_ratio_, alpha_)

    intercept_ : float
        Intercept (a.k.a. bias) added to linear predictor.

    dispersion_ : float
        The dispersion parameter :math:`\\phi` if ``fit_dispersion`` was set.

    n_iter_ : int
        number of iterations run by the coordinate descent solver to reach
        the specified tolerance for the optimal alpha.

    coef_path_ : array, shape (n_folds, n_l1_ratios, n_alphas, n_features)
        Estimated coefficients for the linear predictor (X*coef_+intercept_) in
        the GLM at every point along the regularization path.

    deviance_path_: array, shape(n_folds, n_alphas)
        Deviance for the test set on each fold, varying alpha
    """

    def __init__(
        self,
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
        n_alphas: int = 100,
        alphas: Optional[np.ndarray] = None,
        min_alpha_ratio: Optional[float] = None,
        min_alpha: Optional[float] = None,
        start_params: Optional[np.ndarray] = None,
        selection: str = "cyclic",
        random_state=None,
        copy_X: bool = True,
        check_input: bool = True,
        verbose=0,
        scale_predictors: bool = False,
        lower_bounds: Optional[np.ndarray] = None,
        upper_bounds: Optional[np.ndarray] = None,
        cv=None,
        n_jobs: Optional[int] = None,
    ):
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
            n_alphas=n_alphas,
            alphas=alphas,
            min_alpha_ratio=min_alpha_ratio,
            min_alpha=min_alpha,
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

    def fit(
        self,
        # Can't be ArrayLike or contain mx.MatrixBase because mx.SplitMatrix is not
        # indexable
        X: IndexableArrayLike,
        y: IndexableArrayLike,
        sample_weight: Optional[IndexableArrayLike] = None,
        offset: Optional[IndexableArrayLike] = None,
    ):
        X, y, weights, offset, weights_sum = self.set_up_and_check_fit_args(
            X, y, sample_weight, offset, solver=self.solver, copy_X=self.copy_X
        )
        assert isinstance(X, (mx.MKLSparseMatrix, mx.DenseGLMDataMatrix))

        #########
        # Checks
        self.set_up_for_fit(y)
        if (
            hasattr(self._family_instance, "_power")
            and self._family_instance._power == 1.5  # type: ignore
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

        lower_bounds = check_bounds(self.lower_bounds, X.shape[1], X.dtype)
        upper_bounds = check_bounds(self.upper_bounds, X.shape[1], X.dtype)

        cv = check_cv(self.cv)

        if self._solver == "cd":
            _stype = ["csc"]
        else:
            _stype = ["csc", "csr"]

        def fit_path(
            self,
            train_idx,
            test_idx,
            X,
            y,
            P1,
            P2,
            l1,
            alphas,
            weights,
            offset,
            lower_bounds,
            upper_bounds,
            fit_intercept,
        ):
            deviance_path_ = np.full(len(alphas), np.nan)
            coef_path_ = np.full((len(alphas), X.shape[1] + int(fit_intercept)), np.nan)

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

            P1_no_alpha = setup_p1(self.P1, X, X.dtype, 1, l1)
            P2_no_alpha = setup_p2(self.P2, X, _stype, X.dtype, 1, l1)

            if self.check_input:
                # check if P2 is positive semidefinite
                if not isinstance(self.P2, str):  # self.P2 != 'identity'
                    if not is_pos_semidef(P2_no_alpha):
                        if P2_no_alpha.ndim == 1 or P2_no_alpha.shape[0] == 1:
                            error = "1d array P2 must not have negative values."
                        else:
                            error = "P2 must be positive semi-definite."
                        raise ValueError(error)

            coef = self.solve_regularization_path(
                X=x_train,
                y=y_train,
                weights=w_train,
                alphas=alphas,
                P2_no_alpha=P2_no_alpha,
                P1_no_alpha=P1_no_alpha,
                coef=coef,
                offset=offset_train,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
            )

            if self._center_predictors:
                _, intercept, coef_tmp = _unstandardize(
                    copy.copy(x_train), col_means, col_stds, coef[:, 0], coef[:, 1:]
                )
                coef_path_ = np.concatenate(
                    [intercept[:, np.newaxis], coef_tmp], axis=1
                )
            else:
                coef_path_ = coef

            deviance_path_ = [_get_deviance(_coef) for _coef in coef_path_]

            return coef_path_, deviance_path_

        jobs = (
            delayed(fit_path)(
                self,
                train_idx=train_idx,
                test_idx=test_idx,
                X=X,
                y=y,
                P1=self.P1,
                P2=self.P2,
                l1=this_l1_ratio,
                alphas=this_alphas,
                weights=weights,
                offset=offset,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
                fit_intercept=self.fit_intercept,
            )
            for train_idx, test_idx in cv.split(X, y)
            for this_l1_ratio, this_alphas in zip(l1_ratio, alphas)
        )
        paths_data = Parallel(n_jobs=self.n_jobs, prefer="processes")(jobs)

        self.coef_path_ = [elmt[0] for elmt in paths_data]
        self.deviance_path_ = [elmt[1] for elmt in paths_data]

        self.coef_path_ = np.reshape(
            self.coef_path_, (cv.get_n_splits(), len(l1_ratio), len(alphas[0]), -1)
        )
        self.deviance_path_ = np.reshape(
            self.deviance_path_, (cv.get_n_splits(), len(l1_ratio), len(alphas[0]))
        )

        avg_deviance = self.deviance_path_.mean(axis=0)  # type: ignore

        best_l1, best_alpha = np.unravel_index(
            np.argmin(avg_deviance), avg_deviance.shape
        )

        if len(l1_ratio) > 1:
            self.l1_ratio_ = l1_ratio[best_l1]
            self.alpha_ = self.alphas_[best_l1, best_alpha]
        else:
            self.l1_ratio_ = l1_ratio[best_l1]
            self.alpha_ = self.alphas_[best_alpha]

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

        return self
