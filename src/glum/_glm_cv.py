import copy
from typing import Optional, Union

import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection._split import check_cv

from ._distribution import ExponentialDispersionModel
from ._glm import (
    ArrayLike,
    GeneralizedLinearRegressorBase,
    _standardize,
    _unstandardize,
    check_bounds,
    initialize_start_params,
    is_pos_semidef,
    setup_p1,
    setup_p2,
)
from ._link import Link, LogLink
from ._util import _safe_lin_pred


class GeneralizedLinearRegressorCV(GeneralizedLinearRegressorBase):
    """Generalized linear model with iterative fitting along a regularization path.

    The best model is selected by cross-validation.

    Cross-validated regression via a Generalized Linear Model (GLM) with
    penalties. For more on GLMs and on these parameters, see the documentation
    for :class:`GeneralizedLinearRegressor`. CV conventions follow
    :class:`sklearn.linear_model.LassoCV`.

    Parameters
    ----------
    l1_ratio : float or array of floats, optional (default=0)
        If you pass ``l1_ratio`` as an array, the ``fit`` method will choose the
        best value of ``l1_ratio`` and store it as ``self.l1_ratio``.

    P1 : {'identity', array-like}, shape (n_features,), optional (default='identity')
        This array controls the strength of the regularization for each coefficient
        independently. A high value will lead to higher regularization while a value of
        zero will remove the regularization on this parameter.
        Note that ``n_features = X.shape[1]``. If ``X`` is a pandas DataFrame
        with a categorical dtype and P1 has the same size as the number of columns,
        the penalty of the categorical column will be applied to all the levels of
        the categorical.

    P2 : {'identity', array-like, sparse matrix}, shape (n_features,) \
            or (n_features, n_features), optional (default='identity')
        With this option, you can set the P2 matrix in the L2 penalty
        ``w*P2*w``. This gives a fine control over this penalty (Tikhonov
        regularization). A 2d array is directly used as the square matrix P2. A
        1d array is interpreted as diagonal (square) matrix. The default
        ``'identity'`` sets the identity matrix, which gives the usual squared
        L2-norm. If you just want to exclude certain coefficients, pass a 1d
        array filled with 1 and 0 for the coefficients to be excluded. Note that
        P2 must be positive semi-definite. If ``X`` is a pandas DataFrame
        with a categorical dtype and P2 has the same size as the number of columns,
        the penalty of the categorical column will be applied to all the levels of
        the categorical. Note that if P2 is two-dimensional, its size needs to be
        of the same length as the expanded ``X`` matrix.

    fit_intercept : bool, optional (default=True)
        Specifies if a constant (a.k.a. bias or intercept) should be
        added to the linear predictor (``X * coef + intercept``).

    family : str or ExponentialDispersionModel, optional (default='normal')
        The distributional assumption of the GLM, i.e. the loss function to
        minimize. If a string, one of: ``'binomial'``, ``'gamma'``,
        ``'gaussian'``, ``'inverse.gaussian'``, ``'normal'``, ``'poisson'`` or
        ``'tweedie'``. Note that ``'tweedie'`` sets the power of the Tweedie
        distribution to 1.5; to use another value, specify it in parentheses
        (e.g., ``'tweedie (1.5)'``).

    link : {'auto', 'identity', 'log', 'logit'} or Link, optional (default='auto')
        The link function of the GLM, i.e. mapping from linear predictor
        (``X * coef``) to expectation (``mu``). Option ``'auto'`` sets the link
        depending on the chosen family as follows:

        - ``'identity'`` for family ``'normal'``
        - ``'log'`` for families ``'poisson'``, ``'gamma'`` and
          ``'inverse.gaussian'``
        - ``'logit'`` for family ``'binomial'``

    solver : {'auto', 'irls-cd', 'irls-ls', 'lbfgs'}, optional (default='auto')
        Algorithm to use in the optimization problem:

        - ``'auto'``: ``'irls-ls'`` if ``l1_ratio`` is zero and ``'irls-cd'``
          otherwise.
        - ``'irls-cd'``: Iteratively reweighted least squares with a coordinate
          descent inner solver. This can deal with L1 as well as L2 penalties.
          Note that in order to avoid unnecessary memory duplication of X in the
          ``fit`` method, ``X`` should be directly passed as a
          Fortran-contiguous Numpy array or sparse CSC matrix.
        - ``'irls-ls'``: Iteratively reweighted least squares with a least
          squares inner solver. This algorithm cannot deal with L1 penalties.
        - ``'lbfgs'``: Scipy's L-BFGS-B optimizer. It cannot deal with L1
          penalties.

    max_iter : int, optional (default=100)
        The maximal number of iterations for solver algorithms.

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

    n_alphas : int, optional (default=100)
        Number of alphas along the regularization path

    alphas : array-like, optional (default=None)
        List of alphas for which to compute the models. If ``None``, the alphas
        are set automatically. Setting ``None`` is preferred.

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
        tqdm package installed. For the L-BFGS solver, set ``verbose`` to any
        positive number for verbosity.

    scale_predictors: bool, optional (default=False)
        If ``True``, estimate a scaled model where all predictors have a
        standard deviation of 1. This can result in better estimates if
        predictors are on very different scales (for example, centimeters and
        kilometers).

        Advanced developer note: Internally, predictors are always rescaled for
        computational reasons, but this only affects results if
        ``scale_predictors`` is ``True``.

    lower_bounds : array-like, shape (n_features), optional (default=None)
        Set a lower bound for the coefficients. Setting bounds forces the use
        of the coordinate descent solver (``'irls-cd'``).

    upper_bounds : array-like, shape=(n_features), optional (default=None)
        See ``lower_bounds``.

    A_ineq : array-like, shape=(n_constraints, n_features), optional (default=None)
        Constraint matrix for linear inequality constraints of the form
        ``A_ineq w <= b_ineq``.

    b_ineq : array-like, shape=(n_constraints,), optional (default=None)
        Constraint vector for linear inequality constraints of the form
        ``A_ineq w <= b_ineq``.

    cv : int, cross-validation generator or Iterable, optional (default=None)
        Determines the cross-validation splitting strategy. One of:

        - ``None``, to use the default 5-fold cross-validation,
        - ``int``, to specify the number of folds.
        - ``Iterable`` yielding (train, test) splits as arrays of indices.

        For integer/``None`` inputs, :class:`KFold` is used

    n_jobs : int, optional (default=None)
        The maximum number of concurrently running jobs. The number of jobs that
        are needed is ``len(l1_ratio)`` x ``n_folds``. ``-1`` is the same as the
        number of CPU on your machine. ``None`` means ``1`` unless in a
        :obj:`joblib.parallel_backend` context.

    drop_first : bool, optional (default = False)
        If ``True``, drop the first column when encoding categorical variables.

    Attributes
    ----------
    alpha_: float
        The amount of regularization chosen by cross validation.

    alphas_: array, shape (n_l1_ratios, n_alphas)
        Alphas used by the model.

    l1_ratio_: float
        The compromise between L1 and L2 regularization chosen by cross
        validation.

    coef_ : array, shape (n_features,)
        Estimated coefficients for the linear predictor in the GLM at the
        optimal (``l1_ratio_``, ``alpha_``).

    intercept_ : float
        Intercept (a.k.a. bias) added to linear predictor.

    n_iter_ : int
        The number of iterations run by the CD solver to reach the specified
        tolerance for the optimal alpha.

    coef_path_ : array, shape (n_folds, n_l1_ratios, n_alphas, n_features)
        Estimated coefficients for the linear predictor in the GLM at every
        point along the regularization path.

    deviance_path_: array, shape(n_folds, n_alphas)
        Deviance for the test set on each fold, varying alpha.
    """

    def __init__(
        self,
        l1_ratio=0,
        P1="identity",
        P2="identity",
        fit_intercept=True,
        family: Union[str, ExponentialDispersionModel] = "normal",
        link: Union[str, Link] = "auto",
        solver="auto",
        max_iter=100,
        gradient_tol: Optional[float] = None,
        step_size_tol: Optional[float] = None,
        hessian_approx: float = 0.0,
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
        A_ineq: Optional[np.ndarray] = None,
        b_ineq: Optional[np.ndarray] = None,
        force_all_finite: bool = True,
        cv=None,
        n_jobs: Optional[int] = None,
        drop_first: bool = False,
    ):
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
            solver=solver,
            max_iter=max_iter,
            gradient_tol=gradient_tol,
            step_size_tol=step_size_tol,
            hessian_approx=hessian_approx,
            warm_start=warm_start,
            n_alphas=n_alphas,
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
            A_ineq=A_ineq,
            b_ineq=b_ineq,
            force_all_finite=force_all_finite,
            drop_first=drop_first,
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
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
        offset: Optional[ArrayLike] = None,
    ):
        r"""
        Choose the best model along a 'regularization path' by cross-validation.

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
            :math:`\mathrm{var}(y_i) = \phi \times v(mu) / w_i`. If
            :math:`y_i \sim EDM(\mu, \phi / w_i)`, then
            :math:`\sum w_i y_i / \sum w_i \sim EDM(\mu, \phi / \sum w_i)`,
            i.e. the mean of :math:`y` is a weighted average with weights equal
            to ``sample_weight``.

        offset: array-like, shape (n_samples,), optional (default=None)
            Added to linear predictor. An offset of 3 will increase expected
            ``y`` by 3 if the link is linear and will multiply expected ``y`` by
            3 if the link is logarithmic.
        """
        self._validate_hyperparameters()

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
            solver=self.solver,
            force_all_finite=self.force_all_finite,
        )

        #########
        # Checks
        self._set_up_for_fit(y)
        if (
            hasattr(self._family_instance, "_power")
            and self._family_instance._power == 1.5  # type: ignore
        ):
            assert isinstance(self._link_instance, LogLink)

        l1_ratio = np.atleast_1d(self.l1_ratio)

        if self.alphas is None:
            alphas = [self._get_alpha_path(l1, X, y, sample_weight) for l1 in l1_ratio]
        else:
            alphas = np.tile(np.sort(self.alphas)[::-1], (len(l1_ratio), 1))

        if len(l1_ratio) == 1:
            self.alphas_ = alphas[0]
        else:
            self.alphas_ = np.asarray(alphas)

        lower_bounds = check_bounds(self.lower_bounds, X.shape[1], X.dtype)
        upper_bounds = check_bounds(self.upper_bounds, X.shape[1], X.dtype)

        A_ineq = copy.copy(self.A_ineq)
        b_ineq = copy.copy(self.b_ineq)

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
            l1,
            alphas,
            sample_weight,
            offset,
            lower_bounds,
            upper_bounds,
            A_ineq,
            b_ineq,
        ):

            x_train, y_train, w_train = (
                X[train_idx, :],
                y[train_idx],
                sample_weight[train_idx],
            )
            w_train /= w_train.sum()

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

            def _get_deviance(coef):
                mu = self._link_instance.inverse(
                    _safe_lin_pred(x_test, coef, offset_test)
                )
                return self._family_instance.deviance(y_test, mu, sample_weight=w_test)

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

            P1_no_alpha = setup_p1(P1, X, X.dtype, 1, l1)
            P2_no_alpha = setup_p2(P2, X, _stype, X.dtype, 1, l1)

            (
                x_train,
                col_means,
                col_stds,
                lower_bounds,
                upper_bounds,
                A_ineq,
                P1_no_alpha,
                P2_no_alpha,
            ) = _standardize(
                x_train,
                w_train,
                self._center_predictors,
                self.scale_predictors,
                lower_bounds,
                upper_bounds,
                A_ineq,
                P1_no_alpha,
                P2_no_alpha,
            )

            coef = self._get_start_coef(
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
                    if not is_pos_semidef(P2_no_alpha):
                        if P2_no_alpha.ndim == 1 or P2_no_alpha.shape[0] == 1:
                            error = "1d array P2 must not have negative values."
                        else:
                            error = "P2 must be positive semi-definite."
                        raise ValueError(error)

            coef = self._solve_regularization_path(
                X=x_train,
                y=y_train,
                sample_weight=w_train,
                alphas=alphas,
                P2_no_alpha=P2_no_alpha,
                P1_no_alpha=P1_no_alpha,
                coef=coef,
                offset=offset_train,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
                A_ineq=A_ineq,
                b_ineq=b_ineq,
            )

            if self.fit_intercept:
                intercept_path_, coef_path_ = _unstandardize(
                    col_means, col_stds, coef[:, 0], coef[:, 1:]
                )
                assert isinstance(intercept_path_, np.ndarray)  # make mypy happy
                deviance_path_ = [
                    _get_deviance(_coef)
                    for _coef in np.concatenate(
                        [intercept_path_[:, np.newaxis], coef_path_], axis=1
                    )
                ]
            else:
                # set intercept to zero as the other linear models do
                intercept_path_, coef_path_ = _unstandardize(
                    col_means, col_stds, np.zeros(coef.shape[0]), coef
                )
                deviance_path_ = [_get_deviance(_coef) for _coef in coef_path_]

            return intercept_path_, coef_path_, deviance_path_

        jobs = (
            delayed(fit_path)(
                self,
                train_idx=train_idx,
                test_idx=test_idx,
                X=X,
                y=y,
                l1=this_l1_ratio,
                alphas=this_alphas,
                sample_weight=sample_weight,
                offset=offset,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
                A_ineq=A_ineq,
                b_ineq=b_ineq,
            )
            for train_idx, test_idx in cv.split(X, y)
            for this_l1_ratio, this_alphas in zip(l1_ratio, alphas)
        )
        paths_data = Parallel(n_jobs=self.n_jobs, prefer="processes")(jobs)

        self.intercept_path_ = np.reshape(
            [elmt[0] for elmt in paths_data],
            (cv.get_n_splits(), len(l1_ratio), len(alphas[0]), -1),
        )

        self.coef_path_ = np.reshape(
            [elmt[1] for elmt in paths_data],
            (cv.get_n_splits(), len(l1_ratio), len(alphas[0]), -1),
        )

        self.deviance_path_ = np.reshape(
            [elmt[2] for elmt in paths_data],
            (cv.get_n_splits(), len(l1_ratio), len(alphas[0])),
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

        P1 = setup_p1(P1, X, X.dtype, self.alpha_, self.l1_ratio_)
        P2 = setup_p2(P2, X, _stype, X.dtype, self.alpha_, self.l1_ratio_)

        # Refit with full data and best alpha and lambda
        (
            X,
            col_means,
            col_stds,
            lower_bounds,
            upper_bounds,
            A_ineq,
            P1,
            P2,
        ) = _standardize(
            X,
            sample_weight,
            self._center_predictors,
            self.scale_predictors,
            lower_bounds,
            upper_bounds,
            A_ineq,
            P1,
            P2,
        )

        start_params = initialize_start_params(
            self.start_params,
            n_cols=X.shape[1],
            fit_intercept=self.fit_intercept,
            _dtype=X.dtype,
        )

        coef = self._get_start_coef(
            start_params, X, y, sample_weight, offset, col_means, col_stds
        )

        coef = self._solve(
            X=X,
            y=y,
            sample_weight=sample_weight,
            P2=P2,
            P1=P1,  # type: ignore
            coef=coef,
            offset=offset,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            A_ineq=A_ineq,
            b_ineq=b_ineq,
        )

        if self.fit_intercept:
            self.intercept_, self.coef_ = _unstandardize(
                col_means, col_stds, coef[0], coef[1:]
            )
        else:
            # set intercept to zero as the other linear models do
            self.intercept_, self.coef_ = _unstandardize(col_means, col_stds, 0.0, coef)

        self._tear_down_from_fit()

        return self
