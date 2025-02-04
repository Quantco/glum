import warnings
from collections.abc import Iterable, Mapping
from typing import Any, Optional, Union

import formulaic
import numpy as np
import sklearn as skl
import tabmat as tm
from scipy import sparse

from ._distribution import ExponentialDispersionModel
from ._formula import capture_context
from ._glm import GeneralizedLinearRegressorBase, setup_p1, setup_p2
from ._linalg import is_pos_semidef
from ._link import Link
from ._typing import ArrayLike, ShapedArrayLike
from ._utils import standardize, unstandardize
from ._validation import check_bounds, check_inequality_constraints


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
        formula: Optional[formulaic.FormulaSpec] = None,
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
            context=capture_context(context),
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
        ) = standardize(
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
                self.intercept_path_, self.coef_path_ = unstandardize(
                    self.col_means_, self.col_stds_, coef[:, 0], coef[:, 1:]
                )
                self.intercept_ = self.intercept_path_[-1]  # type: ignore
                self.coef_ = self.coef_path_[-1]
            else:
                # set intercept to zero as the other linear models do
                self.intercept_path_, self.coef_path_ = unstandardize(
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
                self.intercept_, self.coef_ = unstandardize(
                    self.col_means_, self.col_stds_, coef[0], coef[1:]
                )
            else:
                # set intercept to zero as the other linear models do
                self.intercept_, self.coef_ = unstandardize(
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
        return self._get_info_criteria("aic", X, y, sample_weight, context=context)

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
        aicc = self._get_info_criteria("aicc", X, y, sample_weight, context=context)

        if not aicc:
            msg = "Model degrees of freedom should be more than training data points."
            raise ValueError(msg)

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
        return self._get_info_criteria("bic", X, y, sample_weight, context=context)

    def _get_info_criteria(
        self,
        crit: str,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
        context: Optional[Union[int, Mapping[str, Any]]] = None,
    ):
        skl.utils.validation.check_is_fitted(self, "coef_")

        context = capture_context(context)

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
