from abc import ABCMeta, abstractmethod
from functools import partial
from typing import Union

import numexpr
import numpy as np
from scipy import special
from tabmat import MatrixBase, StandardizedMatrix, hstack

from ._functions import (
    binomial_logit_eta_mu_deviance,
    binomial_logit_rowwise_gradient_hessian,
    gamma_deviance,
    gamma_log_eta_mu_deviance,
    gamma_log_likelihood,
    gamma_log_rowwise_gradient_hessian,
    inv_gaussian_deviance,
    inv_gaussian_log_eta_mu_deviance,
    inv_gaussian_log_likelihood,
    inv_gaussian_log_rowwise_gradient_hessian,
    negative_binomial_deviance,
    negative_binomial_log_eta_mu_deviance,
    negative_binomial_log_likelihood,
    negative_binomial_log_rowwise_gradient_hessian,
    normal_deviance,
    normal_identity_eta_mu_deviance,
    normal_identity_rowwise_gradient_hessian,
    normal_log_likelihood,
    poisson_deviance,
    poisson_log_eta_mu_deviance,
    poisson_log_likelihood,
    poisson_log_rowwise_gradient_hessian,
    tweedie_deviance,
    tweedie_log_eta_mu_deviance,
    tweedie_log_likelihood,
    tweedie_log_rowwise_gradient_hessian,
)
from ._link import IdentityLink, Link, LogitLink, LogLink
from ._util import _safe_lin_pred, _safe_sandwich_dot


class ExponentialDispersionModel(metaclass=ABCMeta):
    r"""Base class for reproductive Exponential Dispersion Models (EDM).

    The PDF of :math:`Y \sim \mathrm{EDM}(\theta, \phi)` is given by

    .. math::

        p(y \mid \theta, \phi)
        &= \exp \left(\frac{y \theta - b(\theta)}{\phi / w} + c(y; w / \phi) \right),

    where :math:`\theta` is the scale parameter, :math:`\phi` is the dispersion
    parameter, :math:`w` is a given weight, :math:`b` is the cumulant function
    and :math:`c` is a normalization term.

    It can be shown that :math:`\mathrm{E}(Y) = b'(\theta)` and
    :math:`\mathrm{var}(Y) = b''(\theta) \times \phi / w`.

    References
    ----------
    < https://en.wikipedia.org/wiki/Exponential_dispersion_model >.
    """

    @property
    @abstractmethod
    def lower_bound(self) -> float:
        """Get the lower bound of values for the EDM."""
        pass

    @property
    @abstractmethod
    def upper_bound(self) -> float:
        """Get the upper bound of values for the EDM."""
        pass

    @property
    @abstractmethod
    def include_lower_bound(self) -> bool:
        """Return whether ``lower_bound`` is allowed as a value of ``y``."""
        pass

    @property
    @abstractmethod
    def include_upper_bound(self) -> bool:
        """Return whether ``upper_bound`` is allowed as a value of ``y``."""
        pass

    def in_y_range(self, x) -> np.ndarray:
        """Return ``True`` if ``x`` is in the valid range of the EDM."""
        if self.include_lower_bound:
            lb_op: np.ufunc = np.greater_equal
        else:
            lb_op = np.greater

        if self.include_upper_bound:
            ub_op: np.ufunc = np.less_equal
        else:
            ub_op = np.less

        return lb_op(x, self.lower_bound) & ub_op(x, self.upper_bound)

    def to_tweedie(self, safe=True):
        """Return the Tweedie representation of a distribution if it exists."""
        if hasattr(self, "__tweedie_repr__"):
            return self.__tweedie_repr__()
        if safe:
            raise ValueError("This distribution has no Tweedie representation.")
        return None

    @abstractmethod
    def unit_variance(self, mu):
        r"""Compute the unit variance.

        The unit variance, :math:`v(\mu) \equiv b''((b')^{-1} (\mu))`,
        determines the variance as a function of the mean :math:`\mu` by
        :math:`\mathrm{var}(y_i) = v(\mu_i) \times \phi / w_i`. It can also be
        derived from the unit deviance :math:`d(y, \mu)` as

        .. math::

            v(\mu) = 2 \div \frac{\partial^2 d(y, \mu)}{\partial\mu^2} \big| _{y=\mu}.

        See also :meth:`~ExponentialDispersionModel.variance`.

        Parameters
        ----------
        mu : array-like, shape (n_samples,)
            Predicted mean.
        """
        pass

    @abstractmethod
    def unit_variance_derivative(self, mu):
        r"""Compute the derivative of the unit variance with respect to ``mu``.

        Parameters
        ----------
        mu : array-like, shape (n_samples,)
            Predicted mean.
        """
        pass

    def variance(self, mu, dispersion=1, sample_weight=1) -> np.ndarray:
        r"""Compute the variance function.

        The variance of :math:`Y_i \sim \mathrm{EDM}(\mu_i, \phi / w_i)` takes
        the form :math:`v(\mu_i) \times \phi / w_i`, where :math:`v(\mu)` is the
        unit variance and :math:`w_i` are weights.

        Parameters
        ----------
        mu : array-like, shape (n_samples,)
            Predicted mean.

        dispersion : float, optional (default=1)
            Dispersion parameter :math:`\phi`.

        sample_weight : array-like, shape (n_samples,), optional (default=1)
            Weights or exposure to which variance is inverse proportional.

        Returns
        -------
        array-like, shape (n_samples,)
        """
        return self.unit_variance(mu) * dispersion / sample_weight

    def variance_derivative(self, mu, dispersion=1, sample_weight=1):
        r"""Compute the derivative of the variance with respect to ``mu``.

        The derivative of the variance is equal to
        :math:`v(\mu_i) \times \phi / w_i`, where :math:`v(\mu)` is the unit
        variance and :math:`ws_i` are weights.

        Parameters
        ----------
        mu : array-like, shape (n_samples,)
            Predicted mean.

        dispersion : float, optional (default=1)
            Dispersion parameter :math:`\phi`.

        sample_weight : array-like, shape (n_samples,), optional (default=1)
            Weights or exposure to which variance is inverse proportional.

        Returns
        -------
        array-like, shape (n_samples,)
        """
        return self.unit_variance_derivative(mu) * dispersion / sample_weight

    @abstractmethod
    def unit_deviance(self, y, mu):
        r"""Compute the unit deviance.

        In terms of the unit log likelihood :math:`\ell`, the unit deviance is
        :math:`2 [\ell(y_i, y_i, \phi) - \ell(y_i, \mu, \phi)]`, i.e. twice the
        difference between the log likelihood of a saturated model (with one
        parameter per observation) and the model at hand.

        Parameters
        ----------
        y : array-like, shape (n_samples,)
            Target values.

        mu : array-like, shape (n_samples,)
            Predicted mean.
        """
        pass

    def unit_deviance_derivative(self, y, mu):
        r"""Compute the derivative of the unit deviance with respect to ``mu``.

        The derivative of the unit deviance is given by
        :math:`2 \times (\mu - y) / v(\mu)`, where :math:`v(\mu)` is the unit
        variance.

        Parameters
        ----------
        y : array-like, shape (n_samples,)
            Target values.

        mu : array-like, shape (n_samples,)
            Predicted mean.

        Returns
        -------
        array-like, shape (n_samples,)
        """
        return -2 * (y - mu) / self.unit_variance(mu)

    def deviance(self, y, mu, sample_weight=1) -> float:
        r"""Compute the deviance.

        The deviance is a weighted sum of the unit deviances. In terms of the
        unit log likelihood :math:`\ell`, it equals
        :math:`2 \sum_i [\ell(y_i, y_i, \phi) - \ell(y_i, \mu, \phi)]`,
        i.e. twice the difference between the log likelihood of a saturated
        model (with one parameter per observation) and the model at hand.

        Parameters
        ----------
        y : array-like, shape (n_samples,)
            Target values.

        mu : array-like, shape (n_samples,)
            Predicted mean.

        sample_weight : array-like, shape (n_samples,), optional (default=1)
            Weights or exposure to which the variance is inversely proportional.
        """
        if sample_weight is None:
            return np.sum(self.unit_deviance(y, mu))
        else:
            return np.sum(self.unit_deviance(y, mu) * sample_weight)

    def deviance_derivative(self, y, mu, sample_weight=1):
        r"""Compute the derivative of the deviance with respect to ``mu``.

        Parameters
        ----------
        y : array-like, shape (n_samples,)
            Target values.

        mu : array-like, shape (n_samples,)
            Predicted mean.

        sample_weight : array-like, shape (n_samples,) (default=1)
            Weights or exposure to which variance is inverse proportional.

        Returns
        -------
        array-like, shape (n_samples,)
        """
        return sample_weight * self.unit_deviance_derivative(y, mu)

    def _mu_deviance_derivative(
        self,
        coef,
        X: Union[MatrixBase, StandardizedMatrix],
        y,
        sample_weight,
        link: Link,
        offset=None,
    ):
        """Compute ``mu`` and the derivative of the deviance with respect to
        coefficients.
        """
        lin_pred = _safe_lin_pred(X, coef, offset)
        mu = link.inverse(lin_pred)
        d1 = link.inverse_derivative(lin_pred)
        temp = d1 * self.deviance_derivative(y, mu, sample_weight)

        if coef.size == X.shape[1] + 1:
            devp = np.concatenate(([temp.sum()], temp @ X))
        else:
            devp = temp @ X  # same as X.T @ temp

        return mu, devp

    def eta_mu_deviance(
        self,
        link: Link,
        factor: float,
        cur_eta,
        X_dot_d,
        y,
        sample_weight,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Compute ``eta``, ``mu`` and the deviance.

        Returns
        -------
        numpy.ndarray, shape (X.shape[0],)
            The linear predictor, ``eta``, as ``cur_eta + factor * X_dot_d``.
        numpy.ndarray, shape (X.shape[0],)
            The link-function-transformed prediction, ``mu``.
        float
            The deviance.
        """
        # eta_out and mu_out are filled inside self._eta_mu_deviance,
        # avoiding allocating new arrays for every line search loop
        eta_out = np.empty_like(cur_eta)
        mu_out = np.empty_like(cur_eta)

        deviance = self._eta_mu_deviance(
            link, factor, cur_eta, X_dot_d, y, sample_weight, eta_out, mu_out
        )

        return eta_out, mu_out, deviance

    def _eta_mu_deviance(
        self,
        link: Link,
        factor: float,
        cur_eta,
        X_dot_d,
        y,
        sample_weight,
        eta_out,
        mu_out,
    ) -> float:
        """Update ``eta`` and ``mu`` and compute the deviance.

        This is a default implementation that should work for all valid
        distributions and link functions. To implement a custom optimized
        version for a specific distribution and link function, please override
        this function in the subclass.

        Returns
        -------
        float
            The deviance.
        """
        eta_out[:] = cur_eta + factor * X_dot_d
        mu_out[:] = link.inverse(eta_out)
        return self.deviance(y, mu_out, sample_weight=sample_weight)

    def rowwise_gradient_hessian(
        self,
        link: Link,
        coef,
        dispersion,
        X: Union[MatrixBase, StandardizedMatrix],
        y,
        sample_weight,
        eta,
        mu,
        offset=None,
    ):
        """Compute the gradient and negative Hessian of the log likelihood row-wise.

        Returns
        -------
        numpy.ndarray, shape (X.shape[0],)
            The gradient of the log likelihood, row-wise.
        numpy.ndarray, shape (X.shape[0],)
            The negative Hessian of the log likelihood, row-wise.
        """
        gradient_rows = np.empty_like(mu)
        hessian_rows = np.empty_like(mu)

        self._rowwise_gradient_hessian(
            link, y, sample_weight, eta, mu, gradient_rows, hessian_rows
        )

        # To form the full Hessian matrix from the IRLS sample_weight:
        # hessian_matrix = _safe_sandwich_dot(X, hessian_rows, intercept=intercept)
        return gradient_rows, hessian_rows

    def _rowwise_gradient_hessian(
        self, link, y, sample_weight, eta, mu, gradient_rows, hessian_rows
    ):
        """Update ``gradient_rows`` and ``hessian_rows`` in place.

        This is a default implementation that should work for all valid
        distributions and link functions. To implement a custom optimized
        version for a specific distribution and link function, please override
        this function in the subclass.
        """
        # FOR TWEEDIE: sigma_inv = weights / (mu ** p) during optimization bc phi = 1
        sigma_inv = get_one_over_variance(self, link, mu, eta, 1.0, sample_weight)
        d1 = link.inverse_derivative(eta)  # = h'(eta)
        # Alternatively:
        # h'(eta) = h'(g(mu)) = 1/g'(mu), note that h is inverse of g
        # d1 = 1./link.derivative(mu)
        d1_sigma_inv = d1 * sigma_inv
        gradient_rows[:] = d1_sigma_inv * (y - mu)
        hessian_rows[:] = d1 * d1_sigma_inv

    def _fisher_information(
        self, link, X, y, mu, sample_weight, dispersion, fit_intercept
    ):
        """Compute the expected information matrix.

        Parameters
        ----------
        link : Link
            A link function (i.e. an instance of :class:`~glum._link.Link`).
        X : array-like
            Training data.
        y : array-like
            Target values.
        mu : array-like
            Predicted mean.
        sample_weight : array-like
            Weights or exposure to which variance is inversely proportional.
        dispersion : float
            The dispersion parameter.
        fit_intercept : bool
            Whether the model has an intercept.
        """
        W = (link.inverse_derivative(link.link(mu)) ** 2) * get_one_over_variance(
            self, link, mu, link.link(mu), dispersion, sample_weight
        )

        return _safe_sandwich_dot(X, W, intercept=fit_intercept)

    def _observed_information(
        self, link, X, y, mu, sample_weight, dispersion, fit_intercept
    ):
        """Compute the observed information matrix.

        Parameters
        ----------
        X : array-like
            Training data.
        y : array-like
            Target values.
        mu : array-like
            Predicted mean.
        sample_weight : array-like
            Weights or exposure to which variance is inversely proportional.
        dispersion : float
            The dispersion parameter.
        fit_intercept : bool
            Whether the model has an intercept.
        """
        linpred = link.link(mu)

        W = (
            -link.inverse_derivative2(linpred) * (y - mu)
            + (link.inverse_derivative(linpred) ** 2)
            * (
                1
                + (y - mu) * self.unit_variance_derivative(mu) / self.unit_variance(mu)
            )
        ) * get_one_over_variance(self, link, mu, linpred, dispersion, sample_weight)

        return _safe_sandwich_dot(X, W, intercept=fit_intercept)

    def _score_matrix(self, link, X, y, mu, sample_weight, dispersion, fit_intercept):
        """Compute the score.

        Parameters
        ----------
        X : array-like
            Training data.
        y : array-like
            Target values.
        mu : array-like
            Predicted mean.
        sample_weight : array-like
            Weights or exposure to which variance is inversely proportional.
        dispersion : float
            The dispersion parameter.
        fit_intercept : bool
            Whether the model has an intercept.
        """
        linpred = link.link(mu)

        W = (
            get_one_over_variance(self, link, mu, linpred, dispersion, sample_weight)
            * link.inverse_derivative(linpred)
            * (y - mu)
        ).reshape(-1, 1)
        XW = X.multiply(W)
        if fit_intercept:
            return hstack((W, XW))
        else:
            return XW

    def dispersion(self, y, mu, sample_weight=None, ddof=1, method="pearson") -> float:
        r"""Estimate the dispersion parameter :math:`\phi`.

        Parameters
        ----------
        y : array-like, shape (n_samples,)
            Target values.

        mu : array-like, shape (n_samples,)
            Predicted mean.

        sample_weight : array-like, shape (n_samples,), optional (default=1)
            Weights or exposure to which variance is inversely proportional.

        ddof : int, optional (default=1)
            Degrees of freedom consumed by the model for ``mu``.

        method = {'pearson', 'deviance'}, optional (default='pearson')
            Whether to base the estimate on the Pearson residuals or the deviance.
        """
        y, mu, sample_weight = _as_float_arrays(y, mu, sample_weight)

        if method == "pearson":
            pearson_residuals = ((y - mu) ** 2) / self.unit_variance(mu)
            if sample_weight is None:
                numerator = pearson_residuals.sum()
            else:
                numerator = np.dot(pearson_residuals, sample_weight)
        elif method == "deviance":
            numerator = self.deviance(y, mu, sample_weight)
        else:
            raise NotImplementedError(f"Method {method} hasn't been implemented.")

        if sample_weight is None:
            return numerator / (len(y) - ddof)
        else:
            return numerator / (sample_weight.sum() - ddof)


class TweedieDistribution(ExponentialDispersionModel):
    r"""A class for the Tweedie distribution.

    A Tweedie distribution with mean :math:`\mu = \mathrm{E}(Y)` is uniquely
    defined by its mean-variance relationship
    :math:`\mathrm{var}(Y) \propto \mu^{\mathrm{p}}`.

    Special cases are:

    ====== ================ ============
    Power  Distribution     Support
    ====== ================ ============
    0      Normal           ``(-∞, +∞)``
    1      Poisson          ``[0, +∞)``
    (1, 2) Compound Poisson ``[0, +∞)``
    2      Gamma            ``(0, +∞)``
    3      Inverse Gaussian ``(0, +∞)``
    ====== ================ ============

    See the documentation of the superclass,
    :class:`~glum.ExponentialDispersionModel`, for details.

    Parameters
    ----------
    power : float, optional (default=0)
        The variance power of the ``unit_variance``
        :math:`v(\mu) = \mu^{\mathrm{power}}`. For
        :math:`0 < \mathrm{power} < 1`, no distribution exists.
    """

    upper_bound = np.inf
    include_upper_bound = False

    def __init__(self, power=0):
        # validate power and set _upper_bound, _include_upper_bound attrs
        self.power = power

    def __eq__(self, other):  # noqa D
        return isinstance(other, TweedieDistribution) and (self.power == other.power)

    def __tweedie_repr__(self):  # noqa D
        return self.__class__(self.power)

    @property
    def lower_bound(self) -> float:  # noqa D
        if self.power <= 0:
            return -np.inf
        if self.power >= 1:
            return 0
        raise ValueError

    @property
    def include_lower_bound(self) -> bool:  # noqa D
        if self.power <= 0:
            return False
        if (self.power >= 1) and (self.power < 2):
            return True
        if self.power >= 2:
            return False
        raise ValueError

    @property
    def power(self):
        """Return the Tweedie power parameter."""
        return self._power

    @power.setter
    def power(self, power):
        if not isinstance(power, (int, float, np.number)):
            raise TypeError(f"The power parameter must be numeric; got {power}.")
        if (power > 0) and (power < 1):
            raise ValueError("For `0 < p < 1`, no distribution exists.")

        # Prevents upcasting when working with 32-bit data
        self._power = power if isinstance(power, int) else np.float32(power)

    def unit_variance(self, mu):  # noqa D
        p = self.power  # noqa: F841
        return numexpr.evaluate("mu ** p")

    def unit_variance_derivative(self, mu):  # noqa D
        p = self.power  # noqa: F841
        return numexpr.evaluate("p * mu ** (p - 1)")

    def deviance(self, y, mu, sample_weight=None) -> float:  # noqa D
        y, mu, sample_weight = _as_float_arrays(y, mu, sample_weight)
        sample_weight = np.ones_like(y) if sample_weight is None else sample_weight

        # NOTE: the dispersion parameter is only necessary to convey
        # type information on account of a bug in Cython

        if self.power == 0:
            return normal_deviance(y, sample_weight, mu, dispersion=1.0)
        if self.power == 1:
            return poisson_deviance(y, sample_weight, mu, dispersion=1.0)
        elif self.power == 2:
            return gamma_deviance(y, sample_weight, mu, dispersion=1.0)
        elif self.power == 3:
            return inv_gaussian_deviance(y, sample_weight, mu, dispersion=1.0)
        else:
            return tweedie_deviance(y, sample_weight, mu, p=float(self.power))

    def unit_deviance(self, y, mu):  # noqa D
        if self.power == 0:  # normal distribution
            return (y - mu) ** 2
        if self.power == 1:  # Poisson distribution
            return 2 * (special.xlogy(y, y / mu) - y + mu)
        elif self.power == 2:  # Gamma distribution
            return 2 * (np.log(mu / y) + y / mu - 1)
        elif self.power == 3:  # inverse Gaussian distribution
            return ((y / mu - 1) ** 2) / y
        else:
            mu1mp = mu ** (1 - self.power)
            return 2 * (
                (np.maximum(y, 0) ** (2 - self.power))
                / ((1 - self.power) * (2 - self.power))
                - y * mu1mp / (1 - self.power)
                + mu * mu1mp / (2 - self.power)
            )

    def _rowwise_gradient_hessian(
        self, link, y, sample_weight, eta, mu, gradient_rows, hessian_rows
    ):
        f = None

        if self.power == 0 and isinstance(link, IdentityLink):
            f = normal_identity_rowwise_gradient_hessian
        elif self.power == 1 and isinstance(link, LogLink):
            f = poisson_log_rowwise_gradient_hessian
        elif self.power == 2 and isinstance(link, LogLink):
            f = gamma_log_rowwise_gradient_hessian
        elif 1 < self.power < 2 and isinstance(link, LogLink):
            f = partial(tweedie_log_rowwise_gradient_hessian, p=self.power)
        elif self.power == 3 and isinstance(link, LogLink):
            f = inv_gaussian_log_rowwise_gradient_hessian

        if f is not None:
            return f(y, sample_weight, eta, mu, gradient_rows, hessian_rows)

        return super()._rowwise_gradient_hessian(
            link, y, sample_weight, eta, mu, gradient_rows, hessian_rows
        )

    def _eta_mu_deviance(
        self,
        link: Link,
        factor: float,
        cur_eta: np.ndarray,
        X_dot_d: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray,
        eta_out: np.ndarray,
        mu_out: np.ndarray,
    ):
        f = None

        if self.power == 0 and isinstance(link, IdentityLink):
            f = normal_identity_eta_mu_deviance
        elif self.power == 1 and isinstance(link, LogLink):
            f = poisson_log_eta_mu_deviance
        elif self.power == 2 and isinstance(link, LogLink):
            f = gamma_log_eta_mu_deviance
        elif 1 < self.power < 2 and isinstance(link, LogLink):
            f = partial(tweedie_log_eta_mu_deviance, p=self.power)
        elif self.power == 3 and isinstance(link, LogLink):
            f = inv_gaussian_log_eta_mu_deviance

        if f is not None:
            return f(cur_eta, X_dot_d, y, sample_weight, eta_out, mu_out, factor)

        return super()._eta_mu_deviance(
            link, factor, cur_eta, X_dot_d, y, sample_weight, eta_out, mu_out
        )

    def log_likelihood(self, y, mu, sample_weight=None, dispersion=None) -> float:
        r"""Compute the log likelihood.

        For ``1 < p < 2``, we use the series approximation by Dunn and Smyth
        (2005) to compute the normalization term.

        Parameters
        ----------
        y : array-like, shape (n_samples,)
            Target values.

        mu : array-like, shape (n_samples,)
            Predicted mean.

        sample_weight : array-like, shape (n_samples,), optional (default=1)
            Sample weights.

        dispersion : float, optional (default=None)
            Dispersion parameter :math:`\phi`. Estimated if ``None``.
        """
        p = self.power
        y, mu, sample_weight = _as_float_arrays(y, mu, sample_weight)
        sample_weight = np.ones_like(y) if sample_weight is None else sample_weight

        if (p != 1) and (dispersion is None):
            dispersion = self.dispersion(y, mu, sample_weight)

        if p == 0:
            return normal_log_likelihood(y, sample_weight, mu, float(dispersion))
        if p == 1:
            # NOTE: the dispersion parameter is only necessary to convey
            # type information on account of a bug in Cython
            return poisson_log_likelihood(y, sample_weight, mu, 1.0)
        elif p == 2:
            return gamma_log_likelihood(y, sample_weight, mu, float(dispersion))
        elif p < 2:
            return tweedie_log_likelihood(
                y, sample_weight, mu, float(p), float(dispersion)
            )
        elif p == 3:
            return inv_gaussian_log_likelihood(y, sample_weight, mu, float(dispersion))
        else:
            raise NotImplementedError

    def dispersion(  # noqa D
        self, y, mu, sample_weight=None, ddof=1, method="pearson"
    ) -> float:
        p = self.power  # noqa: F841
        y, mu, sample_weight = _as_float_arrays(y, mu, sample_weight)

        if method == "pearson":
            formula = "((y - mu) ** 2) / (mu ** p)"
            if sample_weight is None:
                return numexpr.evaluate(formula).sum() / (len(y) - ddof)
            else:
                formula = f"sample_weight * {formula}"
                return numexpr.evaluate(formula).sum() / (sample_weight.sum() - ddof)

        return super().dispersion(
            y, mu, sample_weight=sample_weight, ddof=ddof, method=method
        )


class NormalDistribution(ExponentialDispersionModel):
    """Class for the normal (a.k.a. Gaussian) distribution.

    The normal distribution models outcomes ``y`` in ``(-∞, +∞)``.

    See the documentation of the superclass,
    :class:`~glum.ExponentialDispersionModel`, for details.
    """

    lower_bound = -np.inf
    upper_bound = np.inf
    include_lower_bound = False
    include_upper_bound = False

    def __eq__(self, other):  # noqa D
        return isinstance(other, self.__class__)

    def __tweedie_repr__(self):  # noqa D
        return TweedieDistribution(0)

    def unit_variance(self, mu) -> np.ndarray:  # noqa D
        return 1 if np.isscalar(mu) else np.ones_like(mu)  # type: ignore

    def unit_variance_derivative(self, mu) -> np.ndarray:  # noqa D
        return 0 if np.isscalar(mu) else np.zeros_like(mu)  # type: ignore

    def deviance(self, y, mu, sample_weight=None) -> float:  # noqa D
        y, mu, sample_weight = _as_float_arrays(y, mu, sample_weight)
        sample_weight = np.ones_like(y) if sample_weight is None else sample_weight

        # NOTE: the dispersion parameter is only necessary to convey
        # type information on account of a bug in Cython

        return normal_deviance(y, sample_weight, mu, dispersion=1.0)

    def unit_deviance(self, y, mu):  # noqa D
        return (y - mu) ** 2

    def _rowwise_gradient_hessian(
        self, link, y, sample_weight, eta, mu, gradient_rows, hessian_rows
    ):
        if isinstance(link, IdentityLink):
            return normal_identity_rowwise_gradient_hessian(
                y, sample_weight, eta, mu, gradient_rows, hessian_rows
            )

        return super()._rowwise_gradient_hessian(
            link, y, sample_weight, eta, mu, gradient_rows, hessian_rows
        )

    def _eta_mu_deviance(
        self,
        link: Link,
        factor: float,
        cur_eta,
        X_dot_d,
        y,
        sample_weight,
        eta_out,
        mu_out,
    ):
        if isinstance(link, IdentityLink):
            return normal_identity_eta_mu_deviance(
                cur_eta, X_dot_d, y, sample_weight, eta_out, mu_out, factor
            )

        return super()._eta_mu_deviance(
            link, factor, cur_eta, X_dot_d, y, sample_weight, eta_out, mu_out
        )

    def log_likelihood(self, y, mu, sample_weight=None, dispersion=None) -> float:
        r"""Compute the log likelihood.

        Parameters
        ----------
        y : array-like, shape (n_samples,)
            Target values.

        mu : array-like, shape (n_samples,)
            Predicted mean.

        sample_weight : array-like, shape (n_samples,), optional (default=1)
            Sample weights.

        dispersion : float, optional (default=None)
            Dispersion parameter :math:`\phi`. Estimated if ``None``.
        """
        y, mu, sample_weight = _as_float_arrays(y, mu, sample_weight)
        sample_weight = np.ones_like(y) if sample_weight is None else sample_weight

        if dispersion is None:
            dispersion = self.dispersion(y, mu, sample_weight)

        return normal_log_likelihood(y, sample_weight, mu, float(dispersion))

    def dispersion(  # noqa D
        self, y, mu, sample_weight=None, ddof=1, method="pearson"
    ) -> float:
        y, mu, sample_weight = _as_float_arrays(y, mu, sample_weight)

        if method == "pearson":
            formula = "(y - mu) ** 2"
            if sample_weight is None:
                return numexpr.evaluate(formula).sum() / (len(y) - ddof)
            else:
                formula = f"sample_weight * {formula}"
                return numexpr.evaluate(formula).sum() / (sample_weight.sum() - ddof)

        return super().dispersion(
            y, mu, sample_weight=sample_weight, ddof=ddof, method=method
        )


class PoissonDistribution(ExponentialDispersionModel):
    """Class for the scaled Poisson distribution.

    The Poisson distribution models discrete outcomes ``y`` in ``[0, +∞)``.

    See the documentation of the superclass,
    :class:`~glum.ExponentialDispersionModel`, for details.
    """

    lower_bound = 0
    upper_bound = np.inf
    include_lower_bound = True
    include_upper_bound = False

    def __eq__(self, other):  # noqa D
        return isinstance(other, self.__class__)

    def __tweedie_repr__(self):  # noqa D
        return TweedieDistribution(1)

    def unit_variance(self, mu) -> np.ndarray:  # noqa D
        return mu

    def unit_variance_derivative(self, mu) -> np.ndarray:  # noqa D
        return 1.0 if np.isscalar(mu) else np.ones_like(mu)  # type: ignore

    def deviance(self, y, mu, sample_weight=None) -> float:  # noqa D
        y, mu, sample_weight = _as_float_arrays(y, mu, sample_weight)
        sample_weight = np.ones_like(y) if sample_weight is None else sample_weight

        # NOTE: the dispersion parameter is only necessary to convey
        # type information on account of a bug in Cython

        return poisson_deviance(y, sample_weight, mu, dispersion=1.0)

    def unit_deviance(self, y, mu):
        """Compute the unit deviance."""
        return 2 * (special.xlogy(y, y / mu) - y + mu)

    def _rowwise_gradient_hessian(
        self, link, y, sample_weight, eta, mu, gradient_rows, hessian_rows
    ):
        if isinstance(link, LogLink):
            return poisson_log_rowwise_gradient_hessian(
                y, sample_weight, eta, mu, gradient_rows, hessian_rows
            )

        return super()._rowwise_gradient_hessian(
            link, y, sample_weight, eta, mu, gradient_rows, hessian_rows
        )

    def _eta_mu_deviance(
        self,
        link: Link,
        factor: float,
        cur_eta,
        X_dot_d,
        y,
        sample_weight,
        eta_out,
        mu_out,
    ):
        if isinstance(link, LogLink):
            return poisson_log_eta_mu_deviance(
                cur_eta, X_dot_d, y, sample_weight, eta_out, mu_out, factor
            )

        return super()._eta_mu_deviance(
            link, factor, cur_eta, X_dot_d, y, sample_weight, eta_out, mu_out
        )

    def log_likelihood(self, y, mu, sample_weight=None, dispersion=None) -> float:
        r"""Compute the log likelihood.

        Parameters
        ----------
        y : array-like, shape (n_samples,)
            Target values.

        mu : array-like, shape (n_samples,)
            Predicted mean.

        sample_weight : array-like, shape (n_samples,), optional (default=1)
            Sample weights.

        dispersion : float, optional (default=None)
            Dispersion parameter :math:`\phi`. Estimated if ``None``.
        """
        y, mu, sample_weight = _as_float_arrays(y, mu, sample_weight)
        sample_weight = np.ones_like(y) if sample_weight is None else sample_weight

        # NOTE: the dispersion parameter is only necessary to convey
        # type information on account of a bug in Cython

        return poisson_log_likelihood(y, sample_weight, mu, 1.0)

    def dispersion(  # noqa D
        self, y, mu, sample_weight=None, ddof=1, method="pearson"
    ) -> float:
        y, mu, sample_weight = _as_float_arrays(y, mu, sample_weight)

        if method == "pearson":
            formula = "((y - mu) ** 2) / mu"
            if sample_weight is None:
                return numexpr.evaluate(formula).sum() / (len(y) - ddof)
            else:
                formula = f"sample_weight * {formula}"
                return numexpr.evaluate(formula).sum() / (sample_weight.sum() - ddof)

        return super().dispersion(
            y, mu, sample_weight=sample_weight, ddof=ddof, method=method
        )


class GammaDistribution(ExponentialDispersionModel):
    """Class for the gamma distribution.

    The gamma distribution models outcomes ``y`` in ``(0, +∞)``.

    See the documentation of the superclass,
    :class:`~glum.ExponentialDispersionModel`, for details.
    """

    lower_bound = 0
    upper_bound = np.inf
    include_lower_bound = False
    include_upper_bound = False

    def __eq__(self, other):  # noqa D
        return isinstance(other, self.__class__)

    def __tweedie_repr__(self):  # noqa D
        return TweedieDistribution(2)

    def unit_variance(self, mu) -> np.ndarray:  # noqa D
        return mu**2

    def unit_variance_derivative(self, mu) -> np.ndarray:  # noqa D
        return 2 * mu

    def deviance(self, y, mu, sample_weight=None) -> float:  # noqa D
        y, mu, sample_weight = _as_float_arrays(y, mu, sample_weight)
        sample_weight = np.ones_like(y) if sample_weight is None else sample_weight

        # NOTE: the dispersion parameter is only necessary to convey
        # type information on account of a bug in Cython

        return gamma_deviance(y, sample_weight, mu, dispersion=1.0)

    def unit_deviance(self, y, mu):  # noqa D
        return 2 * (np.log(mu / y) + y / mu - 1)

    def _rowwise_gradient_hessian(
        self, link, y, sample_weight, eta, mu, gradient_rows, hessian_rows
    ):
        if isinstance(link, LogLink):
            return gamma_log_rowwise_gradient_hessian(
                y, sample_weight, eta, mu, gradient_rows, hessian_rows
            )

        return super()._rowwise_gradient_hessian(
            link, y, sample_weight, eta, mu, gradient_rows, hessian_rows
        )

    def _eta_mu_deviance(
        self,
        link: Link,
        factor: float,
        cur_eta,
        X_dot_d,
        y,
        sample_weight,
        eta_out,
        mu_out,
    ):
        if isinstance(link, LogLink):
            return gamma_log_eta_mu_deviance(
                cur_eta, X_dot_d, y, sample_weight, eta_out, mu_out, factor
            )

        return super()._eta_mu_deviance(
            link, factor, cur_eta, X_dot_d, y, sample_weight, eta_out, mu_out
        )

    def log_likelihood(self, y, mu, sample_weight=None, dispersion=None) -> float:
        r"""Compute the log likelihood.

        Parameters
        ----------
        y : array-like, shape (n_samples,)
            Target values.

        mu : array-like, shape (n_samples,)
            Predicted mean.

        sample_weight : array-like, shape (n_samples,), optional (default=1)
            Sample weights.

        dispersion : float, optional (default=None)
            Dispersion parameter :math:`\phi`. Estimated if ``None``.
        """
        y, mu, sample_weight = _as_float_arrays(y, mu, sample_weight)
        sample_weight = np.ones_like(y) if sample_weight is None else sample_weight

        if dispersion is None:
            dispersion = self.dispersion(y, mu, sample_weight)

        return gamma_log_likelihood(y, sample_weight, mu, float(dispersion))

    def dispersion(  # noqa D
        self, y, mu, sample_weight=None, ddof=1, method="pearson"
    ) -> float:
        y, mu, sample_weight = _as_float_arrays(y, mu, sample_weight)

        if method == "pearson":
            formula = "((y - mu) ** 2) / (mu ** 2)"
            if sample_weight is None:
                return numexpr.evaluate(formula).sum() / (len(y) - ddof)
            else:
                formula = f"sample_weight * {formula}"
                return numexpr.evaluate(formula).sum() / (sample_weight.sum() - ddof)

        return super().dispersion(
            y, mu, sample_weight=sample_weight, ddof=ddof, method=method
        )


class InverseGaussianDistribution(ExponentialDispersionModel):
    """Class for the inverse Gaussian distribution.

    The inverse Gaussian distribution models outcomes ``y`` in ``(0, +∞)``.

    See the documentation of the superclass,
    :class:`~glum.ExponentialDispersionModel`, for details.
    """

    lower_bound = 0
    upper_bound = np.inf
    include_lower_bound = False
    include_upper_bound = False

    def __eq__(self, other):  # noqa D
        return isinstance(other, self.__class__)

    def __tweedie_repr__(self):  # noqa D
        return TweedieDistribution(3)

    def unit_variance(self, mu) -> np.ndarray:  # noqa D
        return mu**3

    def unit_variance_derivative(self, mu) -> np.ndarray:  # noqa D
        return 3 * (mu**2)

    def deviance(self, y, mu, sample_weight=None) -> float:  # noqa D
        y, mu, sample_weight = _as_float_arrays(y, mu, sample_weight)
        sample_weight = np.ones_like(y) if sample_weight is None else sample_weight

        return tweedie_deviance(y, sample_weight, mu, p=3.0)

    def unit_deviance(self, y, mu):  # noqa D
        return numexpr.evaluate("y / (mu**2) + 1 / y - 2 / mu")

    def _rowwise_gradient_hessian(
        self, link, y, sample_weight, eta, mu, gradient_rows, hessian_rows
    ):
        if isinstance(link, LogLink):
            return inv_gaussian_log_rowwise_gradient_hessian(
                y, sample_weight, eta, mu, gradient_rows, hessian_rows
            )

        return super()._rowwise_gradient_hessian(
            link, y, sample_weight, eta, mu, gradient_rows, hessian_rows
        )

    def _eta_mu_deviance(
        self,
        link: Link,
        factor: float,
        cur_eta,
        X_dot_d,
        y,
        sample_weight,
        eta_out,
        mu_out,
    ):
        if isinstance(link, LogLink):
            return inv_gaussian_log_eta_mu_deviance(
                cur_eta, X_dot_d, y, sample_weight, eta_out, mu_out, factor
            )

        return super()._eta_mu_deviance(
            link, factor, cur_eta, X_dot_d, y, sample_weight, eta_out, mu_out
        )

    def log_likelihood(self, y, mu, sample_weight=None, dispersion=None) -> float:
        r"""Compute the log likelihood.

        Parameters
        ----------
        y : array-like, shape (n_samples,)
            Target values.

        mu : array-like, shape (n_samples,)
            Predicted mean.

        sample_weight : array-like, shape (n_samples,), optional (default=1)
            Sample weights.

        dispersion : float, optional (default=None)
            Dispersion parameter :math:`\phi`. Estimated if ``None``.
        """
        y, mu, sample_weight = _as_float_arrays(y, mu, sample_weight)
        sample_weight = np.ones_like(y) if sample_weight is None else sample_weight

        if dispersion is None:
            dispersion = self.dispersion(y, mu, sample_weight)

        return tweedie_log_likelihood(y, sample_weight, mu, 3.0, float(dispersion))

    def dispersion(  # noqa D
        self, y, mu, sample_weight=None, ddof=1, method="pearson"
    ) -> float:
        y, mu, sample_weight = _as_float_arrays(y, mu, sample_weight)

        if method == "pearson":
            formula = "((y - mu) ** 2) / (mu ** 3)"
            if sample_weight is None:
                return numexpr.evaluate(formula).sum() / (len(y) - ddof)
            else:
                formula = f"sample_weight * {formula}"
                return numexpr.evaluate(formula).sum() / (sample_weight.sum() - ddof)

        return super().dispersion(
            y, mu, sample_weight=sample_weight, ddof=ddof, method=method
        )


class GeneralizedHyperbolicSecant(ExponentialDispersionModel):
    """A class for the Generalized Hyperbolic Secant (GHS) distribution.

    The GHS distribution models outcomes ``y`` in ``(-∞, +∞)``.

    See the documentation of the superclass,
    :class:`~glum.ExponentialDispersionModel`, for details.
    """

    lower_bound = -np.inf
    upper_bound = np.inf
    include_lower_bound = False
    include_upper_bound = False

    def __eq__(self, other):  # noqa D
        return isinstance(other, self.__class__)

    def unit_variance(self, mu) -> np.ndarray:  # noqa D
        return 1 + mu**2

    def unit_variance_derivative(self, mu) -> np.ndarray:  # noqa D
        return 2 * mu

    def unit_deviance(self, y, mu) -> np.ndarray:  # noqa D
        return 2 * y * (np.arctan(y) - np.arctan(mu)) + np.log((1 + mu**2) / (1 + y**2))


class BinomialDistribution(ExponentialDispersionModel):
    """A class for the Binomial distribution.

    The Binomial distribution models outcomes ``y`` in ``[0, 1]``.

    See the documentation of the superclass,
    :class:`~glum.ExponentialDispersionModel`, for details.
    """

    lower_bound = 0
    upper_bound = 1
    include_lower_bound = True
    include_upper_bound = True

    def __eq__(self, other):  # noqa D
        return isinstance(other, self.__class__)

    def unit_variance(self, mu):  # noqa D
        return mu * (1 - mu)

    def unit_variance_derivative(self, mu):  # noqa D
        return 1 - 2 * mu

    def unit_deviance(self, y, mu):  # noqa D
        # see Wooldridge and Papke (1996) for the fractional case
        return -2 * (special.xlogy(y, mu) + special.xlogy(1 - y, 1 - mu))

    def _rowwise_gradient_hessian(
        self, link, y, sample_weight, eta, mu, gradient_rows, hessian_rows
    ):
        if isinstance(link, LogitLink):
            return binomial_logit_rowwise_gradient_hessian(
                y, sample_weight, eta, mu, gradient_rows, hessian_rows
            )
        return super()._rowwise_gradient_hessian(
            link, y, sample_weight, eta, mu, gradient_rows, hessian_rows
        )

    def _eta_mu_deviance(
        self,
        link: Link,
        factor: float,
        cur_eta,
        X_dot_d,
        y,
        sample_weight,
        eta_out,
        mu_out,
    ):
        if isinstance(link, LogitLink):
            return binomial_logit_eta_mu_deviance(
                cur_eta, X_dot_d, y, sample_weight, eta_out, mu_out, factor
            )

        return super()._eta_mu_deviance(
            link, factor, cur_eta, X_dot_d, y, sample_weight, eta_out, mu_out
        )

    def log_likelihood(self, y, mu, sample_weight=None, dispersion=1) -> float:
        """Compute the log likelihood.

        Parameters
        ----------
        y : array-like, shape (n_samples,)
            Target values.

        mu : array-like, shape (n_samples,)
            Predicted mean.

        sample_weight : array-like, shape (n_samples,), optional (default=1)
            Sample weights.

        dispersion : float, optional (default=1)
            Ignored.
        """
        ll = special.xlogy(y, mu) + special.xlogy(1 - y, 1 - mu)
        return np.sum(ll) if sample_weight is None else np.dot(ll, sample_weight)

    def dispersion(  # noqa D
        self, y, mu, sample_weight=None, ddof=1, method="pearson"
    ) -> float:
        y, mu, sample_weight = _as_float_arrays(y, mu, sample_weight)

        if method == "pearson":
            formula = "((y - mu) ** 2) / (mu * (1 - mu))"
            if sample_weight is None:
                return numexpr.evaluate(formula).sum() / (len(y) - ddof)
            else:
                formula = f"sample_weight * {formula}"
                return numexpr.evaluate(formula).sum() / (sample_weight.sum() - ddof)

        return super().dispersion(
            y, mu, sample_weight=sample_weight, ddof=ddof, method=method
        )


class NegativeBinomialDistribution(ExponentialDispersionModel):
    r"""A class for the Negative Binomial distribution.

    A negative binomial distribution with mean :math:`\mu = \mathrm{E}(Y)` is
    uniquely defined by its mean-variance relationship
    :math:`\mathrm{var}(Y) \propto \mu + \theta * \mu^2`.

    Parameters
    ----------
    theta : float, optional (default=1.0)
        The dispersion parameter from the ``unit_variance``
        :math:`v(\mu) = \mu + \theta * \mu^2`. For
        :math:`\theta <= 0`, no distribution exists.

    References
    ----------
    For the log-likelihood and deviance:
        * M. L. Zwilling Negative Binomial Regression, The Mathematica Journal 2013.
          https://www.mathematica-journal.com/2013/06/27/negative-binomial-regression/
    """

    lower_bound = 0
    upper_bound = np.inf
    include_lower_bound = True
    include_upper_bound = False

    def __init__(self, theta=1.0):
        # validate power and set _upper_bound, _include_upper_bound attrs
        self.theta = theta

    def __eq__(self, other):  # noqa D
        return isinstance(other, NegativeBinomialDistribution) and (
            self.theta == other.theta
        )

    @property
    def theta(self):
        """Return the negative binomial theta parameter."""
        return self._theta

    @theta.setter
    def theta(self, theta):
        if not isinstance(theta, (int, float)):
            raise TypeError(f"Theta must be numeric; got {theta}.")
        if not theta > 0:
            raise ValueError(f"Theta must be strictly positive; got was {theta}.")

        # Prevents upcasting when working with 32-bit data
        self._theta = theta if isinstance(theta, int) else np.float32(theta)

    def unit_variance(self, mu):  # noqa D
        return mu + self.theta * mu**2

    def unit_variance_derivative(self, mu):  # noqa D
        return 1 + 2 * self.theta * mu

    def deviance(self, y, mu, sample_weight=None) -> float:  # noqa D
        theta = self.theta
        y, mu, sample_weight = _as_float_arrays(y, mu, sample_weight)
        sample_weight = np.ones_like(y) if sample_weight is None else sample_weight

        return negative_binomial_deviance(y, sample_weight, mu, theta=float(theta))

    def unit_deviance(self, y, mu):  # noqa D
        theta = self.theta

        r = 1.0 / theta

        return 2 * (special.xlogy(y, y / mu) - (y + r) * np.log((y + r) / (mu + r)))

    def _rowwise_gradient_hessian(
        self, link, y, sample_weight, eta, mu, gradient_rows, hessian_rows
    ):
        if isinstance(link, LogLink):
            return negative_binomial_log_rowwise_gradient_hessian(
                y, sample_weight, eta, mu, gradient_rows, hessian_rows, theta=self.theta
            )
        return super()._rowwise_gradient_hessian(
            link, y, sample_weight, eta, mu, gradient_rows, hessian_rows
        )

    def _eta_mu_deviance(
        self,
        link: Link,
        factor: float,
        cur_eta,
        X_dot_d,
        y,
        sample_weight,
        eta_out,
        mu_out,
    ):
        if isinstance(link, LogLink):
            return negative_binomial_log_eta_mu_deviance(
                cur_eta,
                X_dot_d,
                y,
                sample_weight,
                eta_out,
                mu_out,
                factor,
                theta=self.theta,
            )
        return super()._eta_mu_deviance(
            link, factor, cur_eta, X_dot_d, y, sample_weight, eta_out, mu_out
        )

    def log_likelihood(self, y, mu, sample_weight=None, dispersion=1) -> float:
        r"""Compute the log likelihood.

        Parameters
        ----------
        y : array-like, shape (n_samples,)
            Target values.

        mu : array-like, shape (n_samples,)
            Predicted mean.

        sample_weight : array-like, shape (n_samples,), optional (default=1)
            Sample weights.

        dispersion : float, optional (default=1.0)
            Ignored.
        """
        theta = self.theta
        y, mu, sample_weight = _as_float_arrays(y, mu, sample_weight)
        sample_weight = np.ones_like(y) if sample_weight is None else sample_weight

        return negative_binomial_log_likelihood(y, sample_weight, mu, float(theta), 1.0)

    def dispersion(  # noqa D
        self, y, mu, sample_weight=None, ddof=1, method="pearson"
    ) -> float:
        theta = self.theta  # noqa: F841
        y, mu, sample_weight = _as_float_arrays(y, mu, sample_weight)

        if method == "pearson":
            formula = "((y - mu) ** 2) / (mu + theta * mu ** 2)"
            if sample_weight is None:
                return numexpr.evaluate(formula).sum() / (len(y) - ddof)
            else:
                formula = f"sample_weight * {formula}"
                return numexpr.evaluate(formula).sum() / (sample_weight.sum() - ddof)

        return super().dispersion(
            y, mu, sample_weight=sample_weight, ddof=ddof, method=method
        )


def guess_intercept(
    y,
    sample_weight,
    link: Link,
    distribution: ExponentialDispersionModel,
    eta: Union[np.ndarray, float] = None,
):
    """
    Say we want to find the scalar `b` that minimizes ``LL(eta + b)``, with \
    ``eta`` fixed.

    An exact solution exists for Tweedie distributions with a log link and for
    the normal distribution with identity link. An exact solution also exists
    for the case of logit with no offset.

    If the distribution and corresponding link are something else, we use the
    Tweedie or normal solution, depending on the link function.
    """
    if (not isinstance(link, IdentityLink)) and (len(np.unique(y)) == 1):
        raise ValueError("No variation in `y`. Coefficients can't be estimated.")

    avg_y = np.average(y, weights=sample_weight)

    if isinstance(link, IdentityLink):
        # This is only correct for the normal. For other distributions, the
        # answer is unknown, but we assume that we want `sum(y) = sum(mu)`

        if eta is None:
            return avg_y

        avg_eta = eta if np.isscalar(eta) else np.average(eta, weights=sample_weight)

        return avg_y - avg_eta

    elif isinstance(link, LogLink):
        # This is only correct for Tweedie

        log_avg_y = np.log(avg_y)

        assert np.isfinite(log_avg_y).all()

        if eta is None:
            return log_avg_y

        mu = np.exp(eta)

        if isinstance(distribution, TweedieDistribution):
            p = distribution.power
        elif isinstance(distribution, NormalDistribution):
            p = 0
        elif isinstance(distribution, PoissonDistribution):
            p = 1
        elif isinstance(distribution, GammaDistribution):
            p = 2
        elif isinstance(distribution, InverseGaussianDistribution):
            p = 3
        else:
            p = 1  # Like Poisson

        if np.isscalar(mu):
            first = np.log(y.dot(sample_weight) * mu ** (1 - p))  # type: ignore
            second = np.log(sample_weight.sum() * mu ** (2 - p))  # type: ignore
        else:
            first = np.log((y * mu ** (1 - p)).dot(sample_weight))  # type: ignore
            second = np.log((mu ** (2 - p)).dot(sample_weight))  # type: ignore

        return first - second

    elif isinstance(link, LogitLink):
        log_odds = np.log(avg_y) - np.log(1 - avg_y)

        if eta is None:
            return log_odds

        avg_eta = eta if np.isscalar(eta) else np.average(eta, weights=sample_weight)

        return log_odds - avg_eta

    else:
        return link.link(y.dot(sample_weight))


def get_one_over_variance(
    distribution: ExponentialDispersionModel,
    link: Link,
    mu,
    eta,
    dispersion: float,
    sample_weight,
):
    """Get one over the variance.

    For Tweedie: ``sigma_inv = sample_weight / (mu ** p)`` during optimization,
    because ``phi = 1``.

    For Binomial with Logit link: Simplifies to
    ``variance = phi / ( sample_weight * (exp(eta) + 2 + exp(-eta)))``.
    More numerically accurate.
    """
    if isinstance(distribution, BinomialDistribution) and isinstance(link, LogitLink):
        max_float_for_exp = np.log(np.finfo(eta.dtype).max / 10)
        if np.any(np.abs(eta) > max_float_for_exp):
            eta = np.clip(eta, -max_float_for_exp, max_float_for_exp)  # type: ignore
        return sample_weight * (np.exp(eta) + 2 + np.exp(-eta)) / dispersion
    return 1.0 / distribution.variance(
        mu, dispersion=dispersion, sample_weight=sample_weight
    )


def _as_float_arrays(*args):
    """Convert to a float array, passing ``None`` through, and broadcast."""
    never_broadcast = {}  # type: ignore
    maybe_broadcast = {}
    always_broadcast = {}

    for ix, arg in enumerate(args):
        if isinstance(arg, (int, float)):
            maybe_broadcast[ix] = np.array([arg], dtype="float")
        elif arg is None:
            never_broadcast[ix] = None
        else:
            always_broadcast[ix] = np.asanyarray(arg, dtype="float")

    if always_broadcast and maybe_broadcast:
        to_broadcast = {**always_broadcast, **maybe_broadcast}
        _broadcast = np.broadcast_arrays(*to_broadcast.values())
        broadcast = dict(zip(to_broadcast.keys(), _broadcast))
    elif always_broadcast:
        _broadcast = np.broadcast_arrays(*always_broadcast.values())
        broadcast = dict(zip(always_broadcast.keys(), _broadcast))
    else:
        broadcast = maybe_broadcast  # possibly `{}`

    out = {**never_broadcast, **broadcast}

    return [out[ix] for ix in range(len(args))]
