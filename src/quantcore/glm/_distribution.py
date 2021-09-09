from abc import ABCMeta, abstractmethod
from functools import partial
from typing import Tuple, Union

import numexpr
import numpy as np
from quantcore.matrix import MatrixBase, StandardizedMatrix
from scipy import special

from ._functions import (
    binomial_logit_eta_mu_deviance,
    binomial_logit_rowwise_gradient_hessian,
    gamma_log_eta_mu_deviance,
    gamma_log_rowwise_gradient_hessian,
    normal_identity_eta_mu_deviance,
    normal_identity_rowwise_gradient_hessian,
    poisson_log_eta_mu_deviance,
    poisson_log_rowwise_gradient_hessian,
    tweedie_log_eta_mu_deviance,
    tweedie_log_rowwise_gradient_hessian,
)
from ._link import IdentityLink, Link, LogitLink, LogLink
from ._util import _safe_lin_pred, _safe_sandwich_dot


class ExponentialDispersionModel(metaclass=ABCMeta):
    r"""Base class for reproductive Exponential Dispersion Models (EDM).

    The PDF of :math:`Y \sim \mathrm{EDM}(\mu, \phi)` is given by

    .. math::

        p(y \mid \theta, \phi)
        &= c(y, \phi) \exp((\theta y - A(\theta)_ / \phi) \\
        &= \tilde{c}(y, \phi) \exp(-d(y, \mu) / (2\phi))

    with mean :math:`\mathrm{E}(Y) = A'(\theta) = \mu`, variance
    :math:`\mathrm{var}(Y) = \phi \cdot v(\mu)`, unit variance
    :math:`v(\mu)` and unit deviance :math:`d(y, \mu)`.

    Properties
    ----------
    lower_bound
    upper_bound
    include_lower_bound
    include_upper_bound

    Methods
    -------
    in_y_range
    unit_variance
    unit_variance_derivative
    variance
    variance_derivative
    unit_deviance
    unit_deviance_derivative
    deviance
    deviance_derivative
    starting_mu

    _mu_deviance_derivative
    eta_mu_deviance
    gradient_hessian

    References
    ----------
    https://en.wikipedia.org/wiki/Exponential_dispersion_model.
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
    def include_lower_bound(self) -> bool:
        """Return whether ``lower_bound`` is allowed as a value of ``y``."""
        pass

    @property
    def include_upper_bound(self) -> bool:
        """Return whether ``upper_bound`` is allowed as a value of ``y``."""
        pass

    def in_y_range(self, x) -> np.bool_:
        """Return ``True`` if ``x`` is in the valid range of the EDM.

        Parameters
        ----------
        x : array-like, shape (n_samples,)
            Target values.

        Returns
        -------
        np.bool_
        """
        if self.include_lower_bound:
            if self.include_upper_bound:
                return np.logical_and(
                    np.greater_equal(x, self.lower_bound),
                    np.less_equal(x, self.upper_bound),
                )
            else:
                return np.logical_and(
                    np.greater_equal(x, self.lower_bound), np.less(x, self.upper_bound)
                )
        else:
            if self.include_upper_bound:
                return np.logical_and(
                    np.greater(x, self.lower_bound), np.less_equal(x, self.upper_bound)
                )
            else:
                return np.logical_and(
                    np.greater(x, self.lower_bound), np.less(x, self.upper_bound)
                )

    @abstractmethod
    def unit_variance(self, mu):
        r"""Compute the unit variance function.

        The unit variance :math:`v(\mu)` determines the variance as a function
        of the mean :math:`\mu` by
        :math:`\mathrm{var}(y_i) = (\phi / s_i) \times v(\mu_i)`. It can
        also be derived from the unit deviance :math:`d(y, \mu)` as

        .. math::

            v(\mu) = \frac{2}{\frac{\partial^2 d(y, \mu)}{\partial\mu^2}}\big|_{y=\mu}.

        See also :func:`variance`.

        Parameters
        ----------
        mu : array-like, shape (n_samples,)
            Predicted mean.
        """
        pass

    @abstractmethod
    def unit_variance_derivative(self, mu):
        r"""Compute the derivative of the unit variance with respect to ``mu``.

        Return :math:`v'(\mu)`.

        Parameters
        ----------
        mu : array-like, shape (n_samples,)
            Predicted mean.
        """
        pass

    def variance(self, mu: np.ndarray, phi=1, weights=1) -> np.ndarray:
        r"""Compute the variance function.

        The variance of :math:`Y_i \sim \mathrm{EDM}(\mu_i, \phi / s_i)` is
        :math:`\mathrm{var}(Y_i) = (\phi / s_i) * v(\mu_i)`, with unit variance
        :math:`v(\mu)` and weights :math:`s_i`.

        Parameters
        ----------
        mu : array-like, shape (n_samples,)
            Predicted mean.

        phi : float, optional (default=1)
            Dispersion parameter.

        weights : array-like, shape (n_samples,), optional (default=1)
            Weights or exposure to which variance is inverse proportional.

        Returns
        -------
        array-like, shape (n_samples,)
        """
        return phi / weights * self.unit_variance(mu)

    def variance_derivative(self, mu, phi=1, weights=1):
        r"""Compute the derivative of the variance with respect to ``mu``.

        The derivative of the variance is equal to
        :math:`(phi / s_i) * v'(\mu_i)`, where :math:`v(\mu)` is the unit
        variance and :math:`s_i` are weights.

        Parameters
        ----------
        mu : array-like, shape (n_samples,)
            Predicted mean.

        phi : float, optional (default=1)
            Dispersion parameter.

        weights : array-like, shape (n_samples,), optional (default=1)
            Weights or exposure to which variance is inverse proportional.

        Returns
        -------
        array-like, shape (n_samples,)
        """
        return phi / weights * self.unit_variance_derivative(mu)

    @abstractmethod
    def unit_deviance(self, y, mu):
        r"""Compute the unit deviance.

        In terms of the log likelihood :math:`L`, the unit deviance is
        :math:`-2\phi\times [L(y, \mu, \phi) - L(y, y, \phi)].`

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
        :math:`-2 \times (y - \mu) / v(\mu)`, where :math:`v(\mu)` is the unit
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

    def deviance(self, y, mu, weights=1):
        r"""Compute the deviance.

        The deviance is a weighted sum of the unit deviances,
        :math:`\sum_i s_i \times d(y_i, \mu_i)`, where :math:`d(y, \mu)` is the
        unit deviance and :math:`s` are weights. In terms of the log likelihood,
        it is :math:`-2\phi \times [L(y, \mu, \phi / s) - L(y, y, \phi / s)]`.

        Parameters
        ----------
        y : array-like, shape (n_samples,)
            Target values.

        mu : array-like, shape (n_samples,)
            Predicted mean.

        weights : array-like, shape (n_samples,), optional (default=1)
            Weights or exposure to which variance is inversely proportional.

        Returns
        -------
        float
        """
        return np.sum(weights * self.unit_deviance(y, mu))

    def deviance_derivative(self, y, mu, weights=1):
        r"""Compute the derivative of the deviance with respect to ``mu``.

        It gives :math:`\frac{\partial}{\partial\mu} D(y, \mu; weights)`.

        Parameters
        ----------
        y : array-like, shape (n_samples,)
            Target values.

        mu : array-like, shape (n_samples,)
            Predicted mean.

        weights : array-like, shape (n_samples,) (default=1)
            Weights or exposure to which variance is inverse proportional.

        Returns
        -------
        array-like, shape (n_samples,)
        """
        return weights * self.unit_deviance_derivative(y, mu)

    def _mu_deviance_derivative(
        self,
        coef: np.ndarray,
        X,
        y: np.ndarray,
        weights: np.ndarray,
        link: Link,
        offset: np.ndarray = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute ``mu`` and the derivative of the deviance \
            with respect to coefficients."""
        lin_pred = _safe_lin_pred(X, coef, offset)
        mu = link.inverse(lin_pred)
        d1 = link.inverse_derivative(lin_pred)
        temp = d1 * self.deviance_derivative(y, mu, weights)
        if coef.size == X.shape[1] + 1:
            devp = np.concatenate(([temp.sum()], temp @ X))
        else:
            devp = temp @ X  # same as X.T @ temp
        return mu, devp

    def eta_mu_deviance(
        self,
        link: Link,
        factor: float,
        cur_eta: np.ndarray,
        X_dot_d: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
    ):
        """
        Compute ``eta``, ``mu`` and the deviance.

        Compute:
        * the linear predictor, ``eta``, as ``cur_eta + factor * X_dot_d``;
        * the link-function-transformed prediction, ``mu``;
        * the deviance.

        Returns
        -------
        numpy.ndarray, shape (X.shape[0],)
            The linear predictor, ``eta``.
        numpy.ndarray, shape (X.shape[0],)
            The link-function-transformed prediction, ``mu``.
        float
            The deviance.
        """
        eta_out = np.empty_like(cur_eta)
        mu_out = np.empty_like(cur_eta)
        # Note: eta_out and mu_out are filled inside self._eta_mu_deviance.
        # This will be useful in the future to avoid allocating new eta/mu
        # arrays for every line search loop.
        return (
            eta_out,
            mu_out,
            self._eta_mu_deviance(
                link, factor, cur_eta, X_dot_d, y, weights, eta_out, mu_out
            ),
        )

    def _eta_mu_deviance(
        self,
        link: Link,
        factor: float,
        cur_eta: np.ndarray,
        X_dot_d: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        eta_out: np.ndarray,
        mu_out: np.ndarray,
    ):
        """
        Update ``eta`` and ``mu`` and compute the deviance.

        This is a default implementation that should work for all valid
        distributions and link functions. To implement a custom optimized
        version for a specific distribution and link function, please override
        this function in the subclass.

        Returns
        -------
        float
        """
        eta_out[:] = cur_eta + factor * X_dot_d
        mu_out[:] = link.inverse(eta_out)
        return self.deviance(y, mu_out, weights=weights)

    def rowwise_gradient_hessian(
        self,
        link: Link,
        coef: np.ndarray,
        phi,
        X: Union[MatrixBase, StandardizedMatrix],
        y: np.ndarray,
        weights: np.ndarray,
        eta: np.ndarray,
        mu: np.ndarray,
        offset: np.ndarray = None,
    ):
        """
        Compute the gradient and negative Hessian of the log likelihood row-wise.

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
            link, y, weights, eta, mu, gradient_rows, hessian_rows
        )

        # To form the full Hessian matrix from the IRLS weights:
        # hessian_matrix = _safe_sandwich_dot(X, hessian_rows, intercept=intercept)
        return gradient_rows, hessian_rows

    def _rowwise_gradient_hessian(
        self, link, y, weights, eta, mu, gradient_rows, hessian_rows
    ):
        """
        Update ``gradient_rows`` and ``hessian_rows`` in place.

        This is a default implementation that should work for all valid
        distributions and link functions. To implement a custom optimized
        version for a specific distribution and link function, please override
        this function in the subclass.
        """
        # FOR TWEEDIE: sigma_inv = weights / (mu ** p) during optimization bc phi = 1
        sigma_inv = get_one_over_variance(self, link, mu, eta, 1.0, weights)
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
            A link function (i.e. an instance of :class:`~quantcore.glm._link.Link`).
        X : pandas.DataFrame
            Training data.
        y : array-like
            Target values.
        mu : array-like
            Predicted mean.
        sample_weight : array-like
            Weights or exposure to which variance is inversely proportional.
            Should sum up to 1.
        dispersion : float
            The dispersion parameter.
        """
        # sample_weight should sum up to 1.
        assert np.abs(np.sum(sample_weight) - 1) < 1e-5
        W = (link.inverse_derivative(link.link(mu)) ** 2) * get_one_over_variance(
            self, link, mu, link.inverse(mu), dispersion, sample_weight
        )

        return _safe_sandwich_dot(X, W, intercept=fit_intercept)

    def _observed_information(
        self, link, X, y, mu, sample_weight, dispersion, fit_intercept
    ):
        """Compute the observed information matrix.

        Parameters
        ----------
        X : pandas.DataFrame
            The design matrix.
        y : array-like
            Array with outcomes.
        mu : array-like
            Array with predictions.
        sample_weight : array-like
            Array with weights. Should sum up to 1.
        dispersion : float
            The dispersion parameter.
        """
        # sample_weight should sum up to 1.
        assert np.abs(np.sum(sample_weight) - 1) < 1e-5
        linpred = link.link(mu)

        W = (
            -link.inverse_derivative2(linpred) * (y - mu)
            + (link.inverse_derivative(linpred) ** 2)
            * (
                1
                + (y - mu) * self.unit_variance_derivative(mu) / self.unit_variance(mu)
            )
        ) * get_one_over_variance(self, link, mu, linpred, dispersion, sample_weight)

        return _safe_sandwich_dot(np.asanyarray(X), W, intercept=fit_intercept)

    def _score_matrix(self, link, X, y, mu, sample_weight, dispersion, fit_intercept):
        """Compute the score.

        Parameters
        ----------
        X : pandas.DataFrame
            The design matrix.
        y : array-like
            Array with outcomes.
        mu : array-like
            Array with predictions.
        sample_weight: array-like
            Array with sampling weights. Should sum up to 1.
        dispersion : float
            The dispersion parameter.
        """
        # sample_weight should sum up to 1.
        assert np.abs(np.sum(sample_weight) - 1) < 1e-5
        linpred = link.link(mu)

        W = (
            get_one_over_variance(self, link, mu, linpred, dispersion, sample_weight)
            * link.inverse_derivative(linpred)
            * (y - mu)
        ).reshape(-1, 1)

        if fit_intercept:
            return np.hstack((W, np.multiply(X, W)))
        else:
            return np.multiply(X, W)


class TweedieDistribution(ExponentialDispersionModel):
    r"""A class for the Tweedie distribution.

    A Tweedie distribution with mean :math:`\mu = \mathrm{E}(Y)` is uniquely
    defined by its mean-variance relationship
    :math:`\mathrm{var}(Y) \propto \mu^{\mathrm{power}}`.

    Special cases are:

    ====== ================
    Power  Distribution
    ====== ================
    0      Normal
    1      Poisson
    (1, 2) Compound Poisson
    2      Gamma
    3      Inverse Gaussian
    ====== ================

    Parameters
    ----------
    power : float, optional (default=0)
        The variance power of the `unit_variance`
        :math:`v(\mu) = \mu^{\mathrm{power}}`. For
        :math:`0 < \mathrm{power} < 1`, no distribution exists.
    """

    upper_bound = np.Inf
    include_upper_bound = False

    def __init__(self, power=0):
        # validate power and set _upper_bound, _include_upper_bound attrs
        self.power = power

    @property
    def lower_bound(self) -> Union[float, int]:
        """Return the lowest value of ``y`` allowed."""
        if self.power <= 0:
            return -np.Inf
        if self.power >= 1:
            return 0
        raise ValueError

    @property
    def include_lower_bound(self) -> bool:
        """Return whether ``lower_bound`` is allowed as a value of ``y``."""
        if self.power <= 0:
            return False
        if (self.power >= 1) and (self.power < 2):
            return True
        if self.power >= 2:
            return False
        raise ValueError

    @property
    def power(self) -> float:
        """Return the Tweedie power parameter."""
        return self._power

    @power.setter
    def power(self, power):
        if not isinstance(power, (int, float)):
            raise TypeError("power must be an int or float, input was {}".format(power))

        if (power > 0) and (power < 1):
            raise ValueError("For 0<power<1, no distribution exists.")

        # Prevents upcasting when working with 32-bit data
        self._power = power if isinstance(power, int) else np.float32(power)

    def unit_variance(self, mu: np.ndarray) -> np.ndarray:
        """Compute the unit variance of a Tweedie distribution ``v(mu) = mu^power``.

        Parameters
        ----------
        mu : array-like, shape (n_samples,)
            Predicted mean.

        Returns
        -------
        numpy.ndarray, shape (n_samples,)
        """
        p = self.power  # noqa: F841
        return numexpr.evaluate("mu ** p")

    def unit_variance_derivative(self, mu: np.ndarray) -> np.ndarray:
        r"""Compute the derivative of the unit variance of a Tweedie distribution.

        Equation: :math:`v(\mu) = p \times \mu^{(p-1)}`.

        Parameters
        ----------
        mu : array-like, shape (n_samples,)
            Predicted mean.

        Returns
        -------
        numpy.ndarray, shape (n_samples,)
        """
        p = self.power  # noqa: F841
        return numexpr.evaluate("p * mu ** (p - 1)")

    def unit_deviance(self, y, mu):
        """Get the deviance of each observation."""
        p = self.power
        if p == 0:
            # NormalDistribution
            return (y - mu) ** 2
        if p == 1:
            # PoissonDistribution
            # 2 * (y*log(y/mu) - y + mu), with y*log(y/mu)=0 if y=0
            return 2 * (special.xlogy(y, y / mu) - y + mu)
        elif p == 2:
            # GammaDistribution
            return 2 * (np.log(mu / y) + y / mu - 1)
        else:
            # return 2 * (np.maximum(y,0)**(2-p)/((1-p)*(2-p))
            #    - y*mu**(1-p)/(1-p) + mu**(2-p)/(2-p))
            return 2 * (
                np.power(np.maximum(y, 0), 2 - p) / ((1 - p) * (2 - p))
                - y * np.power(mu, 1 - p) / (1 - p)
                + np.power(mu, 2 - p) / (2 - p)
            )

    def _rowwise_gradient_hessian(
        self, link, y, weights, eta, mu, gradient_rows, hessian_rows
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

        if f is not None:
            return f(y, weights, eta, mu, gradient_rows, hessian_rows)

        return super()._rowwise_gradient_hessian(
            link, y, weights, eta, mu, gradient_rows, hessian_rows
        )

    def _eta_mu_deviance(
        self,
        link: Link,
        factor: float,
        cur_eta: np.ndarray,
        X_dot_d: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
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

        if f is not None:
            return f(cur_eta, X_dot_d, y, weights, eta_out, mu_out, factor)

        return super()._eta_mu_deviance(
            link, factor, cur_eta, X_dot_d, y, weights, eta_out, mu_out
        )


class NormalDistribution(TweedieDistribution):
    """Class for the Normal (a.k.a. Gaussian) distribution."""

    def __init__(self):
        super(NormalDistribution, self).__init__(power=0)


class PoissonDistribution(TweedieDistribution):
    """Class for the scaled Poisson distribution."""

    def __init__(self):
        super(PoissonDistribution, self).__init__(power=1)


class GammaDistribution(TweedieDistribution):
    """Class for the Gamma distribution."""

    def __init__(self):
        super(GammaDistribution, self).__init__(power=2)


class InverseGaussianDistribution(TweedieDistribution):
    """Class for the scaled Inverse Gaussian distribution."""

    def __init__(self):
        super(InverseGaussianDistribution, self).__init__(power=3)


class GeneralizedHyperbolicSecant(ExponentialDispersionModel):
    """A class for the Generalized Hyperbolic Secant (GHS) distribution.

    The GHS distribution is for targets ``y`` in ``(-∞, +∞)``.
    """

    lower_bound = -np.Inf
    upper_bound = np.Inf
    include_lower_bound = False
    include_upper_bound = False

    def __init__(self):
        return

    def unit_variance(self, mu: np.ndarray) -> np.ndarray:
        """Get the unit-level expected variance.

        See superclass documentation.

        Parameters
        ----------
        mu : array-like or float

        Returns
        -------
        array-like
        """
        return 1 + mu ** 2

    def unit_variance_derivative(self, mu: np.ndarray) -> np.ndarray:
        """Get the derivative of the unit variance.

        See superclass documentation.

        Parameters
        ----------
        mu : array-like or float

        Returns
        -------
        array-like
        """
        return 2 * mu

    def unit_deviance(self, y: np.ndarray, mu: np.ndarray) -> np.ndarray:
        """Get the unit-level deviance.

        See superclass documentation.

        Parameters
        ----------
        y : array-like
        mu : array-like

        Returns
        -------
        array-like
        """
        return 2 * y * (np.arctan(y) - np.arctan(mu)) + np.log(
            (1 + mu ** 2) / (1 + y ** 2)
        )


class BinomialDistribution(ExponentialDispersionModel):
    """A class for the Binomial distribution.

    The Binomial distribution is for targets ``y`` in ``[0, 1]``.
    """

    lower_bound = 0
    upper_bound = 1
    include_lower_bound = True
    include_upper_bound = True

    def __init__(self):
        return

    def unit_variance(self, mu: np.ndarray) -> np.ndarray:
        """Get the unit-level expected variance.

        See superclass documentation.

        Parameters
        ----------
        mu : array-like

        Returns
        -------
        array-like
        """
        return mu * (1 - mu)

    def unit_variance_derivative(self, mu):
        """Get the derivative of the unit variance.

        See superclass documentation.

        Parameters
        ----------
        mu : array-like or float

        Returns
        -------
        array-like
        """
        return 1 - 2 * mu

    def unit_deviance(self, y: np.ndarray, mu: np.ndarray) -> np.ndarray:
        """Get the unit-level deviance.

        See superclass documentation.

        Parameters
        ----------
        y : array-like
        mu : array-like

        Returns
        -------
        array-like
        """
        return 2 * (
            special.xlogy(y, y)
            - special.xlogy(y, mu)
            + special.xlogy(1 - y, 1 - y)
            - special.xlogy(1 - y, 1 - mu)
        )

    def _rowwise_gradient_hessian(
        self, link, y, weights, eta, mu, gradient_rows, hessian_rows
    ):
        if isinstance(link, LogitLink):
            return binomial_logit_rowwise_gradient_hessian(
                y, weights, eta, mu, gradient_rows, hessian_rows
            )
        return super()._rowwise_gradient_hessian(
            link, y, weights, eta, mu, gradient_rows, hessian_rows
        )

    def _eta_mu_deviance(
        self,
        link: Link,
        factor: float,
        cur_eta: np.ndarray,
        X_dot_d: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        eta_out: np.ndarray,
        mu_out: np.ndarray,
    ):
        if isinstance(link, LogitLink):
            return binomial_logit_eta_mu_deviance(
                cur_eta, X_dot_d, y, weights, eta_out, mu_out, factor
            )
        return super()._eta_mu_deviance(
            link, factor, cur_eta, X_dot_d, y, weights, eta_out, mu_out
        )


def guess_intercept(
    y: np.ndarray,
    weights: np.ndarray,
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
    avg_y = np.average(y, weights=weights)

    if isinstance(link, IdentityLink):
        # This is only correct for normal. For other distributions, answer is unknown,
        # but assume that we want sum(y) = sum(mu)
        if eta is None:
            return avg_y
        avg_eta = eta if np.isscalar(eta) else np.average(eta, weights=weights)
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
        else:
            p = 1  # Like Poisson
        if np.isscalar(mu):
            first = np.log(y.dot(weights) * mu ** (1 - p))
            second = np.log(weights.sum() * mu ** (2 - p))
        else:
            first = np.log((y * mu ** (1 - p)).dot(weights))
            second = np.log((mu ** (2 - p)).dot(weights))
        return first - second
    elif isinstance(link, LogitLink):
        log_odds = np.log(avg_y) - np.log(np.average(1 - y, weights=weights))
        if eta is None:
            return log_odds
        avg_eta = eta if np.isscalar(eta) else np.average(eta, weights=weights)
        return log_odds - avg_eta
    else:
        return link.link(y.dot(weights))


def get_one_over_variance(
    distribution: ExponentialDispersionModel,
    link: Link,
    mu: np.ndarray,
    eta: np.ndarray,
    phi,
    weights: np.ndarray,
):
    """
    Get one over the variance.

    For Tweedie: ``sigma_inv = weights / (mu ** p)`` during optimization,
    because ``phi = 1``.

    For Binomial with Logit link: Simplifies to
    ``variance = phi / ( weights * (exp(eta) + 2 + exp(-eta)))``.
    More numerically accurate.
    """
    if isinstance(distribution, BinomialDistribution) and isinstance(link, LogitLink):
        max_float_for_exp = np.log(np.finfo(eta.dtype).max / 10)
        if np.any(np.abs(eta) > max_float_for_exp):
            eta = np.clip(eta, -max_float_for_exp, max_float_for_exp)  # type: ignore
        return weights * (np.exp(eta) + 2 + np.exp(-eta)) / phi
    return 1.0 / distribution.variance(mu, phi=phi, weights=weights)
