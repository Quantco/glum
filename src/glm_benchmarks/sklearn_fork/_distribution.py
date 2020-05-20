import numbers
from abc import ABCMeta, abstractmethod
from typing import Tuple, Union

import numexpr
import numpy as np
from scipy import special

from ._link import IdentityLink, Link, LogitLink, LogLink
from ._util import _safe_lin_pred


class ExponentialDispersionModel(metaclass=ABCMeta):
    r"""Base class for reproductive Exponential Dispersion Models (EDM).

    The pdf of :math:`Y\sim \mathrm{EDM}(\mu, \phi)` is given by

    .. math:: p(y| \theta, \phi) = c(y, \phi)
        \exp\left(\frac{\theta y-A(\theta)}{\phi}\right)
        = \tilde{c}(y, \phi)
            \exp\left(-\frac{d(y, \mu)}{2\phi}\right)

    with mean :math:`\mathrm{E}[Y] = A'(\theta) = \mu`,
    variance :math:`\mathrm{Var}[Y] = \phi \cdot v(\mu)`,
    unit variance :math:`v(\mu)` and
    unit deviance :math:`d(y,\mu)`.

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
    _eta_mu_score_fisher

    References
    ----------

    https://en.wikipedia.org/wiki/Exponential_dispersion_model.
    """

    @property
    def lower_bound(self):
        """Get the lower bound of values for Y~EDM."""
        return self._lower_bound

    @property
    def upper_bound(self):
        """Get the upper bound of values for Y~EDM."""
        return self._upper_bound

    @property
    def include_lower_bound(self):
        """Get True if lower bound for y is included: y >= lower_bound."""
        return self._include_lower_bound

    @property
    def include_upper_bound(self):
        """Get True if upper bound for y is included: y <= upper_bound."""
        return self._include_upper_bound

    def in_y_range(self, x):
        """Returns ``True`` if x is in the valid range of Y~EDM.

        Parameters
        ----------
        x : array, shape (n_samples,)
            Target values.
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

        The unit variance :math:`v(\mu)` determines the variance as
        a function of the mean :math:`\mu` by
        :math:`\mathrm{Var}[Y_i] = \phi/s_i*v(\mu_i)`.
        It can also be derived from the unit deviance :math:`d(y,\mu)` as

        .. math:: v(\mu) = \frac{2}{\frac{\partial^2 d(y,\mu)}{
            \partial\mu^2}}\big|_{y=\mu}

        See also :func:`variance`.

        Parameters
        ----------
        mu : array, shape (n_samples,)
            Predicted mean.
        """
        pass

    @abstractmethod
    def unit_variance_derivative(self, mu):
        r"""Compute the derivative of the unit variance w.r.t. mu.

        Return :math:`v'(\mu)`.

        Parameters
        ----------
        mu : array, shape (n_samples,)
            Target values.
        """
        pass

    def variance(self, mu: np.ndarray, phi=1, weights=1) -> np.ndarray:
        r"""Compute the variance function.

        The variance of :math:`Y_i \sim \mathrm{EDM}(\mu_i,\phi/s_i)` is
        :math:`\mathrm{Var}[Y_i]=\phi/s_i*v(\mu_i)`,
        with unit variance :math:`v(\mu)` and weights :math:`s_i`.

        Parameters
        ----------
        mu : array, shape (n_samples,)
            Predicted mean.

        phi : float (default=1)
            Dispersion parameter.

        weights : array, shape (n_samples,) (default=1)
            Weights or exposure to which variance is inverse proportional.
        """
        return phi / weights * self.unit_variance(mu)

    def variance_derivative(self, mu, phi=1, weights=1):
        r"""Compute the derivative of the variance w.r.t. mu.

        Returns
        :math:`\frac{\partial}{\partial\mu}\mathrm{Var}[Y_i]
        =phi/s_i*v'(\mu_i)`, with unit variance :math:`v(\mu)`
        and weights :math:`s_i`.

        Parameters
        ----------
        mu : array, shape (n_samples,)
            Predicted mean.

        phi : float (default=1)
            Dispersion parameter.

        weights : array, shape (n_samples,) (default=1)
            Weights or exposure to which variance is inverse proportional.
        """
        return phi / weights * self.unit_variance_derivative(mu)

    @abstractmethod
    def unit_deviance(self, y, mu):
        r"""Compute the unit deviance.

        The unit_deviance :math:`d(y,\mu)` can be defined by the
        log-likelihood as
        :math:`d(y,\mu) = -2\phi\cdot
        \left(loglike(y,\mu,\phi) - loglike(y,y,\phi)\right).`

        Parameters
        ----------
        y : array, shape (n_samples,)
            Target values.

        mu : array, shape (n_samples,)
            Predicted mean.
        """
        pass

    def unit_deviance_derivative(self, y, mu):
        r"""Compute the derivative of the unit deviance w.r.t. mu.

        The derivative of the unit deviance is given by
        :math:`\frac{\partial}{\partial\mu}d(y,\mu) = -2\frac{y-\mu}{v(\mu)}`
        with unit variance :math:`v(\mu)`.

        Parameters
        ----------
        y : array, shape (n_samples,)
            Target values.

        mu : array, shape (n_samples,)
            Predicted mean.
        """
        return -2 * (y - mu) / self.unit_variance(mu)

    def deviance(self, y, mu, weights=1):
        r"""Compute the deviance.

        The deviance is a weighted sum of the per sample unit deviances,
        :math:`D = \sum_i s_i \cdot d(y_i, \mu_i)`
        with weights :math:`s_i` and unit deviance :math:`d(y,\mu)`.
        In terms of the log-likelihood it is :math:`D = -2\phi\cdot
        \left(loglike(y,\mu,\frac{phi}{s})
        - loglike(y,y,\frac{phi}{s})\right)`.

        Parameters
        ----------
        y : array, shape (n_samples,)
            Target values.

        mu : array, shape (n_samples,)
            Predicted mean.

        weights : array, shape (n_samples,) (default=1)
            Weights or exposure to which variance is inverse proportional.
        """
        return np.sum(weights * self.unit_deviance(y, mu))

    def deviance_derivative(self, y, mu, weights=1):
        """Compute the derivative of the deviance w.r.t. mu.

        It gives :math:`\\frac{\\partial}{\\partial\\mu} D(y, \\mu; weights)`.

        Parameters
        ----------
        y : array, shape (n_samples,)
            Target values.

        mu : array, shape (n_samples,)
            Predicted mean.

        weights : array, shape (n_samples,) (default=1)
            Weights or exposure to which variance is inverse proportional.
        """
        return weights * self.unit_deviance_derivative(y, mu)

    def starting_mu(
        self,
        y: np.ndarray,
        weights=1,
        ind_weight=0.5,
        offset: np.ndarray = None,
        link: Link = None,
    ) -> np.ndarray:
        """Set starting values for the mean mu by interpolating between a best-guess
        case of where mu may end up, fitting y perfectly, and a worst-case guess,
        where we can only fit an intercept.

        In the worst case, we have worst_case_mu = link.inverse(eta + intercept),
        where intercept is set so that
        sum(y) = sum(link.inverse(eta + intercept))

        When eta = 0, this simplifies to intercept = link.link(avg(y)), so we can
        simply write the worse-case mu as avg(y).

        These may be good starting points for the (unpenalized) IRLS solver.


        Parameters
        ----------
        y : array, shape (n_samples,)
            Target values.

        weights : array, shape (n_samples,) (default=1)
            Weights or exposure to which variance is inverse proportional.

        ind_weight : float (default=0.5)
            Must be between 0 and 1. Specifies how much weight is given to the
            individual observations instead of the mean of y.
        """

        expected_dtype = np.float64 if y.dtype.itemsize == 8 else np.float32

        # Be careful: combining a 32-bit int and 32-bit float gives 64-bit answers
        def _interpolate(vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
            return np.multiply(ind_weight, vec1, dtype=expected_dtype) + np.multiply(
                1 - ind_weight, vec2, dtype=expected_dtype
            )

        if offset is None:
            # If our predictors are bad, we can at least fit the mean
            worst_case_pred = np.average(y, weights=weights)
        else:
            assert link is not None
            # worst case: guess offset plus an intercept
            best_intercept = guess_intercept(y, weights, link, offset)
            worst_case_pred = link.inverse(offset + best_intercept)

        return _interpolate(y, worst_case_pred)

    def _mu_deviance_derivative(
        self,
        coef: np.ndarray,
        X,
        y: np.ndarray,
        weights: np.ndarray,
        link: Link,
        offset: np.ndarray = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute mu and the derivative of the deviance w.r.t coef."""
        lin_pred = _safe_lin_pred(X, coef, offset)
        mu = link.inverse(lin_pred)
        d1 = link.inverse_derivative(lin_pred)
        temp = d1 * self.deviance_derivative(y, mu, weights)
        if coef.size == X.shape[1] + 1:
            devp = np.concatenate(([temp.sum()], temp @ X))
        else:
            devp = temp @ X  # same as X.T @ temp
        return mu, devp

    def _eta_mu_score_fisher(
        self,
        coef: np.ndarray,
        phi,
        X,
        y: np.ndarray,
        weights: np.ndarray,
        link: Link,
        diag_fisher: bool = False,
        eta: np.ndarray = None,
        mu: np.ndarray = None,
        offset: np.ndarray = None,
    ):
        """Compute linear predictor, mean, score function and fisher matrix.

        It calculates the linear predictor, the mean, score function
        (derivative of log-likelihood) and Fisher information matrix
        all in one go as function of `coef` (:math:`w`) and the data.

        Parameters
        ----------
        diag_fisher : boolean, optional (default=False)
            If ``True``, returns only an array d such that
            fisher = X.T @ np.diag(d) @ X.

        Returns
        -------
        (eta, mu, score, fisher) : tuple with 4 elements
            The 4 elements are:

            * eta: ndarray, shape (X.shape[0],)
            * mu: ndarray, shape (X.shape[0],)
            * score: ndarray, shape (X.shape[0],)
            * fisher:

                * If diag_fisher is ``False``, the full fisher matrix,
                  an array of shape (X.shape[1], X.shape[1])
                * If diag_fisher is ``True`, an array of shape (X.shape[0])
        """
        intercept = coef.size == X.shape[1] + 1
        # eta = linear predictor
        if eta is None:
            eta = _safe_lin_pred(X, coef, offset)
        if mu is None:
            mu = link.inverse(eta)

        # FOR TWEEDIE: sigma_inv = weights / (mu ** p) during optimization bc phi = 1
        sigma_inv = 1.0 / self.variance(mu, phi=phi, weights=weights)
        d1 = link.inverse_derivative(eta)  # = h'(eta)
        # Alternatively:
        # h'(eta) = h'(g(mu)) = 1/g'(mu), note that h is inverse of g
        # d1 = 1./link.derivative(mu)
        d1_sigma_inv = d1 * sigma_inv
        temp = d1_sigma_inv * (y - mu)
        if intercept:
            score = np.concatenate(([temp.sum()], temp @ X))
        else:
            score = temp @ X

        fisher_W = d1 * d1_sigma_inv
        # To form the fisher matrix:
        # fisher_matrix = _safe_sandwich_dot(X, fisher_W, intercept=intercept)
        return eta, mu, score, fisher_W


class TweedieDistribution(ExponentialDispersionModel):
    r"""A class for the Tweedie distribution.

    A Tweedie distribution with mean :math:`\mu=\mathrm{E}[Y]` is uniquely
    defined by it's mean-variance relationship
    :math:`\mathrm{Var}[Y] \propto \mu^power`.

    Special cases are:

    ===== ================
    Power Distribution
    ===== ================
    0     Normal
    1     Poisson
    (1,2) Compound Poisson
    2     Gamma
    3     Inverse Gaussian

    Parameters
    ----------
    power : float (default=0)
            The variance power of the `unit_variance`
            :math:`v(\mu) = \mu^{power}`.
            For ``0<power<1``, no distribution exists.
    """

    def __init__(self, power=0):
        # validate power and set _upper_bound, _include_upper_bound attrs
        self.power = power

    @property
    def power(self):
        return self._power

    @power.setter
    def power(self, power):
        if not isinstance(power, numbers.Real):
            raise TypeError("power must be a real number, input was {}".format(power))

        self._upper_bound = np.Inf
        self._include_upper_bound = False
        if power < 0:
            # Extreme Stable
            self._lower_bound = -np.Inf
            self._include_lower_bound = False
        elif power == 0:
            # NormalDistribution
            self._lower_bound = -np.Inf
            self._include_lower_bound = False
        elif (power > 0) and (power < 1):
            raise ValueError("For 0<power<1, no distribution exists.")
        elif power == 1:
            # PoissonDistribution
            self._lower_bound = 0
            self._include_lower_bound = True
        elif (power > 1) and (power < 2):
            # Compound Poisson
            self._lower_bound = 0
            self._include_lower_bound = True
        elif power == 2:
            # GammaDistribution
            self._lower_bound = 0
            self._include_lower_bound = False
        elif (power > 2) and (power < 3):
            # Positive Stable
            self._lower_bound = 0
            self._include_lower_bound = False
        elif power == 3:
            # InverseGaussianDistribution
            self._lower_bound = 0
            self._include_lower_bound = False
        elif power > 3:
            # Positive Stable
            self._lower_bound = 0
            self._include_lower_bound = False
        else:  # pragma: no cover
            # this branch should be unreachable.
            raise ValueError

        # Prevents upcasting when working with 32-bit data
        self._power = power if isinstance(power, int) else np.float32(power)

    def unit_variance(self, mu: np.ndarray) -> np.ndarray:
        """Compute the unit variance of a Tweedie distribution v(mu)=mu**power.

        Parameters
        ----------
        mu : array, shape (n_samples,)
            Predicted mean.
        """
        p = self.power  # noqa: F841
        return numexpr.evaluate("mu ** p")

    def unit_variance_derivative(self, mu: np.ndarray) -> np.ndarray:
        """Compute the derivative of the unit variance of a Tweedie
        distribution v(mu)=power*mu**(power-1).

        Parameters
        ----------
        mu : array, shape (n_samples,)
            Predicted mean.
        """
        p = self.power  # noqa: F841
        return numexpr.evaluate("p * mu ** (p - 1)")

    def unit_deviance(self, y, mu):
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


class NormalDistribution(TweedieDistribution):
    """Class for the Normal (aka Gaussian) distribution"""

    def __init__(self):
        super(NormalDistribution, self).__init__(power=0)


class PoissonDistribution(TweedieDistribution):
    """Class for the scaled Poisson distribution"""

    def __init__(self):
        super(PoissonDistribution, self).__init__(power=1)


class GammaDistribution(TweedieDistribution):
    """Class for the Gamma distribution"""

    def __init__(self):
        super(GammaDistribution, self).__init__(power=2)


class InverseGaussianDistribution(TweedieDistribution):
    """Class for the scaled InverseGaussianDistribution distribution"""

    def __init__(self):
        super(InverseGaussianDistribution, self).__init__(power=3)


class GeneralizedHyperbolicSecant(ExponentialDispersionModel):
    """A class for the Generalized Hyperbolic Secant (GHS) distribution.

    The GHS distribution is for targets y in (-inf, inf).
    """

    def __init__(self):
        self._lower_bound = -np.Inf
        self._upper_bound = np.Inf
        self._include_lower_bound = False
        self._include_upper_bound = False

    def unit_variance(self, mu):
        return 1 + mu ** 2

    def unit_variance_derivative(self, mu):
        return 2 * mu

    def unit_deviance(self, y, mu):
        return 2 * y * (np.arctan(y) - np.arctan(mu)) + np.log(
            (1 + mu ** 2) / (1 + y ** 2)
        )


class BinomialDistribution(ExponentialDispersionModel):
    """A class for the Binomial distribution.

    The Binomial distribution is for targets y in [0, 1].
    """

    def __init__(self):
        self._lower_bound = 0
        self._upper_bound = 1
        self._include_lower_bound = True
        self._include_upper_bound = True

    def unit_variance(self, mu):
        return mu * (1 - mu)

    def unit_variance_derivative(self, mu):
        return 1 - 2 * mu

    def unit_deviance(self, y, mu):
        return 2 * (special.xlogy(y, y / mu) + special.xlogy(1 - y, (1 - y) / (1 - mu)))


def guess_intercept(
    y: np.ndarray,
    weights: np.ndarray,
    link: Link,
    distribution: ExponentialDispersionModel,
    eta: Union[np.ndarray, float] = None,
):
    """
    Say we want to find the scalar "b" that minimizes LL(eta + b), with eta fixed.

    An exact solution exists for Tweedie distributions with a log link, and for the
    normal distribution with identity link.

    An exact solution also exists for the case of logit with no offset.

    If the distribution and corresponding link are something else, we use the Tweedie
    or normal solution, depending on the link function.
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
        raise NotImplementedError
