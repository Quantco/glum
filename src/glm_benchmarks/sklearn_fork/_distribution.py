import numbers
from abc import ABCMeta, abstractmethod

import numexpr
import numpy as np
from scipy import special

from ._util import _safe_lin_pred, _safe_sandwich_dot


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
    _score
    _fisher_matrix
    _observed_information
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

    def starting_mu(self, y: np.ndarray, weights=1, ind_weight=0.5) -> np.ndarray:
        """Set starting values for the mean mu.

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
        return np.multiply(ind_weight, y, dtype=expected_dtype) + np.multiply(
            1 - ind_weight, np.average(y, weights=weights), dtype=expected_dtype
        )

    def _mu_deviance_derivative(self, coef, X, y, weights, link):
        """Compute mu and the derivative of the deviance w.r.t coef."""
        lin_pred = _safe_lin_pred(X, coef)
        mu = link.inverse(lin_pred)
        d1 = link.inverse_derivative(lin_pred)
        temp = d1 * self.deviance_derivative(y, mu, weights)
        if coef.size == X.shape[1] + 1:
            devp = np.concatenate(([temp.sum()], temp @ X))
        else:
            devp = temp @ X  # sampe as X.T @ temp
        return mu, devp

    def _score(self, coef, phi, X, y, weights, link):
        r"""Compute the score function.

        The score function is the derivative of the
        log-likelihood w.r.t. `coef` (:math:`w`).
        It is given by

        .. math:

            \mathbf{score}(\boldsymbol{w})
            = \frac{\partial loglike}{\partial\boldsymbol{w}}
            = \mathbf{X}^T \mathbf{D}
            \boldsymbol{\Sigma}^-1 (\mathbf{y} - \boldsymbol{\mu})\,,

        with :math:`\mathbf{D}=\mathrm{diag}(h'(\eta_1),\ldots)` and
        :math:`\boldsymbol{\Sigma}=\mathrm{diag}(\mathbf{V}[y_1],\ldots)`.
        Note: The derivative of the deviance w.r.t. coef equals -2 * score.
        """
        lin_pred = _safe_lin_pred(X, coef)
        mu = link.inverse(lin_pred)
        sigma_inv = 1 / self.variance(mu, phi=phi, weights=weights)
        d = link.inverse_derivative(lin_pred)
        temp = sigma_inv * d * (y - mu)
        if coef.size == X.shape[1] + 1:
            score = np.concatenate(([temp.sum()], temp @ X))
        else:
            score = temp @ X  # sampe as X.T @ temp
        return score

    def _fisher_matrix(self, coef, phi, X, weights, link):
        r"""Compute the Fisher information matrix.

        The Fisher information matrix, also known as expected information
        matrix is given by

        .. math:

            \mathbf{F}(\boldsymbol{w}) =
            \mathrm{E}\left[-\frac{\partial\mathbf{score}}{\partial
            \boldsymbol{w}} \right]
            = \mathrm{E}\left[
            -\frac{\partial^2 loglike}{\partial\boldsymbol{w}
            \partial\boldsymbol{w}^T}\right]
            = \mathbf{X}^T W \mathbf{X} \,,

        with :math:`\mathbf{W} = \mathbf{D}^2 \boldsymbol{\Sigma}^{-1}`,
        see func:`_score`.
        """
        lin_pred = _safe_lin_pred(X, coef)
        mu = link.inverse(lin_pred)
        sigma_inv = 1 / self.variance(mu, phi=phi, weights=weights)
        d = link.inverse_derivative(lin_pred)
        d2_sigma_inv = sigma_inv * d * d
        intercept = coef.size == X.shape[1] + 1
        fisher_matrix = _safe_sandwich_dot(X, d2_sigma_inv, intercept=intercept)
        return fisher_matrix

    def _observed_information(self, coef, phi, X, y, weights, link):
        r"""Compute the observed information matrix.

        The observed information matrix, also known as the negative of
        the Hessian matrix of the log-likelihood, is given by

        .. math:

            \mathbf{H}(\boldsymbol{w}) =
            -\frac{\partial^2 loglike}{\partial\boldsymbol{w}
            \partial\boldsymbol{w}^T}
            = \mathbf{X}^T \left[
            - \mathbf{D}' \mathbf{R}
            + \mathbf{D}^2 \mathbf{V} \mathbf{R}
            + \mathbf{D}^2
            \right] \boldsymbol{\Sigma}^{-1} \mathbf{X} \,,

        with :math:`\mathbf{R} = \mathrm{diag}(y_i - \mu_i)`,
        :math:`\mathbf{V} = \mathrm{diag}\left(\frac{v'(\mu_i)}{
        v(\mu_i)}
        \right)`,
        see :func:`score_` function and :func:`_fisher_matrix`.
        """
        lin_pred = _safe_lin_pred(X, coef)
        mu = link.inverse(lin_pred)
        sigma_inv = 1 / self.variance(mu, phi=phi, weights=weights)
        dp = link.inverse_derivative2(lin_pred)
        d2 = link.inverse_derivative(lin_pred) ** 2
        v = self.unit_variance_derivative(mu) / self.unit_variance(mu)
        r = y - mu
        temp = sigma_inv * (-dp * r + d2 * v * r + d2)
        intercept = coef.size == X.shape[1] + 1
        observed_information = _safe_sandwich_dot(X, temp, intercept=intercept)
        return observed_information

    def _eta_mu_score_fisher(
        self, coef, phi, X, y, weights, link, diag_fisher=False, eta=None, mu=None
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
            eta = _safe_lin_pred(X, coef)
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

        d2_sigma_inv = d1 * d1_sigma_inv
        if diag_fisher:
            fisher_matrix = d2_sigma_inv
        else:
            fisher_matrix = _safe_sandwich_dot(X, d2_sigma_inv, intercept=intercept)
        return eta, mu, score, fisher_matrix


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
