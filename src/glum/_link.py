import warnings
from abc import ABCMeta, abstractmethod
from typing import Callable

import numpy as np
from scipy import special


class Link(metaclass=ABCMeta):
    """Abstract base class for link functions."""

    @abstractmethod
    def link(self, mu):
        """Compute the link function.

        The link function ``g`` links the mean, ``mu ≡ E(Y)``, to the linear
        predictor, ``X * w``, so that ``g(mu)`` is equal to the linear predictor.

        Parameters
        ----------
        mu : array-like, shape (n_samples,)
            Usually the (predicted) mean.
        """
        pass

    @abstractmethod
    def derivative(self, mu):
        """Compute the derivative of the link function.

        Parameters
        ----------
        mu : array-like, shape (n_samples,)
            Usually the (predicted) mean.
        """
        pass

    @abstractmethod
    def inverse(self, lin_pred):
        """Compute the inverse link function.

        The inverse link function ``h`` gives the inverse relationship between
        the linear predictor, ``X * w``, and the mean, ``mu ≡ E(Y)``, so that
        ``h(X * w) = mu``.

        Parameters
        ----------
        lin_pred : array-like, shape (n_samples,)
            Usually the (fitted) linear predictor.
        """
        pass

    @abstractmethod
    def inverse_derivative(self, lin_pred):
        """Compute the derivative of the inverse link function.

        Parameters
        ----------
        lin_pred : array-like, shape (n_samples,)
            Usually the (fitted) linear predictor.
        """
        pass

    @abstractmethod
    def inverse_derivative2(self, lin_pred):
        """Compute second derivative of the inverse link function.

        Parameters
        ----------
        lin_pred : array-like, shape (n_samples,)
            Usually the (fitted) linear predictor.
        """
        pass

    def to_tweedie(self, safe=True):
        """Return the Tweedie representation of a link function if it exists."""
        if hasattr(self, "__tweedie_repr__"):
            return self.__tweedie_repr__()
        if safe:
            raise ValueError("This link function has no Tweedie representation.")
        return None


def catch_p(fun) -> Callable:
    """Ensure that linear predictors are compatible with the Tweedie power parameter."""

    def _to_return(*args, **kwargs):
        with np.errstate(invalid="raise"):
            try:
                result = fun(*args, **kwargs)
            except FloatingPointError as e:
                raise ValueError(
                    "Your linear predictors are not supported for power "
                    f"{args[0].power}. For negative linear predictors, consider using "
                    "a log link instead."
                ) from e
        return result

    return _to_return


class TweedieLink(Link):
    """The Tweedie link function ``x^(1-p)`` if ``p≠1`` and ``log(x)`` if ``p=1``.

    See the documentation of the superclass, :class:`~glum.Link`, for details.
    """

    def __init__(self, power):
        self.power = power

    def __eq__(self, other):  # noqa D
        return isinstance(other, self.__class__) and (self.power == other.power)

    def __tweedie__repr__(self):  # noqa D
        return self.__class__(self.power)

    @property
    def power(self):  # noqa D
        return self._power

    @power.setter
    def power(self, power):
        if not isinstance(power, (int, float, np.number)):
            raise TypeError(f"The power parameter must be numeric; got {power}.")
        if (power > 0) and (power < 1):
            raise ValueError("For `0<p<1`, no distribution exists.")

        # Prevents upcasting when working with 32-bit data
        self._power = power if isinstance(power, int) else np.float32(power)

    def link(self, mu):  # noqa D
        if self.power == 0:
            return mu
        if self.power == 1:
            return np.log(mu)
        return mu ** (1 - self.power)

    def derivative(self, mu):  # noqa D
        if self.power == 0:
            return 1.0 if np.isscalar(mu) else np.ones_like(mu)
        if self.power == 1:
            return 1 / mu
        return (1 - self.power) * mu ** (-self.power)

    @catch_p
    def inverse(self, lin_pred):  # noqa D
        if self.power == 0:
            return lin_pred
        if self.power == 1:
            return np.exp(lin_pred)
        return lin_pred ** (1 / (1 - self.power))

    @catch_p
    def inverse_derivative(self, lin_pred):  # noqa D
        if self.power == 0:
            return 1.0 if np.isscalar(lin_pred) else np.ones_like(lin_pred)
        if self.power == 1:
            return np.exp(lin_pred)

        return (1 / (1 - self.power)) * lin_pred ** (self.power / (1 - self.power))

    @catch_p
    def inverse_derivative2(self, lin_pred):  # noqa D
        if self.power == 0:
            return 0.0 if np.isscalar(lin_pred) else np.zeros_like(lin_pred)
        if self.power == 1:
            return np.exp(lin_pred)

        result = lin_pred ** ((2 * self.power - 1) / (1 - self.power))
        result *= self.power / (1 - self.power) ** 2

        return result


class IdentityLink(Link):
    """The identity link function."""

    def __eq__(self, other):  # noqa D
        return isinstance(other, self.__class__)

    def __tweedie_repr__(self):  # noqa D
        return TweedieLink(0)

    def link(self, mu):  # noqa D
        return mu

    def derivative(self, mu):  # noqa D
        return 1 if np.isscalar(mu) else np.ones_like(mu)

    def inverse(self, lin_pred):  # noqa D
        return lin_pred

    def inverse_derivative(self, lin_pred):  # noqa D
        return 1.0 if np.isscalar(lin_pred) else np.ones_like(lin_pred)

    def inverse_derivative2(self, lin_pred):  # noqa D
        return 0.0 if np.isscalar(lin_pred) else np.zeros_like(lin_pred)


class LogLink(Link):
    """The log link function ``log(x)``."""

    def __eq__(self, other):  # noqa D
        return isinstance(other, self.__class__)

    def __tweedie_repr__(self):  # noqa D
        return TweedieLink(1)

    def link(self, mu):  # noqa D
        return np.log(mu)

    def derivative(self, mu):  # noqa D
        return 1 / mu

    def inverse(self, lin_pred):  # noqa D
        return np.exp(lin_pred)

    def inverse_derivative(self, lin_pred):  # noqa D
        return np.exp(lin_pred)

    def inverse_derivative2(self, lin_pred):  # noqa D
        return np.exp(lin_pred)


class LogitLink(Link):
    """The logit link function ``logit(x)``."""

    def __eq__(self, other):  # noqa D
        return isinstance(other, self.__class__)

    def link(self, mu):  # noqa D
        return special.logit(mu)

    def derivative(self, mu):  # noqa D
        return 1.0 / (mu * (1 - mu))

    def inverse(self, lin_pred):  # noqa D
        inv_logit = special.expit(lin_pred)
        eps50 = 50 * np.finfo(inv_logit.dtype).eps

        if np.any(inv_logit > 1 - eps50) or np.any(inv_logit < eps50):
            warnings.warn("Sigmoid function too close to 0 or 1. Clipping.")
            return np.clip(inv_logit, eps50, 1 - eps50)

        return inv_logit

    def inverse_derivative(self, lin_pred):  # noqa D
        ep = special.expit(lin_pred)
        return ep * (1.0 - ep)

    def inverse_derivative2(self, lin_pred):  # noqa D
        ep = special.expit(lin_pred)
        return ep * (1.0 - ep) * (1.0 - 2 * ep)


class CloglogLink(Link):
    """The complementary log-log link function ``log(-log(-p))``."""

    def __eq__(self, other):  # noqa D
        return isinstance(other, self.__class__)

    def link(self, mu):  # noqa D
        return np.log(-np.log1p(-mu))

    def derivative(self, mu):  # noqa D
        return 1.0 / ((mu - 1) * (np.log1p(-mu)))

    def inverse(self, lin_pred):  # noqa D
        lin_pred = lin_pred
        inv_cloglog = -np.expm1(-np.exp(lin_pred))
        eps50 = 50 * np.finfo(inv_cloglog.dtype).eps

        if np.any(inv_cloglog > 1 - eps50) or np.any(inv_cloglog < eps50):
            warnings.warn("Sigmoid function too close to 0 or 1. Clipping.")
            return np.clip(inv_cloglog, eps50, 1 - eps50)

        return inv_cloglog

    def inverse_derivative(self, lin_pred):  # noqa D
        return np.exp(lin_pred - np.exp(lin_pred))

    def inverse_derivative2(self, lin_pred):  # noqa D
        # TODO: check if numerical stability can be improved
        return np.exp(np.exp(lin_pred) - lin_pred) * np.expm1(lin_pred)
