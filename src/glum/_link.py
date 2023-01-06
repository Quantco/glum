import warnings
from abc import ABCMeta, abstractmethod

import numpy as np
from scipy import special

from ._util import _asanyarray


class Link(metaclass=ABCMeta):
    """Abstract base class for Link functions."""

    @abstractmethod
    def link(self, mu):
        """Compute the link function ``g(mu)``.

        The link function links the mean, ``mu ≡ E(Y)``, to the linear predictor
        ``X * w``, i.e. ``g(mu)`` is equal to the linear predictor.

        Parameters
        ----------
        mu : array-like, shape (n_samples,)
            Usually the (predicted) mean.
        """
        pass

    @abstractmethod
    def derivative(self, mu):
        """Compute the derivative of the link ``g'(mu)``.

        Parameters
        ----------
        mu : array-like, shape (n_samples,)
            Usually the (predicted) mean.
        """
        pass

    @abstractmethod
    def inverse(self, lin_pred):
        """Compute the inverse link function ``h(lin_pred)``.

        Gives the inverse relationship between linear predictor,
        ``lin_pred ≡ X * w``, and the mean, ``mu ≡ E(Y)``, i.e.
        ``h(lin_pred) = mu``.

        Parameters
        ----------
        lin_pred : array-like, shape (n_samples,)
            Usually the (fitted) linear predictor.
        """
        pass

    @abstractmethod
    def inverse_derivative(self, lin_pred):
        """Compute the derivative of the inverse link function ``h'(lin_pred)``.

        Parameters
        ----------
        lin_pred : array-like, shape (n_samples,)
            Usually the (fitted) linear predictor.
        """
        pass

    @abstractmethod
    def inverse_derivative2(self, lin_pred):
        """Compute second derivative of the inverse link function ``h''(lin_pred)``.

        Parameters
        ----------
        lin_pred : array-like, shape (n_samples,)
            Usually the (fitted) linear predictor.
        """
        pass


class IdentityLink(Link):
    """The identity link function ``g(x) = x``."""

    def __eq__(self, other):  # noqa D
        return isinstance(other, self.__class__)

    # unnecessary type hint for consistency with other methods
    def link(self, mu):
        """Return mu (identity link).

        See superclass documentation.

        Parameters
        ----------
        mu: array-like
        """
        return _asanyarray(mu)

    def derivative(self, mu):
        """Get the derivative of the identity link, a vector of ones.

        See superclass documentation.

        Parameters
        ----------
        mu: array-like
        """
        return 1.0 if np.isscalar(mu) else np.ones_like(mu)

    # unnecessary type hint for consistency with other methods
    def inverse(self, lin_pred):
        """Compute the inverse link function ``h(lin_pred)``.

        Gives the inverse relationship between linear predictor and the mean
        ``mu ≡ E(Y)``, i.e. ``h(linear predictor) = mu``.

        Parameters
        ----------
        lin_pred : array-like, shape (n_samples,)
            Usually the (fitted) linear predictor.
        """
        return _asanyarray(lin_pred)

    def inverse_derivative(self, lin_pred):
        """Compute the derivative of the inverse link function ``h'(lin_pred)``.

        Parameters
        ----------
        lin_pred : array-like, shape (n_samples,)
            Usually the (fitted) linear predictor.
        """
        return 1.0 if np.isscalar(lin_pred) else np.ones_like(lin_pred)

    def inverse_derivative2(self, lin_pred):
        """Compute second derivative of the inverse link function ``h''(lin_pred)``.

        Parameters
        ----------
        lin_pred : array-like, shape (n_samples,)
            Usually the (fitted) linear predictor.
        """
        return 0.0 if np.isscalar(lin_pred) else np.zeros_like(lin_pred)


class LogLink(Link):
    """The log link function ``log(x)``."""

    def __eq__(self, other):  # noqa D
        return isinstance(other, self.__class__)

    def link(self, mu):
        """Get the log of ``mu``.

        See superclass documentation.

        Parameters
        ----------
        mu: array-like

        Returns
        -------
        numpy.ndarray
        """
        return np.log(mu)

    def derivative(self, mu):
        """Get the derivative of the log link, one over ``mu``.

        Parameters
        ----------
        mu: array-like

        Returns
        -------
        numpy.ndarray
        """
        return 1 / mu

    def inverse(self, lin_pred):
        """Get the inverse of the log link, the exponential function.

        See superclass documentation.

        Parameters
        ----------
        lin_pred: array-like

        Returns
        -------
        numpy.ndarray
        """
        return np.exp(lin_pred)

    def inverse_derivative(self, lin_pred):
        """Compute the derivative of the inverse link function ``h'(lin_pred)``.

        Parameters
        ----------
        lin_pred : array-like, shape (n_samples,)
            Usually the (fitted) linear predictor.
        """
        return np.exp(lin_pred)

    def inverse_derivative2(self, lin_pred):
        """Compute 2nd derivative of the inverse link function ``h''(lin_pred)``.

        Parameters
        ----------
        lin_pred : array-like, shape (n_samples,)
            Usually the (fitted) linear predictor.
        """
        return np.exp(lin_pred)


class LogitLink(Link):
    """The logit link function ``logit(x)``."""

    def __eq__(self, other):  # noqa D
        return isinstance(other, self.__class__)

    def link(self, mu):
        """Get the logit function of ``mu``.

        See superclass documentation.

        Parameters
        ----------
        mu: array-like

        Returns
        -------
        numpy.ndarray
        """
        return special.logit(_asanyarray(mu))

    def derivative(self, mu):
        """Get the derivative of the logit link.

        See superclass documentation.

        Parameters
        ----------
        mu: array-like

        Returns
        -------
        array-like
        """
        mu = _asanyarray(mu)
        return 1.0 / (mu * (1 - mu))

    def inverse(self, lin_pred):
        """Get the inverse of the logit link.

        See superclass documentation.

        Note: since passing a very large value might result in an output of one,
        this function bounds the output to be between ``[50*eps, 1 - 50*eps]``,
        where ``eps`` is floating point epsilon.

        Parameters
        ----------
        lin_pred: array-like

        Returns
        -------
        array-like
        """
        inv_logit = special.expit(_asanyarray(lin_pred))
        eps50 = 50 * np.finfo(inv_logit.dtype).eps
        if np.any(inv_logit > 1 - eps50) or np.any(inv_logit < eps50):
            warnings.warn(
                "Computing sigmoid function gave results too close to 0 or 1. Clipping."
            )
            return np.clip(inv_logit, eps50, 1 - eps50)
        return inv_logit

    def inverse_derivative(self, lin_pred):
        """Compute the derivative of the inverse link function ``h'(lin_pred)``.

        Parameters
        ----------
        lin_pred : array, shape (n_samples,)
            Usually the (fitted) linear predictor.
        """
        ep = special.expit(_asanyarray(lin_pred))
        return ep * (1.0 - ep)

    def inverse_derivative2(self, lin_pred):
        """Compute 2nd derivative of the inverse link function ``h''(lin_pred)``.

        Parameters
        ----------
        lin_pred : array, shape (n_samples,)
            Usually the (fitted) linear predictor.
        """
        ep = special.expit(_asanyarray(lin_pred))
        return ep * (1.0 - ep) * (1.0 - 2 * ep)


def catch_p(fun):
    """
    Decorate ``fun``, ensuring that the given linear predictor is compatible with the \
    relevant Tweedie power parameter.

    Parameters
    ----------
    fun: TweedieLink method

    Returns
    -------
    Callable
    """

    def _to_return(*args, **kwargs):
        with np.errstate(invalid="raise"):
            try:
                result = fun(*args, **kwargs)
            except FloatingPointError as e:
                raise ValueError(
                    f"Your linear predictors are not supported for p={args[0].p}. For "
                    + "negative linear predictors, consider using a log link instead."
                ) from e
        return result

    return _to_return


class TweedieLink(Link):
    """The Tweedie link function ``x^(1-p)`` if ``p≠1`` and ``log(x)`` if ``p=1``."""

    def __new__(cls, p: float):
        """
        Create a new ``TweedieLink`` object.

        Parameters
        ----------
        p: scalar
        """
        if p == 0:
            return IdentityLink()
        if p == 1:
            return LogLink()
        return super().__new__(cls)

    def __init__(self, p):
        self.p = p

    def __eq__(self, other):  # noqa D
        return isinstance(other, self.__class__) and (self.p == other.p)

    def link(self, mu):
        """
        Get the Tweedie canonical link.

        See superclass documentation.

        Parameters
        ----------
        mu: array-like
        """
        return _asanyarray(mu) ** (1 - self.p)

    def derivative(self, mu):
        """
        Get the derivative of the Tweedie link.

        See superclass documentation.

        Parameters
        ----------
        mu: array-like
        """
        return (1 - self.p) * _asanyarray(mu) ** (-self.p)

    @catch_p
    def inverse(self, lin_pred):
        """
        Get the inverse of the Tweedie link.

        See superclass documentation.

        Parameters
        ----------
        mu: array-like
        """
        return _asanyarray(lin_pred) ** (1 / (1 - self.p))

    @catch_p
    def inverse_derivative(self, lin_pred):
        """Compute the derivative of the inverse Tweedie link function ``h'(lin_pred)``.

        Parameters
        ----------
        lin_pred : array-like, shape (n_samples,)
            Usually the (fitted) linear predictor.
        """
        return (1 / (1 - self.p)) * _asanyarray(lin_pred) ** (self.p / (1 - self.p))

    @catch_p
    def inverse_derivative2(self, lin_pred):
        """Compute secondnd derivative of the inverse Tweedie link function \
            ``h''(lin_pred)``.

        Parameters
        ----------
        lin_pred : array, shape (n_samples,)
            Usually the (fitted) linear predictor.
        """
        result = _asanyarray(lin_pred) ** ((2 * self.p - 1) / (1 - self.p))
        result *= self.p / (1 - self.p) ** 2
        return result
