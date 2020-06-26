import warnings
from abc import ABCMeta, abstractmethod

import numexpr
import numpy as np
from scipy import special


class Link(metaclass=ABCMeta):
    """Abstract base class for Link functions."""

    @abstractmethod
    def link(self, mu):
        """Compute the link function g(mu).

        The link function links the mean mu=E[Y] to the so called linear
        predictor (X*w), i.e. g(mu) = linear predictor.

        Parameters
        ----------
        mu : array, shape (n_samples,)
            Usually the (predicted) mean.
        """
        pass

    @abstractmethod
    def derivative(self, mu):
        """Compute the derivative of the link g'(mu).

        Parameters
        ----------
        mu : array, shape (n_samples,)
            Usually the (predicted) mean.
        """
        pass

    @abstractmethod
    def inverse(self, lin_pred):
        """Compute the inverse link function h(lin_pred).

        Gives the inverse relationship between linear predictor and the mean
        mu=E[Y], i.e. h(linear predictor) = mu.

        Parameters
        ----------
        lin_pred : array, shape (n_samples,)
            Usually the (fitted) linear predictor.
        """
        pass

    @abstractmethod
    def inverse_derivative(self, lin_pred):
        """Compute the derivative of the inverse link function h'(lin_pred).

        Parameters
        ----------
        lin_pred : array, shape (n_samples,)
            Usually the (fitted) linear predictor.
        """
        pass

    @abstractmethod
    def inverse_derivative2(self, lin_pred):
        """Compute 2nd derivative of the inverse link function h''(lin_pred).

        Parameters
        ----------
        lin_pred : array, shape (n_samples,)
            Usually the (fitted) linear predictor.
        """
        pass


class IdentityLink(Link):
    """The identity link function g(x)=x."""

    def link(self, mu):
        return mu

    def derivative(self, mu):
        return np.ones_like(mu)

    def inverse(self, lin_pred):
        return lin_pred

    def inverse_derivative(self, lin_pred):
        return np.ones_like(lin_pred)

    def inverse_derivative2(self, lin_pred):
        return np.zeros_like(lin_pred)


class LogLink(Link):
    """The log link function g(x)=log(x)."""

    def link(self, mu):
        return numexpr.evaluate("log(mu)")

    def derivative(self, mu):
        return numexpr.evaluate("1.0 / mu")

    def inverse(self, lin_pred):
        return numexpr.evaluate("exp(lin_pred)")

    def inverse_derivative(self, lin_pred):
        return numexpr.evaluate("exp(lin_pred)")

    def inverse_derivative2(self, lin_pred):
        return numexpr.evaluate("exp(lin_pred)")


class LogitLink(Link):
    """The logit link function g(x)=logit(x)."""

    def link(self, mu):
        return special.logit(mu)

    def derivative(self, mu):
        return 1.0 / (mu * (1 - mu))

    def inverse(self, lin_pred):
        """Note: since passing a very large value might result in an output
        of 1, this function bounds the output to be between
        [50*eps, 1 - 50*eps] where eps is floating point epsilon.
        """
        inv_logit = special.expit(lin_pred)
        eps50 = 50 * np.finfo(inv_logit.dtype).eps
        if np.any(inv_logit > 1 - eps50) or np.any(inv_logit < eps50):
            warnings.warn(
                "Computing sigmoid function gave results too close to 0 or 1. "
                "Clipping."
            )
            return np.clip(inv_logit, eps50, 1 - eps50)
        return inv_logit

    def inverse_derivative(self, lin_pred):
        ep = special.expit(lin_pred)
        return ep * (1.0 - ep)

    def inverse_derivative2(self, lin_pred):
        ep = special.expit(lin_pred)
        return ep * (1.0 - ep) * (1.0 - 2 * ep)
