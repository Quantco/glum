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
        return special.expit(lin_pred)

    def inverse_derivative(self, lin_pred):
        ep = special.expit(lin_pred)
        return ep * (1.0 - ep)

    def inverse_derivative2(self, lin_pred):
        ep = special.expit(lin_pred)
        return ep * (1.0 - ep) * (1.0 - 2 * ep)


def get_best_intercept(
    y: np.ndarray, weights: np.ndarray, link: Link, offset: np.ndarray = None
):
    """
    For the identity or log link functions, the solution can be written as
    intercept = link.link(avg(y)) - link.link(avg(link.inverse(eta)))
    """
    if offset is None:
        return link.link(np.average(y, weights=weights))
    return link.link(np.average(y, weights=weights)) - link.link(
        np.average(link.inverse(offset), weights=weights)
    )
