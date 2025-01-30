from typing import Union

import numpy as np

from ._distribution import (
    ExponentialDispersionModel,
    GammaDistribution,
    InverseGaussianDistribution,
    NormalDistribution,
    PoissonDistribution,
    TweedieDistribution,
)
from ._link import IdentityLink, Link, LogitLink, LogLink


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
