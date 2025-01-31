import numpy as np
import pytest

from glum._distribution import (
    BinomialDistribution,
    ExponentialDispersionModel,
    GammaDistribution,
    GeneralizedHyperbolicSecant,
    InverseGaussianDistribution,
    NegativeBinomialDistribution,
    NormalDistribution,
    PoissonDistribution,
    TweedieDistribution,
    guess_intercept,
)
from glum._link import IdentityLink, Link, LogitLink, LogLink


@pytest.mark.parametrize(
    "link, distribution, tol",
    [
        (IdentityLink(), NormalDistribution(), 1e-4),
        (LogLink(), PoissonDistribution(), 1e-4),
        (LogLink(), GammaDistribution(), 1e-4),
        (LogLink(), TweedieDistribution(1.5), 1e-4),
        (LogLink(), TweedieDistribution(4.5), 1e-4),
        (LogLink(), NormalDistribution(), 1e-4),
        (LogLink(), InverseGaussianDistribution(), 1e-4),
        (LogLink(), NegativeBinomialDistribution(), 1e-2),
        (LogitLink(), BinomialDistribution(), 1e-2),
        (IdentityLink(), GeneralizedHyperbolicSecant(), 1e-1),
    ],
)
@pytest.mark.parametrize("offset", [None, np.array([0.3, -0.1, 0, 0.1]), 0.1])
def test_get_best_intercept(
    link: Link, distribution: ExponentialDispersionModel, tol: float, offset
):
    y = np.array([1, 1, 1, 2], dtype=np.float64)
    if isinstance(distribution, BinomialDistribution):
        y -= 1

    sample_weight = np.array([0.1, 0.2, 5, 1])
    best_intercept = guess_intercept(y, sample_weight, link, distribution, offset)
    assert np.isfinite(best_intercept)

    def _get_dev(intercept):
        eta = intercept if offset is None else offset + intercept
        mu = link.inverse(eta)
        assert np.isfinite(mu).all()
        return distribution.deviance(y, mu, sample_weight)

    obj = _get_dev(best_intercept)
    obj_low = _get_dev(best_intercept - tol)
    obj_high = _get_dev(best_intercept + tol)
    assert obj < obj_low
    assert obj < obj_high
