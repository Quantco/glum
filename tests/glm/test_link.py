import numpy as np
import pytest

from glum._link import CloglogLink, IdentityLink, Link, LogitLink, LogLink, TweedieLink


@pytest.mark.parametrize("link", Link.__subclasses__())
def test_link_properties(link):
    """Test link inverse and derivative."""
    rng = np.random.RandomState(42)
    x = rng.rand(100) * 100

    if link.__name__ == "TweedieLink":
        link = link(1.5)
    else:
        link = link()  # instantiate object

    if isinstance(link, LogitLink):
        # careful for large x, note expit(36) = 1
        # limit max eta to 15
        x = x / 100 * 15
    if isinstance(link, CloglogLink):
        # limit max eta to 3
        # also check negative values as link is not symmetric
        x = x / 100 * 6 - 3

    np.testing.assert_allclose(link.link(link.inverse(x)), x)
    # if f(g(x)) = x, then f'(g(x)) = 1/g'(x)
    np.testing.assert_allclose(
        link.derivative(link.inverse(x)), 1.0 / link.inverse_derivative(x)
    )

    assert link.inverse_derivative2(x).shape == link.inverse_derivative(x).shape

    # Check inverse_derivative2 against numerical second derivative
    h = 1e-7
    numerical_d2 = (link.inverse_derivative(x + h) - link.inverse_derivative(x - h)) / (
        2 * h
    )
    np.testing.assert_allclose(
        link.inverse_derivative2(x), numerical_d2, rtol=1e-4, atol=1e-8
    )


def test_equality():
    assert IdentityLink() == IdentityLink()
    assert LogitLink() == LogitLink()
    assert LogLink() == LogLink()
    assert TweedieLink(0) != IdentityLink()
    assert TweedieLink(0) == IdentityLink().to_tweedie()
    assert TweedieLink(1.5) != LogitLink()
    assert TweedieLink(1.5) != TweedieLink(2.5)
    assert TweedieLink(1.5) == TweedieLink(1.5)
    assert TweedieLink(1) != LogLink()
    assert TweedieLink(1) == LogLink().to_tweedie()


@pytest.mark.parametrize("link", [LogitLink(), CloglogLink()], ids=["logit", "cloglog"])
def test_inverse_clips_silently(link):
    # Warnings are errors in the test suite, so this also checks that clipping does
    # not warn.
    eps50 = 50 * np.finfo(float).eps
    mu = link.inverse(np.array([-50.0, 0.0, 50.0]))
    assert np.all(mu >= eps50)
    assert np.all(mu <= 1 - eps50)


def test_is_saturated():
    eta = np.array([-50.0, 0.0, 5.0, 50.0])
    np.testing.assert_array_equal(
        LogitLink().is_saturated(eta), [True, False, False, True]
    )
    # The cloglog inverse reaches 1 at a linear predictor of about 3.5, which is
    # routine, so only the lower tail counts as saturated.
    np.testing.assert_array_equal(
        CloglogLink().is_saturated(eta), [True, False, False, False]
    )
    for link in [IdentityLink(), LogLink(), TweedieLink(1.5)]:
        assert not link.is_saturated(eta).any()
