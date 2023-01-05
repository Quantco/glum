import numpy as np
import pytest

from glum._link import Link, LogitLink, LogLink, TweedieLink


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

    np.testing.assert_allclose(link.link(link.inverse(x)), x)
    # if f(g(x)) = x, then f'(g(x)) = 1/g'(x)
    np.testing.assert_allclose(
        link.derivative(link.inverse(x)), 1.0 / link.inverse_derivative(x)
    )

    assert link.inverse_derivative2(x).shape == link.inverse_derivative(x).shape


def test_equality():
    assert TweedieLink(1.5) == TweedieLink(1.5)
    assert TweedieLink(1) == LogLink()
    assert LogLink() == LogLink()
    assert TweedieLink(1.5) != TweedieLink(2.5)
    assert TweedieLink(1.5) != LogitLink()
    assert LogitLink() == LogitLink()
