from typing import Union

import numpy as np
import pytest

from quantcore.glm._distribution import (
    ExponentialDispersionModel,
    GammaDistribution,
    InverseGaussianDistribution,
    NormalDistribution,
    PoissonDistribution,
    TweedieDistribution,
)


@pytest.mark.parametrize(
    "distribution, expected",
    [
        (NormalDistribution(), -np.inf),
        (PoissonDistribution(), 0),
        (TweedieDistribution(power=-0.5), -np.inf),
        (GammaDistribution(), 0),
        (InverseGaussianDistribution(), 0),
        (TweedieDistribution(power=1.5), 0),
    ],
)
def test_lower_bounds(
    distribution: ExponentialDispersionModel, expected: Union[float, int]
):
    assert distribution.lower_bound == expected
