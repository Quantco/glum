import numpy as np

from glum._distribution import TweedieDistribution

td = TweedieDistribution(power=1.5)

td.log_likelihood(
    y=np.array([0.1, 0.2], dtype=float),
    mu=np.array([1, 2], dtype=float),
    dispersion=1.0,
)
