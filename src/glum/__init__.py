import pkg_resources

from ._distribution import (
    BinomialDistribution,
    ExponentialDispersionModel,
    GammaDistribution,
    GeneralizedHyperbolicSecant,
    InverseGaussianDistribution,
    NormalDistribution,
    PoissonDistribution,
    TweedieDistribution,
)
from ._glm import GeneralizedLinearRegressor, get_family, get_link
from ._glm_cv import GeneralizedLinearRegressorCV
from ._link import IdentityLink, Link, LogitLink, LogLink, TweedieLink

try:
    __version__ = pkg_resources.get_distribution(__name__).version
except Exception:
    __version__ = "unknown"

__all__ = [
    "BinomialDistribution",
    "ExponentialDispersionModel",
    "GammaDistribution",
    "GeneralizedHyperbolicSecant",
    "InverseGaussianDistribution",
    "NormalDistribution",
    "PoissonDistribution",
    "TweedieDistribution",
    "IdentityLink",
    "Link",
    "LogitLink",
    "LogLink",
    "TweedieLink",
    "GeneralizedLinearRegressor",
    "GeneralizedLinearRegressorCV",
    "get_family",
    "get_link",
]
