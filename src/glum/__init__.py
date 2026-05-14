import importlib.metadata

from ._distribution import (
    BinomialDistribution,
    ExponentialDispersionModel,
    GammaDistribution,
    GeneralizedHyperbolicSecant,
    InverseGaussianDistribution,
    NegativeBinomialDistribution,
    NormalDistribution,
    PoissonDistribution,
    TweedieDistribution,
)
from ._glm import GeneralizedLinearRegressor, get_family, get_link
from ._glm_cv import GeneralizedLinearRegressorCV
from ._link import CloglogLink, IdentityLink, Link, LogitLink, LogLink, TweedieLink
from ._stepwise import CVResult, ScoreTestResult, StepwiseGLM
from ._tabmat_cache import CacheVersionError, TabmatCache

try:
    __version__ = importlib.metadata.distribution(__name__).version
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
    "NegativeBinomialDistribution",
    "IdentityLink",
    "Link",
    "LogitLink",
    "LogLink",
    "TweedieLink",
    "CloglogLink",
    "GeneralizedLinearRegressor",
    "GeneralizedLinearRegressorCV",
    "StepwiseGLM",
    "ScoreTestResult",
    "CVResult",
    "TabmatCache",
    "CacheVersionError",
    "get_family",
    "get_link",
]
