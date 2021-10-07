import pkg_resources

from ._distribution import TweedieDistribution
from ._glm import GeneralizedLinearRegressor
from ._glm_cv import GeneralizedLinearRegressorCV

try:
    __version__ = pkg_resources.get_distribution(__name__).version
except Exception:
    __version__ = "unknown"

__all__ = [
    "GeneralizedLinearRegressor",
    "TweedieDistribution",
    "GeneralizedLinearRegressorCV",
]
