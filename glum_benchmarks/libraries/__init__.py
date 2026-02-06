"""Benchmark functions for different GLM libraries."""

# Always available
from .bench_glum import glum_bench
from .bench_zeros import zeros_bench

# Optional dependencies, only import if available
try:
    from .bench_celer import celer_bench
except ImportError:
    celer_bench = None  # type: ignore

try:
    from .bench_h2o import h2o_bench
except ImportError:
    h2o_bench = None  # type: ignore

try:
    from .bench_skglm import skglm_bench
except ImportError:
    skglm_bench = None  # type: ignore

try:
    from .bench_sklearn import sklearn_bench
except ImportError:
    sklearn_bench = None  # type: ignore

try:
    from .bench_glmnet import glmnet_bench
except ImportError:
    glmnet_bench = None  # type: ignore

__all__ = [
    "celer_bench",
    "glum_bench",
    "h2o_bench",
    "skglm_bench",
    "sklearn_bench",
    "glmnet_bench",
    "zeros_bench",
]
