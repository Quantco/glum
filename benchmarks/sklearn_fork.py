import pandas as pd
from git_root import git_root

from glm_benchmarks.sklearn_fork import GeneralizedLinearRegressor

# get data
df = pd.read_parquet(git_root("data/data.parquet"))
import ipdb
ipdb.set_trace()
X = df[[col for col in df.columns if col not in ["y", "exposure"]]]
y = df["y"]
exposure = df["exposure"]

# inspect
X.shape

# run poisson regression, no regularization
model_free = GeneralizedLinearRegressor(family="poisson", alpha=0).fit(
    X=X, y=y, sample_weight=exposure
)

# run poisson regression, L1 regularization
model_lasso = GeneralizedLinearRegressor(family="poisson", alpha=1, l1_ratio=1).fit(
    X=X, y=y, sample_weight=exposure
)

# run poisson regression, L2 regularization
model_ridge = GeneralizedLinearRegressor(family="poisson", alpha=1, l1_ratio=0).fit(
    X=X, y=y, sample_weight=exposure
)

# run poisson regression, L1 & L2 regularization
model_elnet = GeneralizedLinearRegressor(family="poisson", alpha=1, l1_ratio=0.5).fit(
    X=X, y=y, sample_weight=exposure
)
