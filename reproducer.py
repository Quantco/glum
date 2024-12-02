import pandas as pd
import tabmat as tm
import numpy as np
from glum import GeneralizedLinearRegressor

df = pd.DataFrame(
    {
        "x1": [2, 3, 4, 1, 1, 1] * 1000,
        "x2": [2, 3, 4, 1, 1, 1] * 1000,
        "x3": [2, 3, 4, 1, 1, 1] * 1000,
        "x4": [0, 0, 0, 1, 0, 1] * 1000,
    }
)

y = np.array([1, 1, 1, 1, 0, 0] * 1000)
X = tm.from_pandas(df, sparse_threshold=0.0)

model = GeneralizedLinearRegressor(family="binomial", alpha=1).fit(X, y)
print(model.coef_.tolist())

model = GeneralizedLinearRegressor(family="binomial", alpha=1).fit(X, y)
print(model.coef_.tolist())
