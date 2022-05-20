# %%

# %load_ext autoreload
# %autoreload 2

# %%

import numpy as np
from libsvmdata import fetch_libsvm

from glum import GeneralizedLinearRegressor

X, y = fetch_libsvm("finance")

# %%

np.random.seed(42)
cols = np.random.randint(0, X.shape[1], size=(200000))

X_trunc = X[:, cols]

# %%

# NOTE: If you run this script as is, you'll get a memory error because
# use_sparse_hessian is not yet implemented in this branch (and 200,000 columns is
# too much for the original code to handle). If you have both the use_sparse_hessian
# and diag_fisher integrated already, then the following line (with
# use_sparse_hessian=True) will give you a good idea of how diag_fisher=True compares
# to diag_fisher=False in terms of runtime.

clf = GeneralizedLinearRegressor(
    family="gaussian", l1_ratio=1, alpha=0.1, diag_fisher=True
).fit(X_trunc, y)

# %%
