import numpy as np
from glmnet_python import glmnet

from .util import runtime


def glmnet_python_bench(dat, distribution, alpha, l1_ratio):
    result = dict()
    glmnet_kws = dict(
        x=dat["X"].values.copy(),
        y=dat["y"].values.copy(),
        family=distribution,
        alpha=l1_ratio,
        lambdau=np.array([alpha]),
        standardize=False,
        thresh=1e-7,
    )
    if "weights" in dat.keys():
        glmnet_kws.update({"weights": dat["weights"].values})

    result["runtime"], m = runtime(glmnet, weights=dat["weights"].values, **glmnet_kws)

    result["model_obj"] = m
    result["intercept"] = m["a0"]
    result["coef"] = m["beta"][:, 0]
    result["n_iter"] = m["npasses"]
    return result
