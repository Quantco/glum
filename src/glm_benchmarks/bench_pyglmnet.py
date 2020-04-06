import numpy as np
import pyglmnet

from .util import runtime


def build_and_fit(model_args, fit_args):
    pyglmnet.GLM(**model_args).fit(*fit_args)


def pyglmnet_bench(dat, distribution, alpha, l1_ratio):
    result = dict()
    result["runtime"], m = runtime(
        build_and_fit,
        dict(
            family=distribution,
            alpha=l1_ratio,
            reg_lambda=np.array([alpha]),
            standardize=False,
            thresh=1e-7,
        )[dat["X"].values.copy(), dat["y"].values.copy()],
    )
    result["model_obj"] = m
    result["intercept"] = m["a0"]
    result["coef"] = m["beta"][:, 0]
    result["n_iter"] = m["npasses"]
    return result
