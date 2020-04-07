import warnings

import numpy as np
import statsmodels.api as sm
from scipy import sparse as sps

from .util import runtime


def build_and_fit(model_args, fit_args):
    return sm.GLM(**model_args).fit_regularized(**fit_args)


def statsmodels_bench(dat, distribution, alpha, l1_ratio):

    result = dict()

    if isinstance(dat["X"], sps.spmatrix):
        warnings.warn("statsmodels does not support sparse matrices")
        return result

    dat["X"] = sm.add_constant(dat["X"])

    nb_coefs = dat["X"].shape[1]
    alpha_sm = np.zeros(nb_coefs)
    alpha_sm[1:] = alpha

    dist_map = {
        "poisson": sm.families.Poisson(),
        "gamma": sm.families.Gamma(),
        "tweedie": sm.families.Tweedie(),
        "binomial": sm.families.Binomial(),
        "gaussian": sm.families.Gaussian(),
    }

    model_args = dict(endog=dat["y"], exog=dat["X"], family=dist_map[distribution],)
    if "weights" in dat.keys():
        model_args.update({"freq_weights": dat["weights"]})

    fit_args = dict(alpha=alpha_sm, L1_wt=l1_ratio, maxiter=10000, cnvrg_tol=1e-3,)

    result["runtime"], m = runtime(build_and_fit, model_args, fit_args)
    result["model_obj"] = m
    result["intercept"] = m.params[0]
    result["coef"] = m.params[1:]

    warnings.warn(
        """statsmodels does not store the number of iterations. Runtime per iteration
        will not be accurate."""
    )
    result["n_iter"] = 1

    return result
