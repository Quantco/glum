import statsmodels.api as sm
from utils import runtime


def build_and_fit(model_args, fit_args):
    return sm.GLM(**model_args).fit_regularized(**fit_args)


def statsmodels_bench(dat, distribution, alpha, l1_ratio):

    result = dict()
    dat["X"] = sm.add_constant(dat["X"])

    dist_map = {
        "poisson": sm.families.Poisson(),
        "gamma": sm.families.Gamma(),
        "tweedie": sm.families.Tweedie(),
        "binomial": sm.families.Binomial(),
    }

    model_args = dict(
        endog=dat["y"],
        exog=dat["X"],
        family=dist_map[distribution],
        freq_weight=dat["exposure"],
    )

    fit_args = dict(alpha=alpha, L1_wt=l1_ratio, maxiter=10000, cnvrg_tol=1e-3,)

    result["runtime"], m = runtime(build_and_fit, model_args, fit_args)
    result["model_obj"] = m
    result["intercept"] = m.params["const"]
    result["coeffs"] = m.params[1:].values
    # result["n_iter"] = m.n_iter_

    # result["path"] = compute_path(m.n_iter_, model_args, fit_args)
    # np.testing.assert_almost_equal(
    #     result["path"][-1], result["coeffs"], -np.log10(model_args["tol"]) - 1
    # )
    return result
