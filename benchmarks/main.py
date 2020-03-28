import pandas as pd
import numpy as np
import git_root

def load_data(nrows=None):
    df = pd.read_parquet(git_root.git_root("data/data.parquet"))
    if nrows is not None:
        df = df.iloc[:nrows]
    X = df[[col for col in df.columns if col not in ["y", "exposure"]]]
    y = df["y"]
    exposure = df["exposure"]
    return dict(X=X, y=y, exposure=exposure)

from glmnet_python import glmnet
def glmnet_python_bench(dat, distribution, alpha, l1_ratio):
    m = glmnet(
        x=dat['X'].values.copy(),
        y=dat['y'].values.copy(),
        weights=dat['exposure'].values,
        family=distribution,
        alpha=l1_ratio,
        lambdau=np.ones(1) * alpha,
        standardize=False
        # nlambda=1,
        # lambda_min=np.array([alpha])
    )
    result = dict()
    result['model_obj'] = m
    result['intercept'] = m['a0']
    result['coeffs'] = m['beta'][:,0]
    return result

from glm_benchmarks.sklearn_fork import GeneralizedLinearRegressor
def sklearn_fork_bench(dat, distribution, alpha, l1_ratio):
    result = dict()
    m = GeneralizedLinearRegressor(
        family=distribution,
        alpha=alpha,
        l1_ratio=l1_ratio,
        max_iter=10000
    ).fit(
        X=dat['X'], y=dat['y'],
        sample_weight=dat['exposure']
    )
    result['model_obj'] = m
    result['intercept'] = m.intercept_
    result['coeffs'] = m.coef_
    return result

def main():
    dat = load_data(nrows=1000)
    benchmarks = dict(
        sklearn_fork = sklearn_fork_bench,
        glmnet_python = glmnet_python_bench
    )
    results = dict()
    for name, fnc in benchmarks.items():
        results[name] = fnc(dat, "poisson", 0.001, 0.5)
        print(results[name]['intercept'])
        print(results[name]['coeffs'])
    import ipdb
    ipdb.set_trace()

if __name__ == "__main__":
    main()
