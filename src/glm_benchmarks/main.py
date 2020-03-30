import git_root
import numpy as np
import pandas as pd
from glmnet_python import glmnet

from glm_benchmarks.bench_sklearn_fork import runtime, sklearn_fork_bench


def load_data(nrows=None):
    df = pd.read_parquet(git_root.git_root("data/data.parquet"))
    if nrows is not None:
        df = df.iloc[:nrows]
    X = df[[col for col in df.columns if col not in ["y", "exposure"]]]
    y = df["y"]
    exposure = df["exposure"]
    return dict(X=X, y=y, exposure=exposure)


def glmnet_python_bench(dat, distribution, alpha, l1_ratio):
    result = dict()
    result["runtime"], m = runtime(
        glmnet,
        x=dat["X"].values.copy(),
        y=dat["y"].values.copy(),
        weights=dat["exposure"].values,
        family=distribution,
        alpha=l1_ratio,
        lambdau=np.array([alpha]),
        standardize=False,
        thresh=1e-7,
    )
    result["model_obj"] = m
    result["intercept"] = m["a0"]
    result["coeffs"] = m["beta"][:, 0]
    result["n_iter"] = m["npasses"]
    return result


def main():
    dat = load_data(nrows=None)
    benchmarks = dict(
        sklearn_fork=sklearn_fork_bench, glmnet_python=glmnet_python_bench
    )
    results = dict()
    np.set_printoptions(precision=4, suppress=True)
    for name, fnc in benchmarks.items():
        results[name] = fnc(dat, "poisson", 0.001, 0.5)
        print(results[name]["intercept"])
        print(results[name]["coeffs"])

    for k in results:
        print(k, "number of iterations", results[k]["n_iter"])
        print(k, "runtime", results[k]["runtime"])
        print(k, "runtime per iter", results[k]["runtime"] / results[k]["n_iter"])

    print(results["glmnet_python"]["coeffs"] - results["sklearn_fork"]["coeffs"])


if __name__ == "__main__":
    main()
