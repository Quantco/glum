from typing import Any, Dict

import numpy as np
from glm_benchmarks.utils import load_data, runtime
from glmnet_python import glmnet
from sklearn_fork import sklearn_fork_bench


def glmnet_python_bench(dat, distribution, alpha, l1_ratio) -> Dict[str, Any]:
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
    return result


def main():
    dat = load_data(nrows=1000)
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
        print(k, results[k]["runtime"])

    print(results["glmnet_python"]["coeffs"] - results["sklearn_fork"]["coeffs"])
    import ipdb

    ipdb.set_trace()


if __name__ == "__main__":
    main()
