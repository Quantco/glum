import time
from typing import Any, Callable, Dict, Tuple

import numpy as np
import pandas as pd
from scipy import sparse as sps

from glm_benchmarks.problems import load_simple_insurance_data
from glm_benchmarks.spblas.mkl_spblas import (
    fast_matmul,
    fast_matmul2,
    fast_matmul3,
    mkl_matmat,
)


def load_data(n_rows: int, sparse: bool) -> Tuple[Any, np.ndarray]:
    if sparse:
        x = sps.csc_matrix(load_simple_insurance_data(n_rows)["X"])
    else:
        x = load_simple_insurance_data(n_rows)["X"]
    np.random.seed(0)
    d = np.random.uniform(0, 1, n_rows)
    return x, d


def naive_sandwich(x, d):
    if sps.isspmatrix(x):
        return (x.T @ (x.multiply(d[:, None]))).toarray()
    return x.T @ (x * d[:, None])


def whitened_sandwich(x, d):
    sqrt_d = np.sqrt(d)
    if sps.isspmatrix(x):
        if not isinstance(x, sps.csc_matrix):
            x = x.tocsc()
        sqrt_d_long = sqrt_d[x.indices]
        x.data *= sqrt_d_long
        res = x.T.dot(x)
        x.data /= sqrt_d_long
        return res.toarray()

    sqrt_d = sqrt_d[:, None]
    x *= sqrt_d
    res = x.T.dot(x)
    x /= sqrt_d
    return res


def _safe_sandwich_dot_whitened(X, d):
    sqrt_d = np.sqrt(d)
    if sps.isspmatrix(X):
        sqrt_d_long = sqrt_d[X.indices]
        X.data *= sqrt_d_long
        temp = mkl_matmat(X, X, transpose=True, return_dense=True)
        X.data /= sqrt_d_long
        return temp

    sqrt_d = sqrt_d[:, None]
    X *= sqrt_d
    res = X.T.dot(X)
    X /= sqrt_d
    return res


def _safe_sandwich_dot(X, d):
    if sps.isspmatrix(X):
        if X.getformat() == "csr":
            X2 = X.multiply(d[:, np.newaxis]).tocsr()
        elif X.getformat() == "csc":
            X2 = X.multiply(d[:, np.newaxis]).tocsc()
        return mkl_matmat(X, X2, transpose=True, return_dense=True,)
    return whitened_sandwich(X, d)


def _fast_matmul(X, d):
    if sps.isspmatrix(X):
        temp = fast_matmul(X.data, X.indices, X.indptr, d)
        temp += np.tril(temp, -1).T
        return temp
    return whitened_sandwich(X, d)


def _fast_matmul2(X, d):
    # X.XT = X.T.tocsc()
    if sps.isspmatrix(X):
        temp = fast_matmul2(
            X.data, X.indices, X.indptr, X.XT.data, X.XT.indices, X.XT.indptr, d
        )
        return temp
    return whitened_sandwich(X, d)


def _fast_matmul3(X, d):
    # X.XT = X.T.tocsc()
    if sps.isspmatrix(X):
        temp = fast_matmul3(
            X.data, X.indices, X.indptr, X.XT.data, X.XT.indices, X.XT.indptr, d
        )
        return temp
    return whitened_sandwich(X, d)


def run_one_problem_all_methods(n_rows: int, sparse: bool) -> pd.DataFrame:
    x, d = load_data(n_rows, sparse)
    x.XT = x.T.tocsc()
    funcs: Dict[str, Callable[[Any, np.ndarray], Any]] = {
        # "naive": naive_sandwich,
        "mkl": _safe_sandwich_dot,
        "mkl_whitened": _safe_sandwich_dot_whitened,
        # "whiten": whitened_sandwich,
        "fast_matmul": _fast_matmul,
        "fast_matmul2": _fast_matmul2,
        # "fast_matmul3": _fast_matmul3,
    }

    info: Dict[str, Any] = {}
    for name, func in funcs.items():
        start = time.perf_counter()
        res = func(x, d)
        elapsed = time.perf_counter() - start

        info[name] = {}
        info[name]["res"] = res
        info[name]["time"] = elapsed

    naive_result = info["mkl"]["res"]
    for k in info.keys():
        np.testing.assert_allclose(naive_result, info[k]["res"])

    return pd.DataFrame(
        {
            "method": list(info.keys()),
            "time": [elt["time"] for elt in info.values()],
            "n_rows": n_rows,
            "sparse": sparse,
        }
    )


def main() -> None:
    # "killed" with 1e7 and 4 * 1e6
    row_counts = [int(1e4), int(1e5), int(3e5), int(1e6), int(2e6), int(4e6)]
    # benchmarks = [(n_rows, sp) for n_rows in row_counts for sp in [False, True]]
    benchmarks = [(n_rows, sp) for n_rows in row_counts for sp in [True]]
    result_df = pd.concat(
        [run_one_problem_all_methods(*bench) for bench in benchmarks]
    ).sort_values(["n_rows", "sparse", "method"])

    print(result_df.set_index(["n_rows", "sparse", "method"]))
    return


if __name__ == "__main__":
    main()
