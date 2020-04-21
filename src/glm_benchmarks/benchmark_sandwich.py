import time
from typing import Any, Callable, Dict, Tuple

import numpy as np
import pandas as pd
from scipy import sparse as sps

from glm_benchmarks.problems import (
    load_simple_insurance_data,
    load_sparse_insurance_data,
)
from glm_benchmarks.sklearn_fork._glm import _safe_sandwich_dot


def load_data(n_rows: int, sparse: bool) -> Tuple[Any, np.ndarray]:
    if sparse:
        x = load_sparse_insurance_data(n_rows)["X"].tocsc()
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


def run_one_problem_all_methods(n_rows: int, sparse: bool) -> pd.DataFrame:
    x, d = load_data(n_rows, sparse)
    funcs: Dict[str, Callable[[Any, np.ndarray], Any]] = {
        "naive": naive_sandwich,
        "mkl": _safe_sandwich_dot,
        "whiten": whitened_sandwich,
    }

    info: Dict[str, Any] = {}
    for name, func in funcs.items():
        start = time.perf_counter()
        res = func(x, d)
        elapsed = time.perf_counter() - start

        info[name] = {}
        info[name]["res"] = res
        info[name]["time"] = elapsed

    naive_result = info["naive"]["res"]
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
    row_counts = [int(1e5), int(1e6), 2 * int(1e6)]
    benchmarks = [(n_rows, sp) for n_rows in row_counts for sp in [False, True]]
    result_df = pd.concat(
        [run_one_problem_all_methods(*bench) for bench in benchmarks]
    ).sort_values(["n_rows", "sparse", "method"])

    print(result_df.set_index(["n_rows", "sparse", "method"]))
    return


if __name__ == "__main__":
    main()
