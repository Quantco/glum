import time
from typing import Any, Callable, Dict, Tuple

import numpy as np
import pandas as pd
from scipy import sparse as sps

from glm_benchmarks.problems import load_simple_insurance_data
from glm_benchmarks.sandwich.sandwich import sparse_sandwich


def load_data(n_rows: int) -> Tuple[Any, np.ndarray]:
    x = sps.csc_matrix(load_simple_insurance_data(n_rows)["X"])
    np.random.seed(0)
    d = np.random.uniform(0, 1, n_rows)
    return x, d


def naive_sandwich(x, d):
    return (x.T @ (x.multiply(d[:, None]))).toarray()

def _fast_sandwich(X, d):
    return sparse_sandwich(X, X.XT, d)

def run_one_problem_all_methods(x, d) -> pd.DataFrame:
    x.XT = x.T.tocsc()
    funcs: Dict[str, Callable[[Any, np.ndarray], Any]] = {
        "naive": naive_sandwich,
        "fast_sandwich": _fast_sandwich,
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
        }
    )


def main() -> None:
    # "killed" with 1e7 and 4 * 1e6
    row_counts = [int(1e4), int(1e5), int(3e5)]#, int(1e6), int(2e6), int(4e6)]
    benchmarks = []
    x, d = load_data(row_counts[-1])
    for n_rows in row_counts:
        benchmarks.append(run_one_problem_all_methods(x[:n_rows,:].copy(), d[:n_rows].copy()))
        benchmarks[-1]['n_rows'] = n_rows
    result_df = pd.concat(benchmarks).sort_values(["n_rows", "method"])
    print(result_df.set_index(["n_rows", "method"]))


if __name__ == "__main__":
    main()
