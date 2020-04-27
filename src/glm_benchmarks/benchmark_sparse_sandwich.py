import time
from typing import Any, Callable, Dict, Tuple

import numpy as np
import pandas as pd
from scipy import sparse as sps

from glm_benchmarks.problems import load_narrow_insurance_data
from glm_benchmarks.sandwich.sandwich import sparse_sandwich


def load_data(n_rows: int) -> Tuple[Any, np.ndarray]:
    x = sps.csc_matrix(load_narrow_insurance_data(n_rows)["X"])
    np.random.seed(0)
    d = np.random.uniform(0, 1, n_rows)
    return x, d


def naive_sandwich(x, d):
    return (x.T @ (x.multiply(d[:, None]))).toarray()


def _fast_sandwich(X, d):
    return sparse_sandwich(X, X.XT, d)


def run_one_problem_all_methods(x, d, include_naive, dtype) -> pd.DataFrame:
    x = x.astype(dtype)
    d = d.astype(dtype)
    x.XT = x.T.tocsc()
    funcs: Dict[str, Callable[[Any, np.ndarray], Any]] = {
        "fast_sandwich": _fast_sandwich,
    }
    if include_naive:
        funcs["naive"] = naive_sandwich

    info: Dict[str, Any] = {}
    for name, func in funcs.items():
        ts = []
        for i in range(10):
            start = time.perf_counter()
            res = func(x, d)
            ts.append(time.perf_counter() - start)
        elapsed = np.min(ts)

        info[name] = {}
        info[name]["res"] = res
        info[name]["time"] = elapsed

    if include_naive:
        naive_result = info["naive"]["res"]
        for k in info.keys():
            np.testing.assert_allclose(naive_result, info[k]["res"], 4)

    return pd.DataFrame(
        {"method": list(info.keys()), "time": [elt["time"] for elt in info.values()]}
    )


def main() -> None:
    # "killed" with 1e7 and 4 * 1e6
    row_counts = [
        int(1e4),
        int(1e5),
        int(3e5),
        int(1e6),
    ]  # , int(2e6), int(4e6), int(10e6)]
    benchmarks = []

    x, d = load_data(row_counts[-1])

    for i, n_rows in enumerate(row_counts):
        for dtype in [np.float32, np.float64]:
            benchmarks.append(
                run_one_problem_all_methods(
                    x[:n_rows, :].copy(), d[:n_rows].copy(), i == 0, dtype
                )
            )
            benchmarks[-1]["dtype"] = str(dtype)
            benchmarks[-1]["n_rows"] = n_rows

    result_df = pd.concat(benchmarks).sort_values(["n_rows", "method"])
    print(result_df.set_index(["n_rows", "method"]))


if __name__ == "__main__":
    main()
