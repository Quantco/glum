import time
from typing import Any, Callable

import numpy as np
import pandas as pd
import tabmat as tm
from scipy import sparse as sps
from tabmat.ext.dense import dense_sandwich
from tabmat.ext.sparse import sparse_sandwich

from .problems import (
    generate_narrow_insurance_dataset,
    generate_wide_insurance_dataset,
    load_data,
)


def _load(which: str, n_rows: int) -> tuple[Any, np.ndarray]:
    if which == "narrow":
        x = sps.csc_matrix(load_data(generate_narrow_insurance_dataset, n_rows)["X"])
    else:
        x = sps.csc_matrix(load_data(generate_wide_insurance_dataset, n_rows)["X"])
    np.random.seed(0)
    d = np.random.uniform(0, 1, n_rows)
    return x, d


def _naive_sandwich(x, d):
    return (x.T @ (x.multiply(d[:, None]))).toarray()


def _fast_sandwich(X, d):
    return sparse_sandwich(X, X.XT, d)


def _split_sandwich(X, threshold):
    Xsplit = tm.from_split(X, threshold)
    return lambda _, d: Xsplit.sandwich(d)


def _dense_sandwich(X, d):
    return dense_sandwich(X.X_dense, d)


def _run_one_problem_all_methods(x, d, include_naive, dtype) -> pd.DataFrame:
    x = x.astype(dtype)
    d = d.astype(dtype)
    x.XT = x.T.tocsc()
    x.X_dense = np.asfortranarray(x.toarray())
    funcs: dict[str, Callable[[Any, np.ndarray], Any]] = {
        "sparse_sandwich": _fast_sandwich,
        "dense_sandwich": _dense_sandwich,
    }
    funcs["split_sandwich_0.05"] = _split_sandwich(x, 0.05)
    funcs["split_sandwich_0.1"] = _split_sandwich(x, 0.1)
    # for threshold in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3,0.4,0.5,0.7,0.9,1.0]:
    # for threshold in [0.01, 0.02, 0.03, 0.04, 0.05]:
    #     funcs[f"split_sandwich_{threshold}"] = split_sandwich(x, threshold)

    if include_naive:
        funcs["naive"] = _naive_sandwich

    info: dict[str, Any] = {}
    for name, func in funcs.items():
        ts = []
        for _ in range(7):
            start = time.perf_counter()
            res = func(x, d)
            ts.append(time.perf_counter() - start)
        elapsed = np.min(ts)  # type: ignore

        info[name] = {}
        info[name]["res"] = res
        info[name]["time"] = elapsed

    if include_naive:
        naive_result = info["naive"]["res"]
        for k in info:
            np.testing.assert_allclose(naive_result, info[k]["res"], 4)

    return pd.DataFrame(
        {"method": list(info.keys()), "time": [elt["time"] for elt in info.values()]}
    )


def main() -> None:
    """Run sparse sandwich benchmarks."""
    # "killed" with 1e7 and 4 * 1e6
    row_counts = [
        int(1e4),
        # int(1e5),
        # int(3e5),
        int(1e6),
        # int(2e6),
    ]  # , int(2e6), int(4e6), int(10e6)]
    benchmarks = []

    x, d = _load("narrow", row_counts[-1])

    for i, n_rows in enumerate(row_counts):
        for dtype in [np.float32, np.float64]:
            benchmarks.append(
                _run_one_problem_all_methods(
                    x[:n_rows, :].copy(), d[:n_rows].copy(), i == 0, dtype
                )
            )
            benchmarks[-1]["dtype"] = str(dtype)
            benchmarks[-1]["n_rows"] = n_rows

    result_df = pd.concat(benchmarks).sort_values(["n_rows", "method"])
    print(result_df.set_index(["n_rows", "method"]))


if __name__ == "__main__":
    main()
