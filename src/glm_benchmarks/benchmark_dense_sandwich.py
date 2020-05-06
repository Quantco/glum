import time

import numpy as np
import pandas as pd

from glm_benchmarks.sandwich.sandwich import dense_sandwich


def numpy_mklC(X, d):
    sqrtD = np.sqrt(d)[:, np.newaxis]
    x_d = X[0] * sqrtD
    return x_d.T @ x_d


def numpy_mklF(X, d):
    sqrtD = np.sqrt(d)[:, np.newaxis]
    x_d = X[1] * sqrtD
    return x_d.T @ x_d


def bench(f, iter):
    ts = []
    for i in range(iter):
        start = time.time()
        out = f()
        ts.append(time.time() - start)
    return ts, out


def _dense_sandwich(X, d):
    return dense_sandwich(X[1], d)


def mn_run(m, n, iter, dtype):
    precision = dtype().itemsize * 8
    X = [np.random.rand(n, m).astype(dtype=dtype)]
    d = np.random.rand(n).astype(dtype=dtype)

    X.append(np.asfortranarray(X[0]))

    out = dict()
    out["name"] = []
    out["runtime"] = []
    to_run = [
        "numpy_mklC",
        # "numpy_mklF",
        "_dense_sandwich",
    ]
    for name in to_run:
        ts, result = bench(lambda: globals()[name](X, d), iter)
        if name == "numpy_mklC":
            true = result
        elif "numpy_mklC" in to_run:
            err = np.abs((true - result) / true)
            np.testing.assert_almost_equal(err, 0, 4 if precision == 32 else 7)
        runtime = np.min(ts)
        out["name"].append(name)
        out["runtime"].append(runtime)
        print(name, runtime)
    out_df = pd.DataFrame(out)
    out_df["m"] = m
    out_df["n"] = n
    out_df["precision"] = precision
    return out_df


def main():
    iter = 50
    # for m in [10, 30, 100, 300,  1000]:
    #     for p in np.arange(4, 6):
    # n = int(10 ** p)
    # for m in [10, 48, 100, 1000]:
    #     for p in np.arange(4, 6):
    # n = int(10 ** p)
    # for m in [1000]:
    #     for p in [3.5]:
    # n = int(10 ** p)
    # for m in [10, 30, 100, 300, 1000]:
    # for m in [300]:
    #     for n in [100000]:#, 1000000]:
    Rs = []
    for m, n in [
        (20, 1000000),
        (50, 500000),
        # (150, 200000),
        # (300, 100000),
        # (2048, 2048),
        (1500, 1500),
    ]:
        for dt in [np.float64]:
            Rs.append(mn_run(m, n, iter, dt))
    df = pd.concat(Rs)
    df.set_index(["m", "n", "name", "precision"], inplace=True)
    df.sort_index(inplace=True)
    print(df)


if __name__ == "__main__":
    main()
