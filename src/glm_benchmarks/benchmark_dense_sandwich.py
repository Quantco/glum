import time

import numpy as np
import pandas as pd

from glm_benchmarks.sandwich.sandwich import dense_sandwich, dense_sandwich2


def numpy_mkl(X, d):
    sqrtD = np.sqrt(d)[:, np.newaxis]
    x_d = X[0] * sqrtD
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


def _dense_sandwich2(X, d):
    return dense_sandwich2(X[0], X[1], d)


def mn_run(m, n, iter):
    X = [np.random.rand(n, m)]
    d = np.random.rand(n)

    X.append(np.asfortranarray(X[0]))
    print(len(X))

    out = dict()
    out["name"] = []
    out["runtime"] = []
    for name in ["numpy_mkl", "_dense_sandwich", "_dense_sandwich2"]:
        ts, result = bench(lambda: globals()[name](X, d), iter)
        if name == "numpy_mkl":
            true = result
        else:
            np.testing.assert_almost_equal(result, true)
        runtime = np.min(ts)
        out["name"].append(name)
        out["runtime"].append(runtime)
        print(name, runtime)
    out_df = pd.DataFrame(out)
    out_df["m"] = m
    out_df["n"] = n
    return out_df


def main():
    iter = 10
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
        # (20, 1000000),
        # (50, 500000),
        # (150, 200000),
        # (300, 100000),
        # (2048, 2048),
    ]:
        Rs.append(mn_run(m, n, iter))
    df = pd.concat(Rs)
    df.set_index(["m", "n"], inplace=True)
    print(df)


if __name__ == "__main__":
    main()
