import time
import numpy as np
import pandas as pd
from glm_benchmarks.spblas.mkl_spblas import dense_sandwich

@profile
def numpy_mkl(X, d):
    sqrtD = np.sqrt(d)[:, np.newaxis]
    x_d = X * sqrtD
    return x_d.T @ x_d

def bench(f, iter):
    ts = []
    for i in range(iter):
        start = time.time()
        out = f()
        ts.append(time.time() - start)
    return ts, out

def mn_run(m, n, iter):
    X = np.random.rand(n, m)
    d = np.random.rand(n)

    XF = np.asfortranarray(X)

    ts, true = bench(lambda: numpy_mkl(X,d), iter)
    mkl = np.min(ts)
    print('numpy_mkl', mkl)

    ts, out = bench(lambda: dense_sandwich(XF,d), iter)
    ds = np.min(ts)
    print('dense_sandwich', ds)

    np.testing.assert_almost_equal(true, out)
    return mkl, ds

def main():
    iter = 1
    ms = []
    ns = []
    mklt = []
    dst = []
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
    for m in [2048]:
        for n in [2048]:
            mkl, ds = mn_run(m, n, iter)
            ms.append(m)
            ns.append(n)
            mklt.append(mkl)
            dst.append(ds)
    df = pd.DataFrame(dict(m = ms, n = ns, mkl = mklt, dense_sandwich=dst))
    df.set_index(['m', 'n'], inplace=True)
    print(df)




if __name__ == "__main__":
    main()
