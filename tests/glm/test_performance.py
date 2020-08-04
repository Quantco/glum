import gc
import multiprocessing as mp
import time
from threading import Thread

import numpy as np
import pandas as pd
import psutil
import pytest
import quantcore.matrix as mx
import scipy.sparse as sps
from quantcore.glm_benchmarks.cli_run import get_all_problems
from quantcore.glm_benchmarks.util import get_sklearn_family, runtime
from sparse_dot_mkl import dot_product_mkl

from quantcore.glm import GeneralizedLinearRegressor


def get_memory_usage():
    return psutil.Process().memory_info().rss


class MemoryPoller:
    """
    Example usage:

    with MemoryPoller() as mp:
        do some stuff here
        print('initial memory usage', mp.initial_memory)
        print('max memory usage', mp.max_memory)
        excess_memory_used = mp.max_memory - mp.initial_memory
    """

    def poll_max_memory_usage(self):
        while not self.stop_polling:
            self.memory_usage.append(get_memory_usage())
            self.max_memory = max(self.max_memory, self.memory_usage[-1])
            time.sleep(0.001)

    def __enter__(self):
        self.stop_polling = False
        self.max_memory = 0
        self.initial_memory = get_memory_usage()
        self.memory_usage = [self.initial_memory]
        self.t = Thread(target=self.poll_max_memory_usage)
        self.t.start()
        return self

    def __exit__(self, *excargs):
        self.stop_polling = True
        self.t.join()


def runner(storage):
    gc.collect()
    P = get_all_problems()["wide-insurance-no-weights-lasso-poisson"]
    dat = P.data_loader(num_rows=100000, storage=storage)

    # Measure how much memory we are using before calling the GLM code
    data_memory = 0
    if isinstance(dat["X"], pd.DataFrame):
        X = dat["X"].to_numpy()
        data_memory += X.nbytes
    elif sps.issparse(dat["X"]):
        X = mx.SparseMatrix(dat["X"])
        # In particular, make sure to count X.x_csr for sparse matrices.
        for mat in [X, X.x_csr]:
            data_memory += mat.data.nbytes + mat.indices.nbytes + mat.indptr.nbytes
    elif isinstance(dat["X"], mx.SplitMatrix):
        X = dat["X"]
        for m in dat["X"].matrices:
            if isinstance(m, mx.DenseMatrix):
                data_memory += m.nbytes
            elif isinstance(m, mx.SparseMatrix):
                for mat in [m, m.x_csr]:
                    data_memory += (
                        mat.data.nbytes + mat.indices.nbytes + mat.indptr.nbytes
                    )
            elif isinstance(m, mx.CategoricalMatrix):
                data_memory += m.indices.nbytes

    y = dat["y"]
    data_memory += y.nbytes
    del dat
    gc.collect()

    with MemoryPoller() as mp:
        for i in range(4):
            model = GeneralizedLinearRegressor(
                family="poisson",
                l1_ratio=1.0,
                alpha=0.01,
                copy_X=False,
                force_all_finite=False,
            )
            model.fit(X=X, y=y)

        excess_memory_used = mp.max_memory - mp.initial_memory
        extra_to_initial_ratio = excess_memory_used / data_memory

        # Comments intentionally left here for future memory usage debugging
        # purposes. These are a useful first pass to provide more information
        # when one of these tests is failing.
        # graph = np.array(mp.memory_usage) - mp.initial_memory
        # import matplotlib.pyplot as plt
        # plt.plot(graph)
        # plt.show()
        # print(data_memory / 1e6, extra_to_initial_ratio)

        return extra_to_initial_ratio


@pytest.mark.parametrize(
    "storage, allowed_ratio",
    [("dense", 0.1), ("sparse", 0.45), ("cat", 1.3), ("split0.1", 0.55)],
)
@pytest.mark.slow
def test_memory_usage(storage, allowed_ratio):
    # We run inside a separate process here in order to isolate memory
    # management issues so that the detritus from test #1 doesn't affect test
    # #2.
    with mp.Pool(1) as p:
        extra_to_initial_ratio = p.map(runner, [storage])[0]
    assert extra_to_initial_ratio < allowed_ratio


@pytest.fixture(scope="module")
def spmv_runtime():
    """
    Sparse matrix-vector product runtime should be representative of the memory
    bandwidth of the machine. We use MKL to make sure that this is
    parallelized. Otherwise, the performance will not scale properly in
    comparison to the GLM code for machines with many cores.
    """
    N = 20000000
    diag_data = np.random.rand(5, N)
    mat = sps.spdiags(diag_data, [0, 1, -1, 2, -2], N, N).tocsr()
    v = np.random.rand(N)
    return runtime(lambda: dot_product_mkl(mat, v), 5)[0]


@pytest.fixture(scope="module")
def dense_inv_runtime():
    """
    Dense matrix multiplication runtime should be representative of the
    floating point performance of the machine.
    """
    N = 1300
    X = np.random.rand(N, N)
    return runtime(lambda: np.linalg.inv(X), 5)[0]


def retry_on_except(n=3):
    def wrapper(fn):
        def test_inner(*args, **kwargs):
            for i in range(n):
                try:
                    fn(*args, **kwargs)
                except AssertionError:
                    if i >= n - 1:
                        raise
                else:
                    return

        return test_inner

    return wrapper


@retry_on_except()
def runtime_helper(
    spmv_runtime, dense_inv_runtime, storage, problem, distribution, num_rows, limit
):
    P = get_all_problems()[problem + "-" + distribution]
    dat = P.data_loader(num_rows=num_rows, storage=storage)

    family = get_sklearn_family(distribution)
    model = GeneralizedLinearRegressor(
        family=family, l1_ratio=1.0, alpha=0.01, copy_X=False, force_all_finite=False,
    )
    min_runtime, result = runtime(lambda: model.fit(X=dat["X"], y=dat["y"]), 5)

    # Let's just guess that we're about half flop-limited and half
    # memory-limited.  This is a decent guess because the sandwich product is
    # mostly flop-limited in the dense case and the dense case generally
    # dominates even when we're using split or categorical. On the other hand,
    # everything besides the sandwich product is probably memory limited.
    denominator = 0.5 * dense_inv_runtime + 0.5 * spmv_runtime
    print(
        spmv_runtime,
        dense_inv_runtime,
        min_runtime,
        denominator,
        min_runtime / denominator,
        limit,
    )
    assert min_runtime / denominator < limit


@pytest.mark.parametrize(
    "storage, problem, distribution, num_rows, limit",
    [
        ("dense", "narrow-insurance-no-weights-lasso", "poisson", 200000, 1.5),
        ("sparse", "narrow-insurance-weights-l2", "gaussian", 200000, 2.5),
        ("cat", "wide-insurance-no-weights-l2", "gamma", 100000, 2.5),
        ("split0.1", "wide-insurance-offset-lasso", "tweedie-p=1.5", 100000, 3.0),
        ("split0.1", "intermediate-insurance-no-weights-net", "binomial", 200000, 1.0),
    ],
)
def test_runtime(
    spmv_runtime, dense_inv_runtime, storage, problem, distribution, num_rows, limit
):
    runtime_helper(
        spmv_runtime, dense_inv_runtime, storage, problem, distribution, num_rows, limit
    )
