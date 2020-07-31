import gc
import multiprocessing as mp
import time
from threading import Thread

import pandas as pd
import psutil
import pytest
import scipy.sparse as sps


def get_memory_usage():
    return psutil.Process().memory_info().rss


class MemoryPoller:
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
    # Once issue #286 is solved, these imports can be moved outside the runner
    # function. For now, there will be an indefinite hang if the imports are
    # moved outside.
    from quantcore.glm_benchmarks.cli_run import get_all_problems
    from quantcore.glm import GeneralizedLinearRegressor
    import quantcore.matrix as mx

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
    with mp.Pool(1) as p:
        extra_to_initial_ratio = p.map(runner, [storage])[0]
    assert extra_to_initial_ratio < allowed_ratio
