import gc
import multiprocessing as mp
import time
from threading import Thread

import pandas as pd
import psutil
import pytest
from quantcore.glm_benchmarks.cli_run import get_all_problems
from quantcore.matrix import SparseMatrix

from quantcore.glm import GeneralizedLinearRegressor


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
    print(get_memory_usage())
    gc.collect()
    P = get_all_problems()["wide-insurance-no-weights-lasso-poisson"]
    dat = P.data_loader(num_rows=1000, storage=storage)
    # Measure how much memory we are using before calling the GLM code
    data_memory = 0
    if isinstance(dat["X"], pd.DataFrame):
        X = dat["X"].to_numpy()
        data_memory += X.nbytes
    else:
        X = SparseMatrix(dat["X"])
        # In particular, make sure to count X.x_csr for sparse matrices.
        for mat in [X, X.x_csr]:
            data_memory += mat.data.nbytes + mat.indices.nbytes + mat.indptr.nbytes
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

        # graph = np.array(mp.memory_usage) - mp.initial_memory
        # import matplotlib.pyplot as plt
        # plt.plot(graph)
        # plt.show()
        print(extra_to_initial_ratio)
        return extra_to_initial_ratio


@pytest.mark.parametrize("storage", ["dense", "sparse"])
def test_memory_usage(storage):
    allowed_ratio = 0.05 if storage == "dense" else 0.4
    with mp.Pool(1) as p:
        extra_to_initial_ratio = p.map(runner, [storage])
    assert extra_to_initial_ratio < allowed_ratio
