import gc
import multiprocessing as mp
import time
import warnings
from threading import Thread
from typing import Optional

import numpy as np
import pandas as pd
import psutil
import quantcore.matrix as mx
import scipy.sparse as sps
from quantcore.glm_benchmarks.cli_run import get_all_problems

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
            time.sleep(1e-4)

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


def get_x_bytes(x) -> int:
    if isinstance(x, np.ndarray):
        return x.nbytes
    if sps.issparse(x):
        return sum(
            [
                mat.data.nbytes + mat.indices.nbytes + mat.indptr.nbytes
                for mat in [x, x.x_csr]
            ]
        )
    if isinstance(x, mx.CategoricalMatrix):
        return x.indices.nbytes
    if isinstance(x, mx.SplitMatrix):
        return sum([get_x_bytes(elt) for elt in x.matrices])
    raise NotImplementedError(f"Can't get bytes for matrix of type {type(x)}.")


def runner(storage, copy_X: Optional[bool]):
    gc.collect()

    P = get_all_problems()["wide-insurance-no-weights-lasso-poisson"]
    dat = P.data_loader(num_rows=100000, storage=storage)

    # Measure how much memory we are using before calling the GLM code
    if isinstance(dat["X"], pd.DataFrame):
        X = dat["X"].to_numpy()
    elif sps.issparse(dat["X"]):
        X = mx.SparseMatrix(dat["X"])
    elif isinstance(dat["X"], mx.SplitMatrix):
        X = dat["X"]

    data_memory = get_x_bytes(X)

    y = dat["y"]
    del dat
    gc.collect()

    with MemoryPoller() as mp:
        for i in range(4):
            model = GeneralizedLinearRegressor(
                family="poisson",
                l1_ratio=1.0,
                alpha=0.01,
                force_all_finite=False,
                copy_X=copy_X,
            )
            model.fit(X=X, y=y)

        excess_memory_used = mp.max_memory - mp.initial_memory
        extra_to_initial_ratio = excess_memory_used / data_memory

        import matplotlib.pyplot as plt

        graph = np.array(mp.memory_usage) - mp.initial_memory
        plt.plot(graph)
        plt.savefig(f"figures/memory_{storage}_copy_{copy_X}.png")

        return extra_to_initial_ratio


def make_memory_usage_plots():
    # These values are around double the empirical extra memory used. They inc
    # They increase from dense->sparse->split->cat, because the matrix itself takes less
    # and less memory to store, so all the temporary vectors of length n_rows start to
    # dominate the memory usage.
    storage_allowed_ratio = {"dense": 0.1, "sparse": 0.45, "cat": 1.3, "split0.1": 0.55}
    for storage, allowed_ratio in storage_allowed_ratio.items():
        for copy_X in [False, True, None]:
            with mp.Pool(1) as p:
                extra_to_initial_ratio = p.starmap(runner, [(storage, copy_X)])[0]

            if copy_X is not None and copy_X:
                if extra_to_initial_ratio < 1:
                    warnings.warn(
                        f"Used less memory than expected with copy_X = True and "
                        f"data format {storage}. Memory exceeded initial memory by "
                        f"{extra_to_initial_ratio}."
                    )
            else:
                if extra_to_initial_ratio > allowed_ratio:
                    warnings.warn(
                        f"Used more memory than expected with copy_X = {copy_X} and "
                        f"data format {storage}. Memory exceeded initial memory by "
                        f"{extra_to_initial_ratio}; expected less than {allowed_ratio}."
                    )


if __name__ == "__main__":
    make_memory_usage_plots()
