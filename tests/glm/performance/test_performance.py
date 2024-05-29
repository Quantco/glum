import gc
import multiprocessing as mp
import time
import warnings
from threading import Thread
from typing import Optional

import numpy as np
import pandas as pd
import psutil
import scipy.sparse as sps
import tabmat as tm

from glum import GeneralizedLinearRegressor
from glum_benchmarks.cli_run import get_all_problems
from glum_benchmarks.util import get_sklearn_family, runtime


def _get_memory_usage() -> int:
    return psutil.Process().memory_info().rss


class MemoryPoller:
    """
    Sample memory and compute useful statistics.

    Example usage:

    with MemoryPoller() as mp:
        do some stuff here
        print('initial memory usage', mp.initial_memory)
        print('max memory usage', mp.max_memory)
        excess_memory_used = mp.max_memory - mp.initial_memory
    """

    def _poll_max_memory_usage(self):
        while not self.stop_polling:
            self.memory_usage.append(_get_memory_usage())
            self.max_memory: int = max(self.max_memory, self.memory_usage[-1])
            time.sleep(1e-4)

    def __enter__(self):
        """See example usage above."""
        self.stop_polling = False
        self.max_memory = 0
        self.initial_memory = _get_memory_usage()
        self.memory_usage: list[int] = [self.initial_memory]
        self.t = Thread(target=self._poll_max_memory_usage)
        self.t.start()
        return self

    def __exit__(self, *excargs):
        """Stop polling memory usage."""
        self.stop_polling = True
        self.t.join()


def _get_x_bytes(x) -> int:
    if isinstance(x, np.ndarray):
        return x.nbytes
    if sps.issparse(x):
        return sum(
            mat.data.nbytes + mat.indices.nbytes + mat.indptr.nbytes
            for mat in [x, x.x_csr]
        )
    if isinstance(x, tm.CategoricalMatrix):
        return x.indices.nbytes
    if isinstance(x, tm.SplitMatrix):
        return sum(_get_x_bytes(elt) for elt in x.matrices)
    raise NotImplementedError(f"Can't get bytes for matrix of type {type(x)}.")


def _runner(storage, copy_X: Optional[bool]):
    gc.collect()

    P = get_all_problems()["wide-insurance-no-weights-lasso-poisson"]
    dat = P.data_loader(num_rows=100000, storage=storage)

    # Measure how much memory we are using before calling the GLM code
    if isinstance(dat["X"], pd.DataFrame):
        X = dat["X"].to_numpy()
    elif sps.issparse(dat["X"]):
        X = tm.SparseMatrix(dat["X"])
    elif isinstance(dat["X"], tm.SplitMatrix):
        X = dat["X"]

    data_memory = _get_x_bytes(X)

    y = dat["y"]
    del dat
    gc.collect()

    with MemoryPoller() as mp:
        for _ in range(4):
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

        graph = (np.array(mp.memory_usage) - mp.initial_memory) / data_memory
        plt.plot(graph)
        plt.ylabel("Memory (fraction of X)")
        plt.xlabel("Time (1e-4s)")
        plt.savefig(f"performance/memory_{storage}_copy_{copy_X}.png")

        return extra_to_initial_ratio


def _make_memory_usage_plots():
    # These values are around double the empirical extra memory used. They inc
    # They increase from dense->sparse->split->cat, because the matrix itself takes less
    # and less memory to store, so all the temporary vectors of length n_rows start to
    # dominate the memory usage.
    storage_allowed_ratio = {"dense": 0.1, "sparse": 0.45, "cat": 1.3, "split0.1": 0.55}
    for storage, allowed_ratio in storage_allowed_ratio.items():
        for copy_X in [False, True, None]:
            with mp.Pool(1) as p:
                extra_to_initial_ratio = p.starmap(_runner, [(storage, copy_X)])[0]

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


def get_spmv_runtime():
    """
    Get runtime of sparse matrix-vector product.

    Sparse matrix-vector product runtime should be representative of the memory
    bandwidth of the machine. Automatically scale the according to half the
    number of cores since the scipy.sparse implementation is not parallelized
    and glum is parallelized.
    """
    N = 20000000
    diag_data = np.random.rand(5, N)
    mat = sps.spdiags(diag_data, [0, 1, -1, 2, -2], N, N).tocsr()
    v = np.random.rand(N)
    return runtime(lambda: mat.dot(v), 5)[0] / (mp.cpu_count() // 2)


def get_dense_inv_runtime():
    """
    Get runtime of dense matrix inverse.

    Dense matrix multiplication runtime should be representative of the
    floating point performance of the machine.
    """
    N = 1300
    X = np.random.rand(N, N)
    return runtime(lambda: np.linalg.inv(X), 5)[0]


def runtime_checker():
    """
    Run various operations and check that glum doesn't run too much
    slower than operations expected to be similar. This isn't a perfect test
    but it'll raise a red flag if the code has unexpectedly gotten much slower.
    """
    spmv_runtime = get_spmv_runtime()
    dense_inv_runtime = get_dense_inv_runtime()

    what_to_check = [
        ("dense", "narrow-insurance-no-weights-lasso", "poisson", 200000, 1.5),
        ("sparse", "narrow-insurance-weights-l2", "gaussian", 200000, 2.5),
        ("cat", "wide-insurance-no-weights-l2", "gamma", 100000, 2.5),
        ("split0.1", "wide-insurance-offset-lasso", "tweedie-p=1.5", 100000, 3.0),
        ("split0.1", "intermediate-insurance-no-weights-net", "binomial", 200000, 1.0),
    ]

    for storage, problem, distribution, num_rows, limit in what_to_check:
        P = get_all_problems()[problem + "-" + distribution]
        dat = P.data_loader(num_rows=num_rows, storage=storage)

        family = get_sklearn_family(distribution)
        model = GeneralizedLinearRegressor(
            family=family,
            l1_ratio=1.0,
            alpha=0.01,
            copy_X=False,
            force_all_finite=False,
        )
        min_runtime, result = runtime(
            lambda: model.fit(X=dat["X"], y=dat["y"]),  # noqa B023
            5,
        )

        # Let's just guess that we're about half flop-limited and half
        # memory-limited.  This is a decent guess because the sandwich product is
        # mostly flop-limited in the dense case and the dense case generally
        # dominates even when we're using split or categorical. On the other hand,
        # everything besides the sandwich product is probably memory limited.
        denominator = 0.5 * dense_inv_runtime + 0.5 * spmv_runtime
        if min_runtime / denominator > limit:
            warnings.warn(
                f"runtime {min_runtime} is greater than the expected maximum runtime "
                f"of {limit * denominator}"
            )


if __name__ == "__main__":
    # make_memory_usage_plots()
    runtime_checker()
