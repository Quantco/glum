import gc
import multiprocessing as mp
import time
from threading import Thread
from typing import Optional

import numpy as np
import pandas as pd
import psutil
import pytest
import quantcore.matrix as mx
import scipy.sparse as sps
from quantcore.glm_benchmarks.cli_run import get_all_problems
from test_sklearn import GLM_SOLVERS

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


def runner(storage):
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


@pytest.fixture(scope="module")
def X():
    P = get_all_problems()["wide-insurance-no-weights-lasso-poisson"]
    dat = P.data_loader(num_rows=100000, storage="dense")
    return dat["X"]


@pytest.mark.parametrize(
    "storage, allowed_ratio",
    [("dense", 0.1), ("sparse", 0.45), ("cat", 1.3), ("split0.1", 0.55)],
)
@pytest.mark.slow
def test_memory_usage(storage, allowed_ratio):
    with mp.Pool(1) as p:
        extra_to_initial_ratio = p.map(runner, [storage])[0]
    assert extra_to_initial_ratio < allowed_ratio


def run_test_X_copied_iff_copy_X(
    X, solver, fit_intercept: bool, use_offset: bool, x_format, copy_X: Optional[bool],
):
    convert_x_fn_d = {
        "dense": mx.DenseMatrix,
        "sparse": lambda x: mx.SparseMatrix(sps.csc_matrix(x)),
        "split": lambda x: mx.split_matrix.csc_to_split(sps.csc_matrix(x)),
    }
    convert_x_fn = convert_x_fn_d[x_format]

    # Not exactly true, as some formats will take up more space
    X = convert_x_fn(X)
    x_size = get_x_bytes(X)
    offset = np.zeros(X.shape[0]) if use_offset else None
    y = np.random.random(X.shape[0])

    glm = GeneralizedLinearRegressor(
        family="normal",
        link="identity",
        fit_intercept=fit_intercept,
        solver=solver,
        gradient_tol=1e-7,
        copy_X=copy_X,
    )

    with MemoryPoller() as mp:
        glm.fit(X, y, offset=offset)

    excess = mp.max_memory - mp.initial_memory

    return excess, x_size


@pytest.mark.slow
@pytest.mark.parametrize("solver", [GLM_SOLVERS[0]])
@pytest.mark.parametrize("fit_intercept", [False, True])
@pytest.mark.parametrize("use_offset", [False, True])
@pytest.mark.parametrize("x_format", ["dense", "sparse", "split"])
@pytest.mark.parametrize("copy_X", [False, None, True])
def test_X_copied_iff_copy_X(
    X,
    solver,
    fit_intercept: bool,
    use_offset: bool,
    x_format: str,
    copy_X: Optional[bool],
):
    with mp.Pool(1) as p:
        excess, x_size = p.starmap(
            run_test_X_copied_iff_copy_X,
            [(X, solver, fit_intercept, use_offset, x_format, copy_X)],
        )[0]

    if copy_X:
        assert excess > x_size
    else:
        assert excess < x_size
