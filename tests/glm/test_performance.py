import gc
from threading import Thread
from time import sleep

import numpy as np
import pandas as pd
import psutil
import pytest
from quantcore.glm_benchmarks.cli_run import get_all_problems

from quantcore.glm import GeneralizedLinearRegressor


def get_memory_usage():
    return psutil.Process().memory_info().rss


class MemoryPoller:
    def poll_max_memory_usage(self):
        while not self.stop_polling:
            self.memory_usage.append(get_memory_usage())
            self.max_memory = max(self.max_memory, self.memory_usage[-1])
            sleep(0.001)

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


@pytest.mark.parametrize("storage", ["dense", "sparse"])
def test_memory_usage(storage):
    gc.collect()
    pre_data_memory = get_memory_usage()
    P = get_all_problems()["narrow-insurance-no-weights-lasso-poisson"]
    dat = P.data_loader(num_rows=100000, storage=storage)
    X = dat["X"].to_numpy() if isinstance(dat["X"], pd.DataFrame) else dat["X"]
    y = dat["y"]
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

        data_memory = mp.initial_memory - pre_data_memory
        excess_memory_used = mp.max_memory - mp.initial_memory
        extra_to_initial_ratio = excess_memory_used / data_memory

        graph = np.array(mp.memory_usage) - pre_data_memory
        import matplotlib.pyplot as plt

        plt.plot(graph)
        plt.show()
        print(extra_to_initial_ratio)

        assert extra_to_initial_ratio < 0.5
