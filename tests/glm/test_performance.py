from threading import Thread
from time import sleep

import psutil
from quantcore.glm_benchmarks.cli_run import get_all_libraries, get_all_problems


def get_memory_usage():
    return psutil.Process().memory_info().rss


class MemoryPoller:
    def poll_max_memory_usage(self):
        while not self.stop_polling:
            self.max_memory = max(self.max_memory, get_memory_usage())
            sleep(0.001)

    def __enter__(self):
        self.stop_polling = False
        self.max_memory = 0
        self.initial_memory = get_memory_usage()
        self.t = Thread(target=self.poll_max_memory_usage)
        self.t.start()
        return self

    def __exit__(self, *excargs):
        self.stop_polling = True
        self.t.join()


def test_memory_usage():
    P = get_all_problems()["narrow-insurance-no-weights-lasso-poisson"]
    dat = P.data_loader(num_rows=300000, storage="dense")

    with MemoryPoller() as mp:
        result = get_all_libraries()["quantcore-glm"](
            dat,
            distribution=P.distribution,
            alpha=0.01,
            l1_ratio=1.0,
            iterations=5,
            cv=False,
            diagnostics_level="none",
        )
        print(result)
        extra_to_initial_ratio = (mp.max_memory - mp.initial_memory) / mp.initial_memory
        print(
            mp.initial_memory,
            mp.max_memory,
            extra_to_initial_ratio,
            (mp.max_memory - mp.initial_memory) / 1e6,
        )
        assert extra_to_initial_ratio < 0.5
