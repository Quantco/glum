"""Minimal reproducer: concurrent scipy.linalg.solve(assume_a="pos") in N processes."""

import faulthandler
import multiprocessing as mp
import sys
import time

import numpy as np


def work(idx, n, seconds):
    faulthandler.enable()
    from scipy import linalg

    rng = np.random.default_rng(idx)
    deadline = time.time() + seconds
    count = 0
    while time.time() < deadline:
        a = rng.normal(size=(n + 50, n))
        h = a.T @ a + np.eye(n)
        b = rng.normal(size=n)
        linalg.solve(h, b, assume_a="pos")
        count += 1
    return count


if __name__ == "__main__":
    nproc, n, seconds = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
    import scipy

    print("numpy", np.__version__, "scipy", scipy.__version__, flush=True)
    np.show_config()
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=work, args=(i, n, seconds)) for i in range(nproc)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    codes = [p.exitcode for p in procs]
    print("exit codes:", codes, flush=True)
    sys.exit(0 if all(c == 0 for c in codes) else 1)
