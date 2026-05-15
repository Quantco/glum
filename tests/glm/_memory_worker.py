"""
Subprocess worker for calibration tests in ``test_memory.py``.

Runs a single GLM fit, measures peak RSS via a sidecar polling
thread, and prints a JSON payload to stdout for the parent test
process to consume.

This must run in its own process — Python's memory allocator pool
and GC timing make in-process repeated RSS measurements unreliable.

Uses ``resource.getrusage`` (stdlib) on Unix to read RSS, avoiding
an extra ``psutil`` dependency.  Falls back to a heuristic on
Windows (where ``resource`` is unavailable) — calibration tests are
skipped there.

Invocation:
    python tests/glm/_memory_worker.py \\
        --n 100000 --p_num 10 --p_cat_levels '[5, 5, 100]'
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import threading
import time
from typing import List

import numpy as np
import pandas as pd

import glum


def _read_rss_bytes() -> int:
    """
    Return current RSS in bytes via ``resource.getrusage``.

    On Linux ``ru_maxrss`` is in kibibytes; on macOS it's in bytes.
    Windows lacks ``resource``; the calibration tests skip on Windows.
    """
    if platform.system() == "Windows":  # pragma: no cover
        raise RuntimeError(
            "RSS measurement requires `resource` (Unix-only). "
            "Calibration tests are not supported on Windows."
        )
    import resource
    rss_native = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        # macOS: ru_maxrss is in bytes
        return int(rss_native)
    # Linux: ru_maxrss is in kibibytes (1024 bytes)
    return int(rss_native) * 1024


def _make_dataset(n: int, p_num: int, p_cat_levels: List[int]):
    """Build a Poisson dataset with the given shape."""
    rng = np.random.default_rng(0)
    cols = {}
    for i in range(p_num):
        cols[f"num_{i}"] = rng.standard_normal(n).astype(np.float64)
    for i, levels in enumerate(p_cat_levels):
        cols[f"cat_{i}"] = pd.Categorical(
            rng.integers(0, levels, size=n)
        )
    df = pd.DataFrame(cols)

    # A modest signal so the fit converges quickly.
    if p_num > 0:
        mu = np.exp(0.5 + 0.1 * df["num_0"])
    else:
        mu = np.full(n, 1.5)
    y = rng.poisson(mu).astype(float)
    return df, y


def _poll_rss_peak(stop_flag: List[bool], peak_holder: List[int],
                   interval_s: float = 0.005):
    """Background-thread poller that tracks max RSS seen.

    Note: ``ru_maxrss`` is monotonic (it's a *maximum*), so polling it
    is a bit redundant — but we keep the polling pattern so we can swap
    in a non-monotonic backend (psutil) later without restructuring.
    """
    while not stop_flag[0]:
        rss = _read_rss_bytes()
        if rss > peak_holder[0]:
            peak_holder[0] = rss
        time.sleep(interval_s)


def measure(n: int, p_num: int, p_cat_levels: List[int]) -> dict:
    """Run a fit and report RSS measurements."""
    baseline_rss = _read_rss_bytes()

    # Build dataset
    df, y = _make_dataset(n, p_num, p_cat_levels)
    after_data_rss = _read_rss_bytes()

    # Start sidecar polling
    peak = [_read_rss_bytes()]
    stop = [False]
    poller = threading.Thread(
        target=_poll_rss_peak, args=(stop, peak), daemon=True,
    )
    poller.start()

    # Fit — use irls-cd as the canonical default solver
    glm = glum.GeneralizedLinearRegressor(
        family="poisson", alpha=0.01, solver="irls-cd",
    )
    glm.fit(df, y)

    stop[0] = True
    poller.join(timeout=2.0)

    # ru_maxrss is monotonic so this captures the true peak even if
    # the poller missed a moment.
    peak_rss = max(peak[0], _read_rss_bytes())
    return {
        "n":                      n,
        "p_num":                  p_num,
        "p_cat_levels":           p_cat_levels,
        "baseline_MB":            baseline_rss / 1e6,
        "after_data_MB":          after_data_rss / 1e6,
        "peak_MB":                peak_rss / 1e6,
        "peak_above_baseline_MB": (peak_rss - baseline_rss) / 1e6,
        "data_only_MB":           (after_data_rss - baseline_rss) / 1e6,
        "fit_only_MB":            (peak_rss - after_data_rss) / 1e6,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n",            type=int, required=True)
    parser.add_argument("--p_num",        type=int, required=True)
    parser.add_argument("--p_cat_levels", type=str, default="[]")
    args = parser.parse_args()

    p_cat_levels = json.loads(args.p_cat_levels)
    result = measure(args.n, args.p_num, p_cat_levels)

    # Print exactly one JSON line on stdout for the parent to parse.
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
