import multiprocessing as mp
import time
import warnings
from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy import sparse as sps


def _setup_and_fit(X_np, y, distribution, l1_ratio, alpha, standardize, thresh):
    """
    Run in a child process: set up R/glmnet, fit, return results.

    All arguments must be picklable (no rpy2 objects).
    """
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr
    from rpy2.robjects.vectors import FloatVector, IntVector

    # Packages must be pre-installed. Fail fast if missing.
    # Use suppressPackageStartupMessages to avoid noisy loading messages.
    required_pkgs = ["glmnet", "Matrix"]
    if distribution.startswith("tweedie"):
        required_pkgs += ["tweedie", "statmod"]
    for pkg in required_pkgs:
        ro.r(f'suppressPackageStartupMessages(library("{pkg}"))')

    glmnet_pkg = importr("glmnet")
    ro.r["glmnet.control"](epsnr=1e-15)

    # Build R family object
    if distribution in ("gaussian", "binomial", "poisson"):
        family = distribution
    elif distribution == "gamma":
        family = ro.r["Gamma"](link="log")
    elif distribution.startswith("tweedie-p="):
        p = float(distribution.split("tweedie-p=")[1])
        family = ro.r["tweedie"](link_power=0, var_power=p)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    # Convert data + fit (timed: matches what .fit() does in other libs).
    n, p = X_np.shape
    lambda_vec = FloatVector([alpha])

    start = time.time()

    if sps.issparse(X_np):
        matrix_pkg = importr("Matrix")
        coo = sps.coo_matrix(X_np)
        X_r = matrix_pkg.sparseMatrix(
            i=IntVector(coo.row + 1),
            j=IntVector(coo.col + 1),
            x=FloatVector(coo.data),
            dims=IntVector([n, p]),
        )
    else:
        X_f = np.asarray(X_np, dtype=float, order="F")
        X_r = ro.r.matrix(FloatVector(X_f.ravel(order="F")), nrow=n, ncol=p)

    fit = glmnet_pkg.glmnet(
        x=X_r,
        y=FloatVector(y),
        family=family,
        alpha=l1_ratio,
        intercept=True,
        standardize=standardize,
        thresh=thresh,
        **{"lambda": lambda_vec},
    )
    fit_runtime = time.time() - start

    # Extract results
    coef_mat = ro.r["as.matrix"](ro.r["coef"](fit, s=lambda_vec))
    coef_vec = np.asarray(coef_mat).ravel()

    n_iter = None
    try:
        npasses = fit.rx2("npasses")
        if len(npasses) > 0:
            n_iter = int(npasses[0])
    except Exception:
        pass

    return {
        "intercept": float(coef_vec[0]),
        "coef": coef_vec[1:],
        "n_iter": n_iter,
        "fit_runtime": fit_runtime,
    }


def _worker(queue, *args):
    """Target for child process. Must be module-level to be picklable."""
    try:
        queue.put(("ok", _setup_and_fit(*args)))
    except Exception as e:
        queue.put(("error", str(e)))


def _run_with_hard_timeout(args, timeout):
    """
    Run _setup_and_fit in a child process with a hard kill timeout.

    Uses 'spawn' so R is initialized fresh (fork is unsafe with R).
    """
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_worker, args=(queue, *args))
    proc.start()
    proc.join(timeout)

    if proc.is_alive():
        proc.kill()
        proc.join()
        raise TimeoutError(f"glmnet exceeded {timeout}s hard timeout")

    if queue.empty():
        raise RuntimeError("glmnet child process died without returning results")

    status, payload = queue.get_nowait()
    if status == "error":
        raise RuntimeError(f"glmnet failed in child process: {payload}")
    return payload


def glmnet_bench(
    dat: dict[str, Union[np.ndarray, sps.spmatrix]],
    distribution: str,
    alpha: float,
    l1_ratio: float,
    iterations: int,
    standardize: bool = True,
    timeout: Optional[float] = None,
    **kwargs,
):
    """
    Run the glmnet benchmark via rpy2 in a child process.

    Each call spawns a fresh process so that:
      1. R is cleanly initialized (no fork issues).
      2. A hard timeout can kill R mid-computation.
    """
    # Prepare X as a picklable numpy/scipy object
    X = dat["X"]
    if sps.issparse(X):
        X = sps.csc_matrix(X)
    elif isinstance(X, pd.DataFrame):
        X = X.to_numpy(dtype=float)
    elif not isinstance(X, np.ndarray):
        warnings.warn(
            "glmnet requires data as scipy.sparse matrix, pandas dataframe, or "
            "numpy array. Skipping."
        )
        return {}

    y = np.asarray(dat["y"], dtype=float).ravel()

    # glmnet's thresh is a *relative* convergence criterion (scaled internally
    # by the null deviance).  We use a fixed tight value so glmnet actually
    # converges to the same quality as the other libraries.
    thresh = 1e-12

    fit_args = (X, y, distribution, l1_ratio, alpha, standardize, thresh)

    # Run iterations, keeping the fastest.
    # The child process times data conversion + glmnet() (excludes R startup).
    # The hard timeout covers the whole child process (startup + fit).
    successful_runs: list[tuple[float, dict]] = []

    for i in range(iterations):
        try:
            fit_result = _run_with_hard_timeout(fit_args, timeout)
            successful_runs.append((fit_result["fit_runtime"], fit_result))
        except TimeoutError:
            pass
        except Exception as e:
            warnings.warn(f"glmnet iteration {i + 1} failed: {e}")

    if not successful_runs:
        raise TimeoutError(
            f"All {iterations} glmnet iterations exceeded {timeout}s timeout"
        )

    best_time, best_result = min(successful_runs, key=lambda x: x[0])

    return {
        "runtime": best_time,
        "intercept": best_result["intercept"],
        "coef": best_result["coef"],
        "n_iter": best_result["n_iter"],
        "max_iter": None,
    }
