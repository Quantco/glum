import time
from typing import Dict, Union

import numpy as np
import scipy.linalg
from scipy import sparse as sps

# based on https://github.com/afbujan/admm_lasso/blob/master/lasso_admm.py
# ADMM PAPER: https://web.stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf


def admm_bench(
    dat: Dict[str, Union[np.ndarray, sps.spmatrix]],
    distribution: str,
    alpha: float,
    l1_ratio: float,
):
    # only gaussian lasso is supported.
    if distribution != "gaussian":
        return

    start = time.time()
    A = dat["X"]

    # centering helps a lot!
    means = np.mean(A, axis=0)
    A -= means

    # add a column of ones for the intercept
    A = np.hstack((A, np.ones((A.shape[0], 1))))

    y = dat["y"]

    benchmark_convergence_tolerance = 1e-8

    abstol = benchmark_convergence_tolerance
    reltol = abstol
    rho = np.full(A.shape[1], 1.0)
    relax = 1.5

    # rho update parameters
    mu = 10.0
    tau = 1.1

    # we need to multiply the L1 penalty by the number of rows to match with
    # the sklearn and h2o and glmnet implementations. (this stumped me for a
    # while...)
    alpha = np.full(A.shape[1], A.shape[0] * alpha)

    # make sure we don't regularize the intercept!
    alpha[-1] = 0.0  # type: ignore

    x = np.zeros(A.shape[1])
    z = np.zeros(A.shape[1])
    u = np.zeros(A.shape[1])

    def soft_threshold(x, kappa):
        return np.sign(x) * np.maximum(np.abs(x) - kappa, 0.0)

    ATA = A.T @ A
    ATy = A.T.dot(y)

    def solver_naive(q):
        return np.linalg.solve(ATA + np.diag(rho), q)

    # an eigenvalue decomp allows updating rho from loop to loop
    # without recomputing our decomposition. however, it seems to be slightly
    # worse in terms of conditioning for very poorly conditioned problems
    eigv, P = np.linalg.eigh(ATA)

    def solver_eig(q):
        return P @ ((P.T @ q) / (eigv + rho))

    L = np.linalg.cholesky(ATA + np.diag(rho))

    def solver_chol(q):
        return scipy.linalg.cho_solve((L, True), q)

    for i in range(200000):

        # update, see section 6.4 in ADMM PAPER for the Lasso update formula
        # x = solver_chol(ATy + rho * (z - u))
        x = solver_naive(ATy + rho * (z - u))

        # overrelaxation, see section 3.4.3 in ADMM PAPER
        zold = z.copy()
        x_hat = relax * x + (1.0 - relax) * zold

        kappa = np.where(rho != 0, alpha / rho, 0.0)
        z = soft_threshold(x_hat + u, kappa)
        u += x_hat - z

        rdiff = x - z
        rnorm = np.linalg.norm(rdiff)
        sdiff = -rho * (z - zold)
        snorm = np.linalg.norm(sdiff)

        # we can update rho loop to loop, but it makes the linear system
        # solving more complicated and requires either using the naive or eig
        # solver above. updating rho improves convergence by about 2x generally.
        # see section 3.4.1 in ADMM PAPER
        if rnorm > mu * snorm:
            rho *= tau
        elif snorm > mu * rnorm:
            rho /= tau

        # primal/dual convergence criteria, see section 3.3.1 in ADMM PAPER
        eps_primal = np.sqrt(A.shape[1]) * abstol + reltol * np.maximum(
            np.linalg.norm(x), np.linalg.norm(-z)
        )
        eps_dual = np.sqrt(A.shape[1]) * abstol + reltol * np.linalg.norm(rho * u)
        print(rho[0], rnorm, eps_primal, snorm, eps_dual, z[-1], np.max(z - zold))
        if rnorm < eps_primal and snorm < eps_dual:
            break

    coef = z

    # uncenter the intercept
    coef[-1] -= means.dot(coef[:-1])
    print(coef[0], coef[-1])

    result = dict()
    result["intercept"] = coef[-1]
    result["coef"] = coef[:-1]
    result["runtime"] = time.time() - start
    result["n_iter"] = i

    return result
