import time
from typing import Dict, Union

import numpy as np
from scipy import sparse as sps

# based on https://github.com/afbujan/admm_lasso/blob/master/lasso_admm.py
# ADMM PAPER: https://web.stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf


def admm_bench(
    dat: Dict[str, Union[np.ndarray, sps.spmatrix]],
    distribution: str,
    alpha: float,
    l1_ratio: float,
    iterations: int = None,
    cv: bool = False,
    **kwargs,
):
    # only gaussian lasso is supported.
    if (
        distribution != "gaussian"
        or l1_ratio != 0
        or iterations != 1
        or cv
        or "offset" in dat.keys()
    ):
        return dict()

    start = time.time()
    A = dat["X"]

    # centering helps a lot!
    means = np.mean(A, axis=0)
    A -= means
    stds = np.std(A, axis=0)

    # add a column of ones for the intercept
    A = np.hstack((A, np.ones((A.shape[0], 1))))

    y = dat["y"]

    benchmark_convergence_tolerance = 1e-8

    abstol = benchmark_convergence_tolerance
    reltol = abstol
    rho_start = 1.0
    relax = 1.5

    # rho update parameters
    mu = 10.0
    tau = 2.0

    # we need to multiply the L1 penalty by the number of rows to match with
    # the sklearn and h2o and glmnet implementations. (this stumped me for a
    # while...)
    alpha = np.full(A.shape[1], A.shape[0] * alpha)

    # make sure we don't regularize the intercept!
    alpha[-1] = 0.0  # type: ignore

    rho = np.full(A.shape[1], rho_start)
    rho[:-1] *= stds
    rho[-1] = 0.0
    update_rho = False

    x = np.zeros(A.shape[1])
    z = np.zeros(A.shape[1])
    u = np.zeros(A.shape[1])

    def soft_threshold(x, kappa):
        return np.sign(x) * np.maximum(np.abs(x) - kappa, 0.0)

    ATA = A.T @ A
    ATy = A.T.dot(y)

    # an eigenvalue decomp allows updating rho from loop to loop
    # without recomputing our decomposition. however, it seems to be slightly
    # worse in terms of conditioning for very poorly conditioned problems
    # some problems where solver_naive succeeds, just never converge for solver_eig
    # eigv, P = np.linalg.eigh(ATA)
    # solver_eig = lambda q: P @ ((P.T @ q) / (eigv + rho))

    # L = np.linalg.cholesky(ATA + np.diag(rho))
    # solver_chol = lambda q: scipy.linalg.cho_solve((L, True), q)
    if update_rho:

        def solver(q):
            return np.linalg.solve(ATA + np.diag(rho), q)

    else:
        ATArhoinv = np.linalg.inv(ATA + np.diag(rho))

        def solver(q):
            return ATArhoinv.dot(q)

    diagnostics = []
    for i in range(200000):

        # update, see section 6.4 in ADMM PAPER for the Lasso update formula
        # This step is the critical expensive step of ADMM. If it's slow,
        # everything will be slow.
        x = solver(ATy + rho * (z - u))

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
        if update_rho:
            if rnorm > mu * snorm:
                rho *= tau
            elif snorm > mu * rnorm:
                rho /= tau

        # primal/dual convergence criteria, see section 3.3.1 in ADMM PAPER
        eps_primal = np.sqrt(A.shape[1]) * abstol + reltol * np.maximum(
            np.linalg.norm(x), np.linalg.norm(-z)
        )
        eps_dual = np.sqrt(A.shape[1]) * abstol + reltol * np.linalg.norm(rho * u)
        diagnostics.append(
            (
                i,
                rho[0],
                rnorm,
                eps_primal,
                snorm,
                eps_dual,
                z[-1],
                np.max(np.abs(z - zold)),
                (z - zold).copy(),
            )
        )
        # print(*diagnostics[-1])
        if rnorm < eps_primal and snorm < eps_dual:
            break

    diagnostics = np.array(diagnostics)
    # step_sizes = diagnostics[:, -2]
    # import matplotlib.pyplot as plt
    # plt.plot(np.log10(step_sizes.astype(np.float64)), '.')
    # plt.tight_layout()

    # steps = diagnostics[:, -1]
    # plt.figure()
    # plt.imshow(np.log10(np.abs(np.vstack([s.astype(np.float64) for s in steps]))))
    # plt.colorbar()
    # plt.tight_layout()
    # plt.show()

    coef = z

    # uncenter the intercept
    coef[-1] -= means.dot(coef[:-1])
    print(coef[0], coef[-1], np.sum(coef == 0))

    result = dict()
    result["intercept"] = coef[-1]
    result["coef"] = coef[:-1]
    result["runtime"] = time.time() - start
    result["n_iter"] = i

    return result
