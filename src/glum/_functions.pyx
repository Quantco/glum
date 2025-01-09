from cython cimport floating
from cython.parallel import prange

from libc.math cimport M_PI, ceil, exp, floor, lgamma, log

ctypedef fused numeric:
    short
    int
    long
    float
    double

# If an argument is readonly, that will fail with a typical floating[:]
# memoryview. However, const floating[:] causes failures because currently,
# const and fused types (floating) and memoryviews cannot all be combined in
# Cython. https://github.com/cython/cython/issues/1772
cdef fused const_floating1d:
    const float[:]
    const double[:]


# NOTE: Here and below, the factor argument is left for last. That's because it
# will be a single floating point value being passed from Python. In Python,
# all floating point values are 64-bit. As a result, if it's the first
# argument, it will cause the fused types to always evaluate to double.
# Whereas, if it's the last argument, by the time it's encountered, floating
# will have already been set and the 64-bit float will be cast to the correct
# time.
def normal_identity_eta_mu_deviance(
    const_floating1d cur_eta,
    const_floating1d X_dot_d,
    const_floating1d y,
    const_floating1d weights,
    floating[:] eta_out,
    floating[:] mu_out,
    floating factor
):
    cdef int n = cur_eta.shape[0]
    cdef int i
    cdef floating deviance = 0.0
    for i in prange(n, nogil=True):
        eta_out[i] = cur_eta[i] + factor * X_dot_d[i]
        mu_out[i] = eta_out[i]
        deviance += weights[i] * (y[i] - mu_out[i]) ** 2
    return deviance


def normal_identity_rowwise_gradient_hessian(
    const_floating1d y,
    const_floating1d weights,
    const_floating1d eta,
    const_floating1d mu,
    floating[:] gradient_rows_out,
    floating[:] hessian_rows_out
):
    cdef int n = eta.shape[0]
    cdef int i
    for i in prange(n, nogil=True):
        gradient_rows_out[i] = weights[i] * (y[i] - mu[i])
        # Note: hessian_rows_out yields -1 times the true hessian to match
        # the default calculation in _distribution.py
        hessian_rows_out[i] = weights[i]


def normal_log_likelihood(
    const_floating1d y,
    const_floating1d weights,
    const_floating1d mu,
    floating dispersion,
):
    cdef int i  # loop counter
    cdef floating sum_weights  # helper

    cdef int n = y.shape[0]  # loop length
    cdef floating ll = 0.0  # output

    for i in prange(n, nogil=True):
        ll -= weights[i] * (y[i] - mu[i]) ** 2
        sum_weights -= weights[i]

    return ll / (2 * dispersion) + sum_weights * log(2 * M_PI * dispersion) / 2

def normal_deviance(
    const_floating1d y,
    const_floating1d weights,
    const_floating1d mu,
    floating dispersion,
):
    cdef int i  # loop counter

    cdef int n = y.shape[0]  # loop length
    cdef floating D = 0.0  # output

    for i in prange(n, nogil=True):
        D += weights[i] * (y[i] - mu[i]) ** 2

    return D

def poisson_log_eta_mu_deviance(
    const_floating1d cur_eta,
    const_floating1d X_dot_d,
    const_floating1d y,
    const_floating1d weights,
    floating[:] eta_out,
    floating[:] mu_out,
    floating factor
):
    cdef int n = cur_eta.shape[0]
    cdef int i
    cdef floating deviance = 0.0
    for i in prange(n, nogil=True):
        eta_out[i] = cur_eta[i] + factor * X_dot_d[i]
        mu_out[i] = exp(eta_out[i])
        deviance += weights[i] * (y[i] * eta_out[i] - mu_out[i])
    return -2 * deviance

def poisson_log_rowwise_gradient_hessian(
    const_floating1d y,
    const_floating1d weights,
    const_floating1d eta,
    const_floating1d mu,
    floating[:] gradient_rows_out,
    floating[:] hessian_rows_out
):
    cdef int n = eta.shape[0]
    cdef int i
    for i in prange(n, nogil=True):
        gradient_rows_out[i] = weights[i] * (y[i] - mu[i])
        hessian_rows_out[i] = weights[i] * mu[i]

def poisson_log_likelihood(
    const_floating1d y,
    const_floating1d weights,
    const_floating1d mu,
    floating dispersion,
):
    cdef int i  # loop counter

    cdef int n = y.shape[0]  # loop length
    cdef floating ll = 0.0  # output

    for i in prange(n, nogil=True):
        ll -= weights[i] * mu[i]
        if y[i] > 0:
            ll -= weights[i] * (lgamma(1 + y[i]) - y[i] * log(mu[i]))

    return ll

def poisson_deviance(
    const_floating1d y,
    const_floating1d weights,
    const_floating1d mu,
    floating dispersion,
):
    cdef int i  # loop counter

    cdef int n = y.shape[0]  # loop length
    cdef floating D = 0.0  # output

    for i in prange(n, nogil=True):
        D += weights[i] * mu[i]
        if y[i] > 0:
            D += weights[i] * y[i] * (log(y[i]) - log(mu[i]) - 1)

    return 2 * D

def gamma_log_eta_mu_deviance(
    const_floating1d cur_eta,
    const_floating1d X_dot_d,
    const_floating1d y,
    const_floating1d weights,
    floating[:] eta_out,
    floating[:] mu_out,
    floating factor
):
    cdef int n = cur_eta.shape[0]
    cdef int i
    cdef floating deviance = 0.0
    for i in prange(n, nogil=True):
        eta_out[i] = cur_eta[i] + factor * X_dot_d[i]
        mu_out[i] = exp(eta_out[i])
        deviance += weights[i] * (y[i] / mu_out[i] + eta_out[i])
    return 2 * deviance

def gamma_log_rowwise_gradient_hessian(
    const_floating1d y,
    const_floating1d weights,
    const_floating1d eta,
    const_floating1d mu,
    floating[:] gradient_rows_out,
    floating[:] hessian_rows_out
):
    cdef int n = eta.shape[0]
    cdef int i
    for i in prange(n, nogil=True):
        gradient_rows_out[i] = weights[i] * (y[i] / mu[i] - 1)
        hessian_rows_out[i] = weights[i] * (y[i] / mu[i])

def gamma_log_likelihood(
    const_floating1d y,
    const_floating1d weights,
    const_floating1d mu,
    floating dispersion,
):
    cdef int i  # loop counter
    cdef floating ln_y, sum_weights  # helpers

    cdef int n = y.shape[0]  # loop length
    cdef floating ll = 0.0  # output
    cdef floating inv_dispersion = 1 / dispersion
    cdef floating normalization = log(dispersion) * inv_dispersion + lgamma(inv_dispersion)

    for i in prange(n, nogil=True):
        ln_y = log(y[i])
        ll += weights[i] * (inv_dispersion * (ln_y - log(mu[i]) - y[i] / mu[i]) - ln_y)
        sum_weights += weights[i]

    return ll - normalization * sum_weights

def gamma_deviance(
    const_floating1d y,
    const_floating1d weights,
    const_floating1d mu,
    floating dispersion,
):
    cdef int i  # loop counter

    cdef int n = y.shape[0]  # loop length
    cdef floating D = 0.0  # output

    for i in prange(n, nogil=True):
        D += weights[i] * (log(mu[i]) - log(y[i]) + y[i] / mu[i] - 1)

    return 2 * D

def inv_gaussian_log_eta_mu_deviance(
    const_floating1d cur_eta,
    const_floating1d X_dot_d,
    const_floating1d y,
    const_floating1d weights,
    floating[:] eta_out,
    floating[:] mu_out,
    floating factor
):
    cdef int n = cur_eta.shape[0]
    cdef int i  # loop counter
    cdef floating sq_err  # helper
    cdef floating deviance = 0.0  # output

    for i in prange(n, nogil=True):

        eta_out[i] = cur_eta[i] + factor * X_dot_d[i]
        mu_out[i] = exp(eta_out[i])

        sq_err = (y[i] / mu_out[i] - 1) ** 2

        deviance += weights[i] * sq_err / y[i]

    return deviance

def inv_gaussian_log_rowwise_gradient_hessian(
    const_floating1d y,
    const_floating1d weights,
    const_floating1d eta,
    const_floating1d mu,
    floating[:] gradient_rows_out,
    floating[:] hessian_rows_out
):
    cdef int n = eta.shape[0]
    cdef int i  # loop counter

    cdef floating inv_mu, inv_mu2

    for i in prange(n, nogil=True):

        inv_mu = 1 / mu[i]
        inv_mu2 = inv_mu ** 2

        gradient_rows_out[i] = weights[i] * (y[i] * inv_mu2 - inv_mu)
        # Use the FIM instead of the true Hessian, as the latter is not
        # necessarily positive definite.
        hessian_rows_out[i] = weights[i] * inv_mu

def inv_gaussian_log_likelihood(
    const_floating1d y,
    const_floating1d weights,
    const_floating1d mu,
    floating dispersion,
):
    cdef int n = y.shape[0]  # loop length
    cdef int i  # loop counter
    cdef floating sum_weights  # helper
    cdef floating ll = 0.0  # output

    cdef floating sq_err  # helper
    cdef floating inv_dispersion = 1 / (2 * dispersion)  # helper

    for i in prange(n, nogil=True):

        sq_err = (y[i] / mu[i] - 1) ** 2

        ll -= weights[i] * (inv_dispersion * sq_err / y[i] + log(y[i]) * 3 / 2)
        sum_weights -= weights[i]

    return ll + sum_weights * log(inv_dispersion / M_PI)

def inv_gaussian_deviance(
    const_floating1d y,
    const_floating1d weights,
    const_floating1d mu,
    floating dispersion,
):
    cdef int i  # loop counter
    cdef int n = y.shape[0]  # loop length
    cdef floating sq_err  # helper
    cdef floating D = 0.0  # output

    for i in prange(n, nogil=True):
        sq_err = (y[i] / mu[i] - 1) ** 2
        D += weights[i] * sq_err / y[i]

    return D

def tweedie_log_eta_mu_deviance(
    const_floating1d cur_eta,
    const_floating1d X_dot_d,
    const_floating1d y,
    const_floating1d weights,
    floating[:] eta_out,
    floating[:] mu_out,
    floating factor,
    floating p
):
    cdef int n = cur_eta.shape[0]
    cdef int i
    cdef floating deviance = 0.0
    cdef floating mu1mp
    for i in prange(n, nogil=True):
        eta_out[i] = cur_eta[i] + factor * X_dot_d[i]
        mu_out[i] = exp(eta_out[i])
        mu1mp = mu_out[i] ** (1 - p)
        deviance += weights[i] * mu1mp * (mu_out[i] / (2 - p) - y[i] / (1 - p))
    return 2 * deviance

def tweedie_log_rowwise_gradient_hessian(
    const_floating1d y,
    const_floating1d weights,
    const_floating1d eta,
    const_floating1d mu,
    floating[:] gradient_rows_out,
    floating[:] hessian_rows_out,
    floating p
):
    cdef int n = eta.shape[0]
    cdef int i
    cdef floating mu1mp, ymm
    for i in prange(n, nogil=True):
        mu1mp = mu[i] ** (1 - p)
        ymm = y[i] - mu[i]
        gradient_rows_out[i] = weights[i] * mu1mp * ymm
        # This hessian will be positive definite for 1 < p < 2. Don't use it
        # outside that range.
        hessian_rows_out[i] = weights[i] * mu1mp * (mu[i] - (1 - p) * ymm)

def tweedie_deviance(
    const_floating1d y,
    const_floating1d weights,
    const_floating1d mu,
    floating p,
):
    cdef int i  # loop counter
    cdef floating mu1mp, yo1mp  # helpers

    cdef int n = y.shape[0]  # loop length
    cdef floating D = 0.0  # output

    for i in prange(n, nogil=True):

        mu1mp = mu[i] ** (1 - p)
        D += weights[i] * (mu1mp * mu[i]) / (2 - p)

        if y[i] > 0:
            yo1mp = y[i] / (1 - p)
            D += weights[i] * ((y[i] ** (1 - p)) * yo1mp / (2 - p) - yo1mp * mu1mp)

    return 2 * D

def tweedie_log_likelihood(
    const_floating1d y,
    const_floating1d weights,
    const_floating1d mu,
    floating p,
    floating dispersion,
):
    cdef int i  # loop counter

    cdef int n = y.shape[0]  # loop length
    cdef floating ll = 0.0  # output

    for i in prange(n, nogil=True):
        ll += weights[i] * _tweedie_unit_loglikelihood(y[i], mu[i], p, dispersion)

    return ll

cdef floating _tweedie_unit_loglikelihood(floating y, floating mu, floating power, floating dispersion) nogil:

    cdef floating kappa, normalization, theta

    if y == 0:
        return -(mu ** (2 - power)) / (dispersion * (2 - power))
    else:
        theta = mu ** (1 - power)
        kappa = mu * theta / (2 - power)
        theta = theta / (1 - power)
        normalization = _tweedie_normalization(y, power, dispersion)
        return (theta * y - kappa) / dispersion + normalization

cdef floating _tweedie_normalization(floating y, floating power, floating dispersion) nogil:
    # This implementation follows https://doi.org/10.1007/s11222-005-4070-y.

    cdef int j, j_lower, j_upper

    cdef floating j_max = exp((2 - power) * log(y) - log(dispersion) - log(2 - power))
    cdef floating w_max = _log_w_j(y, power, dispersion, j_max)
    cdef floating w_summand = 0.0

    j_lower, j_upper = _sum_limits(y, power, dispersion, j_max)

    for j in range(j_lower, j_upper + 1):
        w_summand += exp(_log_w_j(y, power, dispersion, j) - w_max)

    return w_max + log(w_summand) - log(y)

cdef (int, int) _sum_limits(floating y, floating power, floating dispersion, floating j_max) nogil:

    cdef floating w_lower

    cdef floating j_lower = 1.0
    cdef floating j_upper = ceil(j_max)
    cdef floating w_upper = _log_w_j(y, power, dispersion, j_upper)

    cdef floating w_crt = _log_w_j(y, power, dispersion, j_max) - 37
    cdef floating w_one = _log_w_j(y, power, dispersion, 1)

    if w_one <= w_crt:
        j_lower = floor(j_max)
        w_lower = _log_w_j(y, power, dispersion, j_lower)
        while w_lower >= w_crt:
            j_lower -= 1
            w_lower = _log_w_j(y, power, dispersion, j_lower)
    while w_upper >= w_crt:
        j_upper += 1
        w_upper = _log_w_j(y, power, dispersion, j_upper)

    return int(j_lower), int(j_upper)

cdef floating _log_w_j(floating y, floating power, floating dispersion, numeric j) nogil:
    cdef floating alpha = (2 - power) / (1 - power)
    return j * _log_z(y, power, dispersion) - lgamma(1 + j) - lgamma(-alpha * j)

cdef floating _log_z(floating y, floating power, floating dispersion) nogil:
    cdef floating alpha = (2 - power) / (1 - power)
    return (
        alpha * log(power - 1)
        - alpha * log(y)
        - (1 - alpha) * log(dispersion)
        - log(2 - power)
    )

def binomial_logit_eta_mu_deviance(
    const_floating1d cur_eta,
    const_floating1d X_dot_d,
    const_floating1d y,
    const_floating1d weights,
    floating[:] eta_out,
    floating[:] mu_out,
    floating factor
):
    cdef int n = cur_eta.shape[0]
    cdef int i
    cdef floating unit_deviance
    cdef floating deviance = 0.0
    cdef floating expposeta, expnegeta
    for i in prange(n, nogil=True):
        eta_out[i] = cur_eta[i] + factor * X_dot_d[i]
        # When eta is positive, we want to use formulas that depend on
        # exp(-eta), but when eta is negative we want to use formulas that
        # depend on exp(eta), rederived based on the suggestions here:
        # http://fa.bianp.net/blog/2013/numerical-optimizers-for-logistic-regression/
        # That article assumes y in {-1, +1} whereas we use y in {0, 1}. Thus
        # the difference in formulas.
        # The same approach is used in sklearn.linear_model.LogisticRegression
        # and in LIBLINEAR
        if eta_out[i] > 0:
            expnegeta = exp(-eta_out[i])
            unit_deviance = weights[i] * (y[i] * eta_out[i] - eta_out[i] - log(1 + expnegeta))
            mu_out[i] = 1 / (1 + expnegeta)
        else:
            expposeta = exp(eta_out[i])
            unit_deviance = weights[i] * (y[i] * eta_out[i] - log(1 + expposeta))
            mu_out[i] = expposeta / (expposeta + 1)
        deviance += unit_deviance
    return -2 * deviance

def binomial_logit_rowwise_gradient_hessian(
    const_floating1d y,
    const_floating1d weights,
    const_floating1d eta,
    const_floating1d mu,
    floating[:] gradient_rows_out,
    floating[:] hessian_rows_out
):
    cdef int n = eta.shape[0]
    cdef int i
    for i in prange(n, nogil=True):
        gradient_rows_out[i] = weights[i] * (y[i] - mu[i])
        hessian_rows_out[i] = weights[i] * mu[i] * (1 - mu[i])

def negative_binomial_log_eta_mu_deviance(
    const_floating1d cur_eta,
    const_floating1d X_dot_d,
    const_floating1d y,
    const_floating1d weights,
    floating[:] eta_out,
    floating[:] mu_out,
    floating factor,
    floating theta
):
    cdef int n = cur_eta.shape[0]
    cdef int i
    cdef floating deviance = 0.0
    cdef floating r = 1.0 / theta  # helper

    for i in prange(n, nogil=True):
        eta_out[i] = cur_eta[i] + factor * X_dot_d[i]
        mu_out[i] = exp(eta_out[i])
        deviance += weights[i] * (-y[i] * eta_out[i] + (y[i] + r) * log(mu_out[i] + r))
    return 2 * deviance

def negative_binomial_log_rowwise_gradient_hessian(
    const_floating1d y,
    const_floating1d weights,
    const_floating1d eta,
    const_floating1d mu,
    floating[:] gradient_rows_out,
    floating[:] hessian_rows_out,
    floating theta
):
    cdef int n = eta.shape[0]
    cdef int i
    for i in prange(n, nogil=True):
        gradient_rows_out[i] = weights[i] * (y[i] - mu[i]) / (1.0 + theta * mu[i])
        hessian_rows_out[i] = weights[i] * mu[i]  / (1.0 + theta * mu[i])

def negative_binomial_log_likelihood(
    const_floating1d y,
    const_floating1d weights,
    const_floating1d mu,
    floating theta,
    floating dispersion,
):
    cdef int i  # loop counter

    cdef int n = y.shape[0]  # loop length
    cdef floating ll = 0.0  # output
    cdef floating r = 1.0 / theta  # helper

    for i in prange(n, nogil=True):
        ll += weights[i] * (
            y[i] * log(theta * mu[i]) -
            (y[i] + r) * log(1 + theta * mu[i]) +
            lgamma(y[i] + r) -
            lgamma(r) -
            lgamma(y[i] + 1.0)
        )

    return ll

def negative_binomial_deviance(
    const_floating1d y,
    const_floating1d weights,
    const_floating1d mu,
    floating theta,
):
    cdef int i  # loop counter

    cdef int n = y.shape[0]  # loop length
    cdef floating D = 0.0  # output
    cdef floating r = 1.0 / theta  # helper

    for i in prange(n, nogil=True):
        D += - weights[i] * (y[i] + r) * log((y[i] + r) / (mu[i] + r))
        if y[i] > 0:
            D += weights[i] * y[i] * log(y[i] / mu[i])

    return 2 * D