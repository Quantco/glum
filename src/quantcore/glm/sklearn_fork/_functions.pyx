# cython: boundscheck=False, wraparound=False, cdivision=True

from cython cimport floating
from cython.parallel import prange

from libc.math cimport exp, log, fmax

import numpy as np

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
def normal_identity_eta_mu_loglikelihood(
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
    cdef floating unit_loglikelihood
    cdef floating loglikelihood = 0.0
    for i in prange(n, nogil=True):
        eta_out[i] = cur_eta[i] + factor * X_dot_d[i]
        mu_out[i] = eta_out[i]
        # Note: loglikelihood is equal to -2 times the true log likelihood to match
        # the default calculation using unit_deviance in _distribution.py
        # True log likelihood: -1/2 * (y - mu)**2
        loglikelihood += weights[i] * (y[i] - mu_out[i]) ** 2
    return loglikelihood

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

def poisson_log_eta_mu_loglikelihood(
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
    cdef floating unit_loglikelihood
    cdef floating loglikelihood = 0.0
    for i in prange(n, nogil=True):
        eta_out[i] = cur_eta[i] + factor * X_dot_d[i]
        mu_out[i] = exp(eta_out[i])
        # True log likelihood: y * eta - mu
        loglikelihood += weights[i] * -2 * (y[i] * eta_out[i] - mu_out[i])
    return loglikelihood

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

def gamma_log_eta_mu_loglikelihood(
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
    cdef floating unit_loglikelihood
    cdef floating loglikelihood = 0.0
    for i in prange(n, nogil=True):
        eta_out[i] = cur_eta[i] + factor * X_dot_d[i]
        mu_out[i] = exp(eta_out[i])
        # True log likelihood: -(y / mu + eta)
        loglikelihood += weights[i] * 2 * (y[i] / mu_out[i] + eta_out[i])
    return loglikelihood

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
        hessian_rows_out[i] = weights[i]

def tweedie_log_eta_mu_loglikelihood(
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
    cdef floating unit_loglikelihood
    cdef floating loglikelihood = 0.0
    for i in prange(n, nogil=True):
        eta_out[i] = cur_eta[i] + factor * X_dot_d[i]
        mu_out[i] = exp(eta_out[i])
        # No nice expression for likelihood, so derived from unit deviance
        loglikelihood += weights[i] * 2 * (fmax(y[i], 0) ** (2 - p) / ((1 - p) * (2 - p)) \
        - y[i] * mu_out[i] ** (1 - p) / (1 - p) + mu_out[i] ** (2 - p) / (2 - p))
    return loglikelihood

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
    for i in prange(n, nogil=True):
        gradient_rows_out[i] = weights[i] * mu[i] ** (1 - p) * (y[i] - mu[i])
        hessian_rows_out[i] = weights[i] * mu[i] ** (2 - p)

def binomial_logit_eta_mu_loglikelihood(
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
    cdef floating unit_loglikelihood
    cdef floating loglikelihood = 0.0
    for i in prange(n, nogil=True):
        eta_out[i] = cur_eta[i] + factor * X_dot_d[i]
        mu_out[i] = 1 / (1 + exp(-eta_out[i]))
        # Clipping is used to match the mu calculation in _link.py
        if mu_out[i] > 1 - 1e-10:
            mu_out[i] = 1 - 1e-10
        elif mu_out[i] < 1e-20:
            mu_out[i] = 1e-20
        # True log likelihood: log(mu) - eta * (1 - y)
        loglikelihood += weights[i] * (-2 * (y[i] * log(mu_out[i]) + (1 - y[i]) * log(1 - mu_out[i])))
    return loglikelihood

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
    # Clipping is used to match the eta calculation in _distribution.py
    cdef floating mu_unclipped
    cdef floating eta_clipped
    cdef floating max_float_for_exp = np.log(np.finfo(eta.base.dtype).max / 10)
    for i in prange(n, nogil=True):
        mu_unclipped = 1 / (1 + exp(-eta[i]))
        if eta[i] > max_float_for_exp:
            eta_clipped = max_float_for_exp
        elif eta[i] < -max_float_for_exp:
            eta_clipped = -max_float_for_exp
        else:
            eta_clipped = eta[i]
        gradient_rows_out[i] = weights[i] * mu_unclipped * (1 - mu_unclipped) * \
        (exp(eta_clipped) + 2 + exp(-eta_clipped)) * (y[i] - mu[i])
        hessian_rows_out[i] = weights[i] * mu_unclipped * (1 - mu_unclipped)