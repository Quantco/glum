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
    cdef floating unit_deviance
    cdef floating deviance = 0.0
    for i in prange(n, nogil=True):
        eta_out[i] = cur_eta[i] + factor * X_dot_d[i]
        mu_out[i] = eta_out[i]
        # Note: deviance is equal to -2 times the true log likelihood to match
        # the default calculation using unit_deviance in _distribution.py
        # True log likelihood: -1/2 * (y - mu)**2
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
    cdef floating unit_deviance
    cdef floating deviance = 0.0
    for i in prange(n, nogil=True):
        eta_out[i] = cur_eta[i] + factor * X_dot_d[i]
        mu_out[i] = exp(eta_out[i])
        # True log likelihood: y * eta - mu
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
    cdef floating unit_deviance
    cdef floating deviance = 0.0
    for i in prange(n, nogil=True):
        eta_out[i] = cur_eta[i] + factor * X_dot_d[i]
        mu_out[i] = exp(eta_out[i])
        # True log likelihood: -(y / mu + eta)
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
    cdef floating unit_deviance
    cdef floating deviance = 0.0
    cdef floating mu1mp
    for i in prange(n, nogil=True):
        eta_out[i] = cur_eta[i] + factor * X_dot_d[i]
        mu_out[i] = exp(eta_out[i])
        mu1mp = mu_out[i] ** (1 - p)
        deviance += weights[i] * mu1mp * (
            mu_out[i] / (2 - p) - y[i] / (1 - p)
        )
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
