# cython: boundscheck=False, wraparound=False, cdivision=True

from cython cimport floating
from cython.parallel import prange

from libc.math cimport exp, log

def poisson_log_line_search_update(floating la, floating[:] eta, floating[:] X_dot_d, floating[:] y, floating[:] weights, floating[:] eta_wd_out, floating[:] mu_wd_out):
    cdef int n = eta.shape[0]
    cdef int i
    cdef floating unit_deviance
    cdef floating deviance = 0.0
    for i in prange(n, nogil=True):
        eta_wd_out[i] = eta[i] + la * X_dot_d[i]
        mu_wd_out[i] = exp(eta_wd_out[i])
        if y[i] == 0:
            unit_deviance = 2 * (-y[i] + mu_wd_out[i])
        else:
            unit_deviance = 2 * ((y[i] * (log(y[i]) - eta_wd_out[i] - 1)) + mu_wd_out[i])
        deviance += weights[i] * unit_deviance
    return deviance

def poisson_log_line_search_deviance(floating[:] y, floating[:] eta, floating[:] mu, floating[:] weights):
    cdef int n = eta.shape[0]
    cdef int i
    cdef floating unit_deviance
    cdef floating deviance = 0.0
    for i in prange(n, nogil=True):
        if y[i] == 0:
            unit_deviance = 2 * (-y[i] + mu[i])
        else:
            unit_deviance = 2 * ((y[i] * (log(y[i]) - eta[i] - 1)) + mu[i])
        deviance += weights[i] * unit_deviance
    return deviance

def poisson_log_gradient_hessian_update(floating[:] y, floating[:] weights, floating[:] eta, bint update_mu, floating[:] mu_out, floating[:] gradient_rows_out, floating[:] fisher_W_out):
    cdef int n = eta.shape[0]
    cdef int i
    cdef floating unit_variance, sigma_inv, d1, d1_sigma_inv
    for i in prange(n, nogil=True):
        if update_mu:
            mu_out[i] = exp(eta[i])
        unit_variance = mu_out[i]
        sigma_inv = weights[i] / unit_variance
        d1 = mu_out[i]
        d1_sigma_inv = d1 * sigma_inv
        gradient_rows_out[i] = d1_sigma_inv * (y[i] - mu_out[i])
        fisher_W_out[i] = d1 * d1_sigma_inv

